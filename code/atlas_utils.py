import torch
import torch.nn as nn
import numpy as np
import SimpleITK as sitk
from losses import LNCCLoss, DiceLossMultiClass, BendingEnergyLoss, GradLoss, NCCLoss
import random
import pytorch_lightning as pl
from functools import partial
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from torchio.data.io import _read_itk_matrix


def convertDistanceMapToLabelMap(distanceMap, ignoreBackground=False):
    labelMap = torch.zeros_like(distanceMap)
    valToAdd = 0
    if ignoreBackground:
        valToAdd = 1

    for channel in range(0, distanceMap.shape[1]):
        labelMap[:, channel, ...][distanceMap[:, channel, ...] <= 0.0] = (channel + valToAdd)
    return labelMap


def resampleSitkImage(image, transform, nn=False, reference=None):
    if reference is not None:
        reference_image = reference
    else:
        reference_image = image
    if nn:
        interpolator = sitk.sitkNearestNeighbor
    else:
        interpolator = sitk.sitkCosineWindowedSinc
    default_value = 0.0
    return sitk.Resample(image, reference_image, transform, interpolator, default_value)

    # as an alternative we could also use the scaling matrix of the singular value decomposition


def getScaling(npTransformationMatrix):
    scalingX = np.linalg.norm(npTransformationMatrix[:, 0])
    scalingY = np.linalg.norm(npTransformationMatrix[:, 1])
    scalingZ = np.linalg.norm(npTransformationMatrix[:, 2])
    return scalingX, scalingY, scalingZ


def normalize(npTransformationMatrix):
    scalingX, scalingY, scalingZ = getScaling(npTransformationMatrix)
    npTransformationMatrix[:, 0] = npTransformationMatrix[:, 0] / scalingX
    npTransformationMatrix[:, 1] = npTransformationMatrix[:, 1] / scalingY
    npTransformationMatrix[:, 2] = npTransformationMatrix[:, 2] / scalingZ


def addScaling(npTransformationMatrix, scaleX, scaleY, scaleZ):
    npTransformationMatrix[:, 0] = npTransformationMatrix[:, 0] * scaleX
    npTransformationMatrix[:, 1] = npTransformationMatrix[:, 1] * scaleY
    npTransformationMatrix[:, 2] = npTransformationMatrix[:, 2] * scaleZ


def _from_itk_convention(matrix):
    """LPS to RAS"""
    FLIPXY = np.diag([-1, -1, 1, 1])
    matrix = np.dot(matrix, FLIPXY)
    matrix = np.dot(FLIPXY, matrix)
    matrix = np.linalg.inv(matrix)
    return matrix


def itkToRasMatrix(transform):
    """Read an affine transform in ITK's .tfm format"""
    parameters = transform.GetParameters()
    rotation_parameters = parameters[:9]
    rotation_matrix = np.array(rotation_parameters).reshape(3, 3)
    translation_parameters = parameters[9:]
    translation_vector = np.array(translation_parameters).reshape(3, 1)
    matrix = np.hstack([rotation_matrix, translation_vector])
    homogeneous_matrix_lps = np.vstack([matrix, [0, 0, 0, 1]])
    homogeneous_matrix_ras = _from_itk_convention(homogeneous_matrix_lps)
    return homogeneous_matrix_ras


def applyRigidRegistrationToImgHeader(image: sitk.Image, transform: sitk.Transform):
    transformationMatrix = transform.GetParameters()

    npTransformationMatrix = np.asarray(
        [
            (transformationMatrix[0], transformationMatrix[1], transformationMatrix[2], transformationMatrix[9]),
            (transformationMatrix[3], transformationMatrix[4], transformationMatrix[5], transformationMatrix[10]),
            (transformationMatrix[6], transformationMatrix[7], transformationMatrix[8], transformationMatrix[11]),
            (0, 0, 0, 1),
        ]
    )
    npTransformationMatrix = np.linalg.inv(npTransformationMatrix)

    imgOrigin = image.GetOrigin()
    imgSpacing = image.GetSpacing()
    imgDir = image.GetDirection()

    imageOrientationPatient = np.array(
        [
            (imgDir[0], imgDir[1], imgDir[2], 0),
            (imgDir[3], imgDir[4], imgDir[5], 0),
            (imgDir[6], imgDir[7], imgDir[8], 0),
            (0, 0, 0, 1),
        ]
    )

    imgOriginNew = npTransformationMatrix @ np.append(imgOrigin, 1)
    imgOriginNew = imgOriginNew[0:3].tolist()

    newImageOrientationPatient = npTransformationMatrix @ imageOrientationPatient

    scalingX, scalingY, scalingZ = getScaling(newImageOrientationPatient)
    imgSpacingNew = (imgSpacing[0] * scalingX, imgSpacing[1] * scalingY, imgSpacing[2] * scalingZ)

    normalize(newImageOrientationPatient)

    newImageOrientationPatient = newImageOrientationPatient[0:3, 0:3].ravel().tolist()

    image.SetOrigin(imgOriginNew)
    image.SetDirection(newImageOrientationPatient)
    image.SetSpacing(imgSpacingNew)


def createSignedDistanceMap(sitkLabel, ignoreBackground=False):
    array = sitk.GetArrayViewFromImage(sitkLabel)
    uniqueValues = np.unique(array)
    if ignoreBackground:
        uniqueValues = uniqueValues[uniqueValues != 0]
    distanceMaps = []
    for uniqueVal in uniqueValues:
        tmpImage = sitkLabel == uniqueVal
        distanceMaps.append(sitk.SignedDanielssonDistanceMap(tmpImage, useImageSpacing=True))
    distanceMap4D = sitk.JoinSeries(distanceMaps)
    distanceMapTensor = sitk.GetArrayFromImage(distanceMap4D)
    distanceMapTensor[distanceMapTensor > 0.01] = 0.01
    return distanceMapTensor


def loadDefField(filename):
    defFieldITK = sitk.ReadImage(str(filename))
    defFieldSpacing = defFieldITK.GetSpacing()
    defFieldDirection = defFieldITK.GetDirection()

    defField = sitk.GetArrayFromImage(defFieldITK)
    defField[..., 0] = (defField[..., 0] / defFieldSpacing[0]) * defFieldDirection[
        0
    ]  # should be sign of direction or reorient to standard direction?
    defField[..., 1] = (defField[..., 1] / defFieldSpacing[1]) * defFieldDirection[4]
    defField[..., 2] = (defField[..., 2] / defFieldSpacing[2]) * defFieldDirection[8]

    defField[..., 0] = defField[..., 0] / ((defField.shape[2] - 1) / 2.0)
    defField[..., 1] = defField[..., 1] / ((defField.shape[1] - 1) / 2.0)
    defField[..., 2] = defField[..., 2] / ((defField.shape[0] - 1) / 2.0)

    defField = np.expand_dims(defField, axis=0)
    defField = torch.from_numpy(defField.astype(np.float32))
    defField = defField.permute([0, 4, 3, 2, 1])

    return defField


def getMeshSpacing(mesh):
    spacing0 = torch.sqrt(torch.sum(torch.pow((mesh[:, 0, 0, 0] - mesh[:, 1, 0, 0]), 2)))
    spacing1 = torch.sqrt(torch.sum(torch.pow((mesh[:, 0, 0, 0] - mesh[:, 0, 1, 0]), 2)))
    spacing2 = torch.sqrt(torch.sum(torch.pow((mesh[:, 0, 0, 0] - mesh[:, 0, 0, 1]), 2)))

    return torch.tensor((spacing0, spacing1, spacing2))


def saveDefField(filename, defField, origin, spacing, direction):
    defField = defField[0, ...].detach()
    defField = defField.permute([3, 2, 1, 0])

    defField[..., 0] = defField[..., 0] * ((defField.shape[2] - 1) / 2.0)
    defField[..., 1] = defField[..., 1] * ((defField.shape[1] - 1) / 2.0)
    defField[..., 2] = defField[..., 2] * ((defField.shape[0] - 1) / 2.0)

    defField[..., 0] = (defField[..., 0] * spacing[0]) * direction[0]
    defField[..., 1] = (defField[..., 1] * spacing[1]) * direction[4]
    defField[..., 2] = (defField[..., 2] * spacing[2]) * direction[8]

    defDataToSave = sitk.GetImageFromArray(defField, isVector=True)

    defDataToSave.SetSpacing(spacing)
    defDataToSave.SetOrigin(origin)
    defDataToSave.SetDirection(direction)

    sitk.WriteImage(defDataToSave, filename)


def saveImageTensor(imageData, imageName, origin, spacing, direction):
    imageDataToSave = imageData.squeeze(0).squeeze(0).permute([2, 1, 0])
    saveImage(imageDataToSave, imageName, origin, spacing, direction)


def saveImage(imageData, imageName, origin, spacing, direction):
    sitkImage = sitk.GetImageFromArray(imageData)
    sitkImage.SetOrigin(origin)
    sitkImage.SetDirection(direction)
    sitkImage.SetSpacing(spacing)
    sitk.WriteImage(sitkImage, imageName)


def setSeeds(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    pl.seed_everything(seed, workers=True)

    torch.autograd.set_detect_anomaly(True)


def getOptimizer(optimizerType):
    if optimizerType == "sgd":
        return partial(torch.optim.SGD, momentum=0.9, nesterov=True)
    if optimizerType == "adam":
        return torch.optim.Adam
    return torch.optim.AdamW


def get_test_list():
    with open("/playpen-raid1/zpd/remote/MAS/Data/OAI-ZIB/test.txt", "r") as f_test:
        test_list = list(f_test.read().splitlines())

    final_test_list = test_list[-100:]
    final_test_pair_list = []
    for i in range(len(final_test_list)):
        for j in range(i + 1, len(final_test_list)):
            final_test_pair_list.append((final_test_list[i], final_test_list[j]))

    return final_test_list, final_test_pair_list


def identity_map(sz, dtype=np.float32):
    """
    Returns an identity map.

    :param sz: just the spatial dimensions, i.e., XxYxZ
    :param spacing: list with spacing information [sx,sy,sz]
    :param dtype: numpy data-type ('float32', 'float64', ...)
    :return: returns the identity map of dimension dimxXxYxZ
    """
    dim = len(sz)
    if dim == 1:
        id = np.mgrid[0 : sz[0]]
    elif dim == 2:
        id = np.mgrid[0 : sz[0], 0 : sz[1]]
    elif dim == 3:
        id = np.mgrid[0 : sz[0], 0 : sz[1], 0 : sz[2]]
    else:
        raise ValueError("Only dimensions 1-3 are currently supported for the identity map")
    id = np.array(id.astype(dtype))
    if dim == 1:
        id = id.reshape(1, sz[0])  # add a dummy first index
    spacing = 1.0 / (np.array(sz) - 1)

    for d in range(dim):
        id[d] *= spacing[d]
        id[d] = id[d] * 2 - 1

    return torch.from_numpy(id.astype(np.float32))


def not_normalized_identity_map(sz):
    """
    Returns an identity map.

    :param sz: just the spatial dimensions, i.e., XxYxZ
    :param spacing: list with spacing information [sx,sy,sz]
    :param dtype: numpy data-type ('float32', 'float64', ...)
    :return: returns the identity map of dimension dimxXxYxZ
    """
    dim = len(sz)
    if dim == 1:
        id = np.mgrid[0 : sz[0]]
    elif dim == 2:
        id = np.mgrid[0 : sz[0], 0 : sz[1]]
    elif dim == 3:
        id = np.mgrid[0 : sz[0], 0 : sz[1], 0 : sz[2]]
    else:
        raise ValueError("Only dimensions 1-3 are currently supported for the identity map")
    # id= id*2-1
    return torch.from_numpy(id.astype(np.float32))


def gen_identity_map(img_sz, resize_factor=1.0, normalized=True):
    """
    given displacement field,  add displacement on grid field  todo  now keep for reproduce  this function will be disabled in the next release, replaced by spacing version
    """
    dim = 3
    if isinstance(resize_factor, list):
        img_sz = [int(img_sz[i] * resize_factor[i]) for i in range(dim)]
    else:
        img_sz = [int(img_sz[i] * resize_factor) for i in range(dim)]
    if normalized:
        grid = identity_map(img_sz)
    else:
        grid = not_normalized_identity_map(img_sz)
    return grid


def gen_identity_ap():
    """
    get the idenityt affine parameter

    :return:
    """
    affine_identity = torch.zeros(12).cuda()
    affine_identity[0] = 1.0
    affine_identity[4] = 1.0
    affine_identity[8] = 1.0

    return affine_identity


def get_sim_loss(warped, target, loss_name):
    """
    compute the similarity loss

    :param loss_fn: the loss function
    :param output: the warped image
    :param target: the target image
    :return: the similarity loss average on batch
    """
    # loss_fn = self.ncc if self.epoch < self.epoch_activate_extern_loss else loss_fn
    if loss_name == "LNCC":
        sim_criterion = LNCCLoss()
    elif loss_name == "NCC":
        sim_criterion = NCCLoss()
    elif loss_name == "SSD":
        sim_criterion = nn.MSELoss(size_average=True)
    else:
        raise ValueError("Undefined loss for similarity measure")
    sim_loss = sim_criterion(warped, target)

    return sim_loss / warped.shape[0]


def get_pair_sim_loss(warped_img, loss_name):
    batch_size = warped_img.shape[0]
    if loss_name == "LNCC":
        pair_criterion = LNCCLoss()
    elif loss_name == "NCC":
        pair_criterion = NCCLoss()
    elif loss_name == "SSD":
        pair_criterion = nn.MSELoss(size_average=True)
    else:
        raise ValueError("Undefined loss for similarity measure")
    pair_loss = pair_criterion(warped_img[: int(batch_size / 2)], warped_img[int(batch_size / 2) :])

    return pair_loss / (batch_size / 2.0)


def get_pair_sim_loss_image_space(warped_img1, warped_img2, loss_name):
    if loss_name == "LNCC":
        pair_criterion = LNCCLoss()
    elif loss_name == "NCC":
        pair_criterion = NCCLoss()
    elif loss_name == "SSD":
        pair_criterion = nn.MSELoss(size_average=True)
    else:
        raise ValueError("Undefined loss for similarity measure")
    pair_loss = pair_criterion(warped_img1, warped_img2)

    return pair_loss / warped_img1.shape[0]


def get_atlas_seg_loss(warped_segs, atlas_segs):
    """
    compute the similarity loss

    :param loss_fn: the loss function
    :param output: the warped image
    :param target: the target image
    :return: the similarity loss average on batch
    """
    batch_size = warped_segs.shape[0]
    seg_criterion = DiceLossMultiClass(n_class=5, weight_type="Uniform", no_bg=True)
    seg_loss = seg_criterion(warped_segs, atlas_segs)

    return seg_loss / batch_size


def get_sym_loss(rec_src_phi_warped, rec_tar_phi_warped, n_batch):
    """
    compute the symmetric loss,
    :math: `loss_{sym} = \|(\varphi^{s t})^{-1} \circ(\varphi^{t s})^{-1}-i d\|_{2}^{2}`

    :param rec_phiWarped:the transformation map, including two direction ( s-t, t-s in batch dimension)
    :return: mean(`loss_{sym}`)
    """
    src_A_map = rec_src_phi_warped[:n_batch]
    src_B_map = rec_src_phi_warped[n_batch:]
    tar_B_map = rec_tar_phi_warped[:n_batch]
    tar_A_map = rec_tar_phi_warped[n_batch:]

    return torch.mean((src_A_map - tar_A_map) ** 2 + (src_B_map - tar_B_map) ** 2)


def get_reg_loss(disp_flow):
    reg_criterion = BendingEnergyLoss()
    reg_loss = reg_criterion(disp_flow)

    return reg_loss


def get_first_order_reg_loss(disp_flow):
    reg_criterion = GradLoss(penalty="l2")
    reg_loss = reg_criterion(disp_flow)

    return reg_loss


def jacobianDeterminant(deform_field, spacing):
    """
    jacobian determinant of a displacement field.
    NB: to compute the spatial gradients, we use np.gradient.
    Parameters:
        disp: 2D or 3D displacement field of size [*vol_shape, nb_dims],
              where vol_shape is of len nb_dims
    Returns:
        jacobian determinant (scalar)
    """

    # check inputs
    deform_map_np = deform_field.permute([0, 2, 3, 4, 1]).squeeze().detach().cpu().numpy()
    volshape = deform_map_np.shape[:-1]
    nb_dims = len(volshape)
    assert len(volshape) in (2, 3), "flow has to be 2D or 3D"

    # compute gradients
    # specify the voxel spacing!!!
    J = np.gradient(deform_map_np, *[*spacing, 1.0])

    # 3D flow
    if nb_dims == 3:
        dx = J[0]  # (deform_map_np[1:, ...] - deform_map_np[:-1, ...]) / spacing[0]
        # dx = dx[:, :-1, :-1, :]
        dy = J[1]  # (deform_map_np[:, 1:, ...] - deform_map_np[:, :-1, ...]) / spacing[1]
        # dy = dy[:-1, :, :-1, :]
        dz = J[2]  # (deform_map_np[:, :, 1:, ...] - deform_map_np[:, :, :-1, ...]) / spacing[2]  #
        # dz = dz[:-1, :-1, ...]

        # compute jacobian components
        Jdet0 = dx[..., 0] * (dy[..., 1] * dz[..., 2] - dy[..., 2] * dz[..., 1])
        Jdet1 = dx[..., 1] * (dy[..., 0] * dz[..., 2] - dy[..., 2] * dz[..., 0])
        Jdet2 = dx[..., 2] * (dy[..., 0] * dz[..., 1] - dy[..., 1] * dz[..., 0])

        # p0 = dx[..., 0] * dy[..., 1] * dz[..., 2]
        # p1 = dx[..., 1] * dy[..., 2] * dz[..., 0]
        # p2 = dx[..., 2] * dy[..., 0] * dz[..., 1]
        #
        # m0 = dx[..., 2] * dy[..., 1] * dz[..., 0]
        # m1 = dx[..., 0] * dy[..., 2] * dz[..., 1]
        # m2 = dx[..., 1] * dy[..., 0] * dz[..., 2]
        #
        # det = 1.0 + p0 + p1 + p2 - m0 - m1 - m2

        return 1.0 + Jdet0 - Jdet1 + Jdet2

    else:  # must be 2
        dfdx = J[0]
        dfdy = J[1]

        return 1.0 + dfdx[..., 0] * dfdy[..., 1] - dfdy[..., 0] * dfdx[..., 1]


def save_updated_atlas(atlas_img, atlas_seg, save_atlas_img_name, save_atlas_est_name, save_atlas_prob_name):
    torch.save(atlas_seg, save_atlas_prob_name)
    atlas_img_np = atlas_img.detach().squeeze().cpu().numpy()
    atlas_seg_np = atlas_seg.detach().squeeze().cpu().numpy()
    atlas_est_np = torch.max(atlas_seg, 1)[1].detach().squeeze().cpu().numpy()
    tmp_img = sitk.ReadImage(
        "/playpen-raid1/zpd/remote/MAS/Data/OAI-ZIB/Nifti_rescaled_2Left_downsample/9001104_image.nii.gz"
    )
    tmp_seg = sitk.ReadImage(
        "/playpen-raid1/zpd/remote/MAS/Data/OAI-ZIB/Nifti_rescaled_2Left_downsample/9001104_masks.nii.gz"
    )
    atlas_img_nii = sitk.GetImageFromArray(atlas_img_np.astype("float32"))
    atlas_img_nii.CopyInformation(tmp_img)
    sitk.WriteImage(atlas_img_nii, save_atlas_img_name)
    atlas_est_nii = sitk.GetImageFromArray(atlas_est_np.astype("float32"))
    atlas_est_nii.CopyInformation(tmp_seg)
    sitk.WriteImage(atlas_est_nii, save_atlas_est_name)


def setMatmulPrecision(matMulPrecision):
    torch.set_float32_matmul_precision(matMulPrecision)


def plot_grad_flow(named_parameters):
    """Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.

    Usage: Plug this function in Trainer class after loss.backwards() as
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow"""
    ave_grads = []
    max_grads = []
    layers = []
    for n, p in named_parameters:
        if (p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads) + 1, lw=2, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom=-0.001, top=np.max(max_grads))
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend(
        [Line2D([0], [0], color="c", lw=4), Line2D([0], [0], color="b", lw=4), Line2D([0], [0], color="k", lw=4)],
        ["max-gradient", "mean-gradient", "zero-gradient"],
    )


def plot_weights(named_parameters):
    """Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.

    Usage: Plug this function in Trainer class after loss.backwards() as
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow"""
    ave_grads = []
    max_grads = []
    layers = []
    for n, p in named_parameters:
        if (p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.detach().abs().mean())
            max_grads.append(p.detach().abs().max())
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads) + 1, lw=2, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom=-0.001, top=np.max(max_grads))
    plt.xlabel("Layers")
    plt.ylabel("average weight")
    plt.title("Weights")
    plt.grid(True)
    plt.legend(
        [Line2D([0], [0], color="c", lw=4), Line2D([0], [0], color="b", lw=4), Line2D([0], [0], color="k", lw=4)],
        ["max-weight", "mean-weight", "zero-weight"],
    )
