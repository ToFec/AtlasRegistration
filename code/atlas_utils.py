import torch
import numpy as np
import SimpleITK as sitk
import random
import pytorch_lightning as pl
from functools import partial
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from imageTransformation import Transformation


def getMissingCorrespondence(target_flow, source_flow, transformation):
    targetFlowDefField = transformation.getDeformationField(target_flow)
    warpedSourceFlow = transformation.sampleImage(source_flow, targetFlowDefField)
    targetFlowMinusWarpedSourceFlow = warpedSourceFlow + source_flow
    return targetFlowMinusWarpedSourceFlow


def segmentMisssingCorrespondences(pos_flow, neg_flow, transformation=None):
    if transformation is None:
        transformation = Transformation()

    negFlowMinusWarpedPosFlow = getMissingCorrespondence(neg_flow, pos_flow, transformation)

    meshStepLength0 = 2 / (negFlowMinusWarpedPosFlow.shape[2] - 1)
    meshStepLength1 = 2 / (negFlowMinusWarpedPosFlow.shape[3] - 1)
    meshStepLength2 = 2 / (negFlowMinusWarpedPosFlow.shape[4] - 1)
    meshLenghtOf1Unit = torch.linalg.norm(torch.tensor((meshStepLength0, meshStepLength1, meshStepLength2)))
    allowedDeviation = meshLenghtOf1Unit * 2

    negFlowMinusWarpedPosFlowNorm = torch.linalg.norm(negFlowMinusWarpedPosFlow, dim=1)
    negFlowMinusWarpedPosFlowNorm = torch.nn.functional.avg_pool3d(
        torch.nn.functional.avg_pool3d(negFlowMinusWarpedPosFlowNorm, kernel_size=5, stride=1, padding=2),
        kernel_size=5,
        stride=1,
        padding=2,
    )
    segmentationNegFlowMinusWarpedPosFlow = (negFlowMinusWarpedPosFlowNorm > allowedDeviation).short()

    posFlowMinusWarpedNegFlow = getMissingCorrespondence(pos_flow, neg_flow, transformation)
    posFlowMinusWarpedNegFlowNorm = torch.linalg.norm(posFlowMinusWarpedNegFlow, dim=1)
    posFlowMinusWarpedNegFlowNorm = torch.nn.functional.avg_pool3d(
        torch.nn.functional.avg_pool3d(posFlowMinusWarpedNegFlowNorm, kernel_size=5, stride=1, padding=2),
        kernel_size=5,
        stride=1,
        padding=2,
    )

    segmentationPosFlowMinusWarpedNegFlow = (posFlowMinusWarpedNegFlowNorm > allowedDeviation).short()

    # return segmentationPosFlowMinusWarpedNegFlow, segmentationNegFlowMinusWarpedPosFlow
    return posFlowMinusWarpedNegFlowNorm, negFlowMinusWarpedPosFlowNorm


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
    transform = transform.Downcast()
    dimension = transform.GetDimension()
    matrix = np.eye(dimension + 1)

    if getattr(transform, "GetMatrix", None) is not None:
        rotation = np.array(transform.GetMatrix()).reshape((dimension, dimension))
        matrix[:dimension, :dimension] = rotation

    # transformationMatrix = transform.GetParameters()

    translation = np.array(transform.GetTranslation())
    matrix[:dimension, dimension] = translation

    # npTransformationMatrix = np.asarray(
    #     [
    #         (transformationMatrix[0], transformationMatrix[1], transformationMatrix[2], transformationMatrix[9]),
    #         (transformationMatrix[3], transformationMatrix[4], transformationMatrix[5], transformationMatrix[10]),
    #         (transformationMatrix[6], transformationMatrix[7], transformationMatrix[8], transformationMatrix[11]),
    #         (0, 0, 0, 1),
    #     ]
    # )

    npTransformationMatrix = np.linalg.inv(matrix)

    imgOrigin = np.array(image.GetOrigin())
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

    imgOriginNew = npTransformationMatrix @ np.append(imgOrigin - transform.GetCenter(), 1)
    imgOriginNew = (imgOriginNew[0:3] + transform.GetCenter()).tolist()

    newImageOrientationPatient = npTransformationMatrix @ imageOrientationPatient

    scalingX, scalingY, scalingZ = getScaling(newImageOrientationPatient)
    imgSpacingNew = (imgSpacing[0] * scalingX, imgSpacing[1] * scalingY, imgSpacing[2] * scalingZ)

    normalize(newImageOrientationPatient)

    newImageOrientationPatient = newImageOrientationPatient[0:3, 0:3].ravel().tolist()

    image.SetOrigin(imgOriginNew)
    image.SetDirection(newImageOrientationPatient)
    image.SetSpacing(imgSpacingNew)


def customCollateTensorFunction(batch, *, collate_fn_map):
    return batch


def roundToHighestPosition(arr):
    log10 = torch.floor(torch.log10(torch.abs(arr)))
    base = 10**log10
    rounded = torch.floor(arr / base) * base
    return rounded


def convertDistanceMapToLabelMap(distanceMap):
    labelMap = torch.floor(distanceMap) - 1
    return labelMap


def createSignedDistanceMap(sitkLabel, maxValue=None):
    array = sitk.GetArrayViewFromImage(sitkLabel)
    uniqueValues = np.unique(array)
    array = array + 1
    distanceMaps = []
    for uniqueVal in uniqueValues:
        tmpImage = sitkLabel == uniqueVal
        distanceMaps.append(sitk.SignedDanielssonDistanceMap(tmpImage, useImageSpacing=True))
    distanceMap4D = sitk.JoinSeries(distanceMaps)
    distanceMapTensor = sitk.GetArrayFromImage(distanceMap4D)
    # the following two lines could be used to reduce memory load with distance maps
    # the positive part is truncated and the channels are merged
    distanceMapTensor[distanceMapTensor > 0.0] = 0.0
    distanceMapTensor = np.min(distanceMapTensor, axis=0, keepdims=True)
    distanceMapTensor = np.abs(distanceMapTensor)
    if maxValue is None:
        maxValue = np.ceil(distanceMapTensor.max())
    distanceMapTensor = np.clip(distanceMapTensor,0,maxValue) / (maxValue + 0.0001)
    array = array# * maxValue
    distanceMapTensor[0, ...] = distanceMapTensor[0, ...] + array
    return distanceMapTensor


def loadDefField(filename):
    defFieldITK = sitk.ReadImage(str(filename))
    defFieldSpacing = defFieldITK.GetSpacing()
    defFieldDirection = defFieldITK.GetDirection()

    defField = sitk.GetArrayFromImage(defFieldITK)
    defField[..., 0] = (defField[..., 0] / defFieldSpacing[0]) * defFieldDirection[
        0
    ]  # should be sign of direction or reorient to standard direction?
    defField[..., 1] = defField[..., 1] / defFieldSpacing[1] * defFieldDirection[4]
    defField[..., 2] = defField[..., 2] / defFieldSpacing[2] * defFieldDirection[8]

    defField[..., 0] = defField[..., 0] / ((defField.shape[2] - 1) / 2.0)
    defField[..., 1] = defField[..., 1] / ((defField.shape[1] - 1) / 2.0)
    defField[..., 2] = defField[..., 2] / ((defField.shape[0] - 1) / 2.0)

    defField = np.expand_dims(defField, axis=0)
    defField = torch.from_numpy(defField.astype(np.float32))
    defField = defField.permute([0, 4, 3, 2, 1])

    return defField


# deprecated
def getMeshSpacing(mesh):
    spacing0 = torch.sqrt(torch.sum(torch.pow((mesh[:, 0, 0, 0] - mesh[:, 1, 0, 0]), 2)))
    spacing1 = torch.sqrt(torch.sum(torch.pow((mesh[:, 0, 0, 0] - mesh[:, 0, 1, 0]), 2)))
    spacing2 = torch.sqrt(torch.sum(torch.pow((mesh[:, 0, 0, 0] - mesh[:, 0, 0, 1]), 2)))

    return torch.tensor((spacing0, spacing1, spacing2))


def scaleAndOrientDeffield(defField, spacing, direction):
    defField = defField[0, ...].detach().clone()
    defField = defField.permute([3, 2, 1, 0])

    defField[..., 0] = defField[..., 0] * ((defField.shape[2] - 1) / 2.0)
    defField[..., 1] = defField[..., 1] * ((defField.shape[1] - 1) / 2.0)
    defField[..., 2] = defField[..., 2] * ((defField.shape[0] - 1) / 2.0)

    defField[..., 0] = defField[..., 0] * spacing[0] * direction[0]
    defField[..., 1] = defField[..., 1] * spacing[1] * direction[4]
    defField[..., 2] = defField[..., 2] * spacing[2] * direction[8]
    defField = defField.permute([3, 2, 1, 0])
    return defField


def saveDefField(filename, defField, origin, spacing, direction):
    defField = defField[0, ...].detach().clone()
    defField = defField.permute([3, 2, 1, 0])

    defField[..., 0] = defField[..., 0] * ((defField.shape[2] - 1) / 2.0)
    defField[..., 1] = defField[..., 1] * ((defField.shape[1] - 1) / 2.0)
    defField[..., 2] = defField[..., 2] * ((defField.shape[0] - 1) / 2.0)

    defField[..., 0] = defField[..., 0] * spacing[0] * direction[0]
    defField[..., 1] = defField[..., 1] * spacing[1] * direction[4]
    defField[..., 2] = defField[..., 2] * spacing[2] * direction[8]

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


# all but (jacobian_det[jacobian_det != 0.0] = 1.0 / jacobian_det[jacobian_det != 0.0])
# equal to sitk implementation; we do 1/x because sitk gives values above 1 in compressed areas
# and values below 1 in expanded areas
def jacobianDeterminant(deform_field, spacing=(1.0, 1.0, 1.0)):
    dx = (deform_field[:, :, 2:, :, :] - deform_field[:, :, :-2, :, :]) / (spacing[0] * 2)
    dy = (deform_field[:, :, :, 2:, :] - deform_field[:, :, :, :-2, :]) / (spacing[1] * 2)
    dz = (deform_field[:, :, :, :, 2:] - deform_field[:, :, :, :, :-2]) / (spacing[2] * 2)

    # Pad the gradients to match the original size
    dx = torch.nn.functional.pad(dx, (0, 0, 0, 0, 1, 1))
    dy = torch.nn.functional.pad(dy, (0, 0, 1, 1, 0, 0))
    dz = torch.nn.functional.pad(dz, (1, 1, 0, 0, 0, 0))

    jacobian_det = (
        (1 + dx[:, 0, None]) * (1 + dy[:, 1, None]) * (1 + dz[:, 2, None])
        + dx[:, 1, None] * dy[:, 2, None] * dz[:, 0, None]
        + dx[:, 2, None] * dy[:, 0, None] * dz[:, 1, None]
        - (1 + dx[:, 0, None]) * dy[:, 2, None] * dz[:, 1, None]
        - dx[:, 1, None] * dy[:, 0, None] * (1 + dz[:, 2, None])
        - dx[:, 2, None] * (1 + dy[:, 1, None]) * dz[:, 0, None]
    )
    #jacobian_det[jacobian_det != 0.0] = 1.0 / jacobian_det[jacobian_det != 0.0]
    return jacobian_det


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
