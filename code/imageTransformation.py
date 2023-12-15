"""
Created on Apr 28, 2023

@author: fechter
"""

import torch
import numpy as np


class Transformation(torch.nn.Module):
    def __init__(self, shape=None):
        super(Transformation, self).__init__()
        self.setIdentityTransform(shape)

    def setIdentityTransform(self, shape, dtype=np.float32):
        identityTransform = None
        if shape:
            dim = len(shape)

            if dim == 3:
                imgshape = shape[-1:]
                id = np.mgrid[0 : imgshape[0]]
            elif dim == 4:
                imgshape = shape[-2:]
                id = np.mgrid[0 : imgshape[0], 0 : imgshape[1]]
            elif dim == 5:
                imgshape = shape[-3:]
                id = np.mgrid[0 : imgshape[0], 0 : imgshape[1], 0 : imgshape[2]]
            else:
                raise ValueError("Only dimensions 1-3 are currently supported for the identity map")

            id = np.array(id.astype(dtype))
            if dim == 3:
                id = id.reshape(1, imgshape[0])  # add a dummy first index
            spacing = 1.0 / (np.array(imgshape) - 1)

            for d in range(len(imgshape)):
                id[d] *= spacing[d]
                id[d] = id[d] * 2 - 1

            identityTransform = torch.from_numpy(id.astype(np.float32))
        self.register_buffer("identityTransform", identityTransform)

    def getDeformationField(self, flowField):
        if self.identityTransform is None:
            self.setIdentityTransform(flowField.shape)
            self.identityTransform = self.identityTransform.to(flowField)

        return self.identityTransform + flowField

    def sampleImage(self, images, meshes, alignCorners=True):
        meshes = torch.moveaxis(meshes, 1, -1)
        meshes = meshes.flip(-1)
        sampledImage = torch.nn.functional.grid_sample(images, meshes, padding_mode="zeros", align_corners=alignCorners)
        return sampledImage


class Bilinear(Transformation):
    """
    Spatial transform function for 1D, 2D, and 3D. In BCXYZ format (this IS the format used in the current toolbox).
    """

    def __init__(self, shape=None, zero_boundary=False, using_scale=False):
        """
        Constructor

        :param ndim: (int) spatial transformation of the transform
        """
        super(Bilinear, self).__init__(shape)
        self.zero_boundary = "zeros" if zero_boundary else "border"
        self.using_scale = using_scale
        """ scale [-1,1] image intensity into [0,1], this is due to the zero boundary condition we may use here """

    def forward_stn(self, input1, input2):
        input2_ordered = torch.zeros_like(input2)
        input2_ordered[:, 0, ...] = input2[:, 2, ...]
        input2_ordered[:, 1, ...] = input2[:, 1, ...]
        input2_ordered[:, 2, ...] = input2[:, 0, ...]

        output = torch.nn.functional.grid_sample(
            input1, input2_ordered.permute([0, 2, 3, 4, 1]), padding_mode=self.zero_boundary, align_corners=True
        )
        return output

    def forward(self, input1, input2):
        """
        Perform the actual spatial transform

        :param input1: image in BCXYZ format
        :param input2: spatial transform in BdimXYZ format
        :return: spatially transformed image in BCXYZ format
        """
        if self.using_scale:
            output = self.forward_stn((input1 + 1) / 2, input2)
            # print(STNVal(output, ini=-1).sum())
            return output * 2 - 1
        else:
            output = self.forward_stn(input1, input2)
            # print(STNVal(output, ini=-1).sum())
            return output
