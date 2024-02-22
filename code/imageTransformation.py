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

    def sampleImage(self, images, meshes, alignCorners=True, paddMode="border", interpolationType="bilinear"):
        meshes = torch.moveaxis(meshes, 1, -1)
        meshes = meshes.flip(-1)
        sampledImage = torch.nn.functional.grid_sample(
            images, meshes, padding_mode=paddMode, align_corners=alignCorners, mode=interpolationType
        )
        return sampledImage

    def combineMeshesAndFlowField(self, meshes, flowField):
        normVec0 = torch.nn.functional.normalize(meshes[:, :, -1, 0, 0] - meshes[:, :, 0, 0, 0])
        normVec1 = torch.nn.functional.normalize(meshes[:, :, 0, -1, 0] - meshes[:, :, 0, 0, 0])
        normVec2 = torch.nn.functional.normalize(meshes[:, :, 0, 0, -1] - meshes[:, :, 0, 0, 0])
        orientationMatrices = torch.cat((normVec0, normVec1, normVec2), 1).reshape(-1, 3, 3)
        orientationMatrix = torch.inverse(orientationMatrices)

        scaling = torch.zeros_like(orientationMatrix)
        scaling[:, 0, 0] = torch.linalg.vector_norm(meshes[:, :, 0, 0, 0] - meshes[:, :, -1, 0, 0], dim=1) / 2.0
        scaling[:, 1, 1] = torch.linalg.vector_norm(meshes[:, :, 0, 0, 0] - meshes[:, :, 0, -1, 0], dim=1) / 2.0
        scaling[:, 2, 2] = torch.linalg.vector_norm(meshes[:, :, 0, 0, 0] - meshes[:, :, 0, 0, -1], dim=1) / 2.0

        combinedMatrix = torch.matmul(orientationMatrix, scaling)

        tmp = torch.moveaxis(flowField, 1, -1)
        a = tmp.reshape(tmp.shape[0], -1, 3)
        c = torch.matmul(combinedMatrix, a.moveaxis(-1, -2))
        c = c.moveaxis(-2, -1)
        newField = c.reshape(tmp.shape).moveaxis(-1, 1)

        return meshes + newField
