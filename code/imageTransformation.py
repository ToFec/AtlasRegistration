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

        if isinstance(images, list):
            sampledImageList = [
                torch.nn.functional.grid_sample(
                    img[None, ...],
                    meshes[idx, None, ...],
                    padding_mode=paddMode,
                    align_corners=alignCorners,
                    mode=interpolationType,
                )
                for idx, img in enumerate(images)
            ]
            sampledImage = torch.cat(sampledImageList, dim=0)
        else:
            sampledImage = torch.nn.functional.grid_sample(
                images, meshes, padding_mode=paddMode, align_corners=alignCorners, mode=interpolationType
            )
        return sampledImage

    def combineMeshesAndFlowField(self, meshes: torch.Tensor, flowField: torch.Tensor) -> torch.Tensor:
        """
        Compose atlas-normalized displacement with the mesh mapping to obtain a single sampling grid.

        meshes:     (B, 3, X, Y, Z) subject image normalized index coordinates in [-1, 1]
        flowField:  (B, 3, X, Y, Z) displacement in atlas normalized coordinates in [-1, 1]

        Returns:
            meshes + A · flowField, where A maps atlas-normalized basis vectors to subject
            image normalized index vectors derived from the mesh.
        """
        # Columns of A are half the span along each axis because normalized coordinates span length 2.
        vec_x = meshes[:, :, -1, 0, 0] - meshes[:, :, 0, 0, 0]
        vec_y = meshes[:, :, 0, -1, 0] - meshes[:, :, 0, 0, 0]
        vec_z = meshes[:, :, 0, 0, -1] - meshes[:, :, 0, 0, 0]
        A = torch.stack((vec_x, vec_y, vec_z), dim=2) * 0.5  # (B, 3, 3)

        # Transform atlas-normalized displacement into subject image normalized index coordinates.
        delta = torch.einsum("bij,bjxyz->bixyz", A, flowField.to(meshes.dtype))  # (B, 3, X, Y, Z)
        return meshes + delta
