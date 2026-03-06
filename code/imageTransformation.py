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
        Combine normalized mesh coordinates with a deformation field given in world (physical) coordinates.

        meshes:     (B, 3, X, Y, Z) normalized index coordinates in [-1, 1]
        flowField:  (B, 3, X, Y, Z) world displacement vectors (physical space, mm)

        Returns:
            Deformed mesh in normalized index space.
        """

        # ---------------------------------------------------------------------
        # OLD IMPLEMENTATION (kept for reference)
        #
        # normVec0 = torch.nn.functional.normalize(meshes[:, :, -1, 0, 0] - meshes[:, :, 0, 0, 0])
        # normVec1 = torch.nn.functional.normalize(meshes[:, :, 0, -1, 0] - meshes[:, :, 0, 0, 0])
        # normVec2 = torch.nn.functional.normalize(meshes[:, :, 0, 0, -1] - meshes[:, :, 0, 0, 0])
        # orientationMatrices = torch.cat((normVec0, normVec1, normVec2), 1).reshape(-1, 3, 3)
        # orientationMatrix = torch.inverse(orientationMatrices)
        #
        # scaling = torch.zeros_like(orientationMatrix)
        # scaling[:, 0, 0] = torch.linalg.vector_norm(meshes[:, :, 0, 0, 0] - meshes[:, :, -1, 0, 0], dim=1) / 2.0
        # scaling[:, 1, 1] = torch.linalg.vector_norm(meshes[:, :, 0, 0, 0] - meshes[:, :, 0, -1, 0], dim=1) / 2.0
        # scaling[:, 2, 2] = torch.linalg.vector_norm(meshes[:, :, 0, 0, 0] - meshes[:, :, 0, 0, -1], dim=1) / 2.0
        #
        # combinedMatrix = torch.matmul(orientationMatrix, scaling)
        #
        # tmp = torch.moveaxis(flowField, 1, -1)
        # a = tmp.reshape(tmp.shape[0], -1, 3)
        # c = torch.matmul(combinedMatrix, a.moveaxis(-1, -2))
        # c = c.moveaxis(-2, -1)
        # newField = c.reshape(tmp.shape).moveaxis(-1, 1)
        #
        # return meshes + newField
        # ---------------------------------------------------------------------

        batch_size: int = meshes.shape[0]
        spatial_shape = meshes.shape[2:]  # (X, Y, Z)
        device = meshes.device
        dtype = meshes.dtype

        # --- Reconstruct direction vectors from mesh ---
        vec_x = meshes[:, :, -1, 0, 0] - meshes[:, :, 0, 0, 0]
        vec_y = meshes[:, :, 0, -1, 0] - meshes[:, :, 0, 0, 0]
        vec_z = meshes[:, :, 0, 0, -1] - meshes[:, :, 0, 0, 0]

        length_x = torch.linalg.vector_norm(vec_x, dim=1, keepdim=True)
        length_y = torch.linalg.vector_norm(vec_y, dim=1, keepdim=True)
        length_z = torch.linalg.vector_norm(vec_z, dim=1, keepdim=True)

        dir_x = vec_x / length_x
        dir_y = vec_y / length_y
        dir_z = vec_z / length_z

        direction_matrix = torch.stack((dir_x, dir_y, dir_z), dim=2)  # (B, 3, 3)

        voxel_counts = torch.tensor(
            [spatial_shape[0] - 1, spatial_shape[1] - 1, spatial_shape[2] - 1],
            device=device,
            dtype=dtype,
        )

        half_lengths = torch.stack(
            (
                torch.linalg.vector_norm(vec_x, dim=1),
                torch.linalg.vector_norm(vec_y, dim=1),
                torch.linalg.vector_norm(vec_z, dim=1),
            ),
            dim=1,
        ) / 2.0

        spacing = half_lengths * 2.0 / voxel_counts  # spacing per axis

        spacing_matrix = torch.zeros((batch_size, 3, 3), device=device, dtype=dtype)
        spacing_matrix[:, 0, 0] = spacing[:, 0]
        spacing_matrix[:, 1, 1] = spacing[:, 1]
        spacing_matrix[:, 2, 2] = spacing[:, 2]

        ds_matrix = torch.matmul(direction_matrix, spacing_matrix)  # D·S
        inv_ds_matrix = torch.inverse(ds_matrix)

        # --- World → voxel ---
        flow_flat = flowField.permute(0, 2, 3, 4, 1).reshape(batch_size, -1, 3)
        flow_voxel = torch.matmul(inv_ds_matrix, flow_flat.transpose(1, 2)).transpose(1, 2)

        # --- Voxel → normalized ---
        scale_to_norm = 2.0 / voxel_counts
        flow_norm = flow_voxel * scale_to_norm

        flow_norm = flow_norm.reshape(batch_size, *spatial_shape, 3).permute(0, 4, 1, 2, 3)

        return meshes + flow_norm
