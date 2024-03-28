"""
Created on Jul 10, 2023

@author: fechter
"""
from pytorch_lightning.callbacks import BasePredictionWriter
import os

from imageTransformation import Transformation
import atlas_utils
import torchio as tio
import torch


class DeformationFieldAndDeformedImageWriter(BasePredictionWriter):
    def __init__(self, config, write_interval):
        super().__init__(write_interval)
        self.output_dir = config.getParam("outputPath")
        self.meshDir = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        self.meshSpacing = config.getParam("registrationGridSpacing")

        labelLoss = config.getParam("labelLoss")
        if labelLoss == "NCC" or labelLoss == "SSD":
            self.transformDistanceMaps = True
        else:
            self.transformDistanceMaps = False

        self.transformer = Transformation()
        _fileType = config.getParam("fileTypeToWrite")
        if _fileType is None:
            self.fileType = ".mha"  ##would prefer nrrd, but had difficulties with vector orientation in Slicer
        else:
            self.fileType = _fileType

    def write_on_batch_end(self, trainer, pl_module, prediction, batch_indices, batch, batch_idx, dataloader_idx):
        images, meshes, labels = pl_module.prepare_batch(batch)

        atlasImages = pl_module.getInputAtlasImage(images.shape[0])
        atlasMeshes = pl_module.getInputAtlasMesh(images.shape[0])
        atlasLabels = pl_module.getInputAtlasLabel(images.shape[0])

        distanceMapsImg = None
        distanceMapsAtlas = None
        if self.transformDistanceMaps:
            distanceMapsImg = self.transformer.sampleImage(labels, meshes)
            distanceMapsImg = distanceMapsImg.cpu()
            labels = atlas_utils.convertDistanceMapToLabelMap(labels)
            distanceMapsAtlas = atlasLabels.cpu()
            atlasLabels = atlas_utils.convertDistanceMapToLabelMap(atlasLabels)

        sampledImages = self.transformer.sampleImage(images, meshes)
        sampledLabels = self.transformer.sampleImage(labels, meshes, interpolationType="nearest")

        pos_flow = prediction[0]
        neg_flow = prediction[1]

        posDeformationFieldAtlas = self.transformer.combineMeshesAndFlowField(atlasMeshes, pos_flow)
        # warpedAtlas = self.transformer.sampleImage(
        #     atlasImages,
        #     posDeformationFieldAtlas,
        # )
        # warpedAtlasLabels = self.transformer.sampleImage(
        #     atlasLabels, posDeformationFieldAtlas, interpolationType="nearest"
        # )

        # negDeformationFieldImages = self.transformer.combineMeshesAndFlowField(meshes, neg_flow)
        # warpedImages = self.transformer.sampleImage(images, negDeformationFieldImages)
        # warpedLabels = self.transformer.sampleImage(labels, negDeformationFieldImages, interpolationType="nearest")

        imageNames = batch["imagePath"]
        # meshOrigin = batch["meshOrigin"]
        atlasOrigin = pl_module.atlasOrigin.tolist()

        atlasImages = atlasImages.cpu()
        atlasLabels = torch.argmax(atlasLabels, dim=1, keepdim=True).cpu()
        sampledImages = sampledImages.cpu()
        sampledLabels = torch.argmax(sampledLabels, dim=1, keepdim=True).cpu()
        # warpedImages = warpedImages.cpu()
        # warpedLabels = torch.argmax(warpedLabels, dim=1, keepdim=True).cpu()
        # warpedAtlas = warpedAtlas.cpu()
        # warpedAtlasLabels = torch.argmax(warpedAtlasLabels, dim=1, keepdim=True).cpu()
        neg_flow = neg_flow.cpu()
        pos_flow = pos_flow.cpu()

        atlas_utils.saveImageTensor(
            atlasImages[0, None, ...],
            os.path.join(self.output_dir, "Atlas" + self.fileType),
            atlasOrigin,
            self.meshSpacing,
            self.meshDir,
        )

        atlas_utils.saveImageTensor(
            atlasLabels[0, None, ...],
            os.path.join(self.output_dir, "AtlasLabel" + self.fileType),
            atlasOrigin,
            self.meshSpacing,
            self.meshDir,
        )

        if distanceMapsAtlas is not None:
            for channel in range(0, distanceMapsAtlas.shape[1]):
                distanceMapChannel = distanceMapsAtlas[:, channel, None, ...]
                distanceMapChannel = distanceMapChannel.cpu()
                atlas_utils.saveImageTensor(
                    distanceMapChannel[0, None, ...],
                    os.path.join(self.output_dir, "AtlasDistanceMapChannel" + str(channel) + self.fileType),
                    atlasOrigin,
                    self.meshSpacing,
                    self.meshDir,
                )

        for i in range(0, sampledImages.shape[0]):
            fileBaseName = os.path.splitext(os.path.basename(imageNames[i]))[0]

            if os.path.exists(os.path.join(self.output_dir, fileBaseName + self.fileType)):
                fileIdx = 0
                fileBaseNameBUP = fileBaseName + str(fileIdx)
                while os.path.exists(os.path.join(self.output_dir, fileBaseNameBUP + self.fileType)):
                    fileIdx = fileIdx + 1
                    fileBaseNameBUP = fileBaseName + str(fileIdx)
                fileBaseName = fileBaseNameBUP

            ## save resampled original image
            atlas_utils.saveImageTensor(
                sampledImages[i, None, ...],
                os.path.join(self.output_dir, fileBaseName + self.fileType),
                atlasOrigin,
                self.meshSpacing,
                self.meshDir,
            )

            ## save ditanceMaps
            if distanceMapsImg is not None:
                currDistanceMaps = distanceMapsImg[i, None, ...]
                for channel in range(0, currDistanceMaps.shape[1]):
                    distanceMapChannel = currDistanceMaps[0, None, channel, None, ...]
                    atlas_utils.saveImageTensor(
                        distanceMapChannel,
                        os.path.join(self.output_dir, fileBaseName + "DistanceMap" + str(channel) + self.fileType),
                        atlasOrigin,
                        self.meshSpacing,
                        self.meshDir,
                    )

            ## save resampled original label
            atlas_utils.saveImageTensor(
                sampledLabels[i, None, ...],
                os.path.join(self.output_dir, fileBaseName + "Label" + self.fileType),
                atlasOrigin,
                self.meshSpacing,
                self.meshDir,
            )

            # ## save deformed images in atlas space
            # atlas_utils.saveImageTensor(
            #     warpedImages[i, None, ...],
            #     os.path.join(self.output_dir, fileBaseName + "Def" + self.fileType),
            #     atlasOrigin,
            #     self.meshSpacing,
            #     self.meshDir,
            # )

            # ## save deformed labels in atlas space
            # atlas_utils.saveImageTensor(
            #     warpedLabels[i, None, ...],
            #     os.path.join(self.output_dir, fileBaseName + "LabelDef" + self.fileType),
            #     atlasOrigin,
            #     self.meshSpacing,
            #     self.meshDir,
            # )

            ## save deformation fields: image space -> atlas space
            atlas_utils.saveDefField(
                os.path.join(self.output_dir, fileBaseName + "DefField" + self.fileType),
                neg_flow[i, None, ...],
                atlasOrigin,
                self.meshSpacing,
                self.meshDir,
            )

            # ## save deformed atlas in image space
            # atlas_utils.saveImageTensor(
            #     warpedAtlas[i, None, ...],
            #     os.path.join(self.output_dir, fileBaseName + "AtlasDef" + self.fileType),
            #     atlasOrigin,  # meshOrigin[i].tolist(),
            #     self.meshSpacing,
            #     self.meshDir,
            # )

            # ## save deformed atlas labels in image space
            # atlas_utils.saveImageTensor(
            #     warpedAtlasLabels[i, None, ...],
            #     os.path.join(self.output_dir, fileBaseName + "AtlasLabelDef" + self.fileType),
            #     atlasOrigin,  # meshOrigin[i].tolist(),
            #     self.meshSpacing,
            #     self.meshDir,
            # )

            ## save deformation fields: atlas space -> image space
            atlas_utils.saveDefField(
                os.path.join(self.output_dir, fileBaseName + "AtlasDefField" + self.fileType),
                pos_flow[i, None, ...],
                atlasOrigin,  # meshOrigin[i].tolist(),
                self.meshSpacing,
                self.meshDir,
            )

    def write_on_epoch_end(self, trainer, pl_module, predictions, batch_indices):
        pass
