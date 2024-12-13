"""
Created on Jul 10, 2023

@author: fechter
"""
from pytorch_lightning.callbacks import Callback
import os

from imageTransformation import Transformation
import atlas_utils
import torchio as tio
import torch
import SimpleITK as sitk


class DeformationFieldAndDeformedImageWriter(Callback):
    def __init__(self, config, isStageTypePredict=False):
        # super().__init__(write_interval)
        self.output_dir = config.getParam("outputPath")
        self.meshDir = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        self.meshSpacing = config.getParam("registrationGridSpacing")

        if config.getParam("convertToDistanceMaps"):
            self.transformDistanceMaps = True
        else:
            self.transformDistanceMaps = False

        atlasLabelName = config.getParam("atlasLabel")
        atlasImageName = config.getParam("atlasImage")
        meshName = os.path.splitext(atlasImageName)[0] + "Mesh.pt"
        sampleMesh, _, _ = torch.load(meshName)
        sitkLabel = sitk.ReadImage(atlasLabelName, sitk.sitkInt64)
        self.atlasLabelImage = tio.LabelMap.from_sitk(sitkLabel)
        imgShape = list(sampleMesh.shape)
        imgShape[0] = 1
        imgShape = [1] + imgShape
        transformer = Transformation(imgShape)
        tmpImg = self.atlasLabelImage[tio.DATA].unsqueeze(0).type(torch.FloatTensor)
        self.atlasLabelImage = transformer.sampleImage(tmpImg, sampleMesh.unsqueeze(0), interpolationType="nearest")

        self.isStageTypePredict = isStageTypePredict

        self.transformer = Transformation()
        _fileType = config.getParam("fileTypeToWrite")
        if _fileType is None:
            self.fileType = ".mha"  ##would prefer nrrd, but had difficulties with vector orientation in Slicer
        else:
            self.fileType = _fileType

        self.ignoreBackground = config.getParam("ignoreBackground")

        self.maximalDistanceForDitanceMaps = config.getParam("maxDistanceForDistanceMaps")

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        self.write_on_batch_end(trainer, pl_module, outputs, None, batch, batch_idx, dataloader_idx)

    def on_predict_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        self.write_on_batch_end(trainer, pl_module, outputs, None, batch, batch_idx, dataloader_idx)

    def write_on_batch_end(self, trainer, pl_module, prediction, batch_indices, batch, batch_idx, dataloader_idx):
        images, meshes, labels = pl_module.prepare_batch(batch)

        atlasImages = pl_module.getInputAtlasImage(meshes.shape[0])
        # atlasMeshes = pl_module.getInputAtlasMesh(images.shape[0])
        atlasLabels = pl_module.getInputAtlasLabel(meshes.shape[0])
        loadedAtlasLabels = self.atlasLabelImage.data.expand(meshes.shape[0], -1, -1, -1, -1).to(meshes.device)

        loadedLabels = []
        for labelIdx in range(len(batch["labelPath"])):
            labelFileName = batch["labelPath"][labelIdx]
            sitkLabel = sitk.ReadImage(labelFileName, sitk.sitkInt64)
            # as long as we deal only with affine registrations we do not need to consider them here
            # because they are only applied to the header and the information is already considered in the provided mesh
            # transformationFileName = batch["preTransformaton"][labelIdx]
            # if transformationFileName is not None:
            #     transform = sitk.ReadTransform(transformationFileName)
            #     atlasUtils.applyRigidRegistrationToImgHeader(sitkLabel, transform)
            loadedLabels.append(tio.LabelMap.from_sitk(sitkLabel).data.to(torch.float32).to(meshes.device))

        distanceMapsImg = None
        distanceMapsAtlas = None
        if self.transformDistanceMaps:
            distanceMapsImg = self.transformer.sampleImage(labels, meshes)
            distanceMapsImg = distanceMapsImg.cpu()
            distanceMapsAtlas = atlasLabels.cpu()

        sampledImages = self.transformer.sampleImage(images, meshes)
        sampledLabels = self.transformer.sampleImage(loadedLabels, meshes, interpolationType="nearest")

        pos_flow = prediction[0]
        neg_flow = prediction[1]

        imageNames = batch["imagePath"]
        atlasOrigin = pl_module.atlasOrigin.tolist()

        atlasImages = atlasImages.cpu()
        atlasLabels = loadedAtlasLabels.cpu()
        sampledImages = sampledImages.cpu()
        sampledLabels = sampledLabels.cpu()
        neg_flow = neg_flow.cpu()
        pos_flow = pos_flow.cpu()

        # posFlowMinusWarpedNegFlow, negFlowMinusWarpedPosFlow = atlas_utils.segmentMisssingCorrespondences(
        #     pos_flow, neg_flow, self.transformer
        # )

        if not self.isStageTypePredict:
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
            if self.isStageTypePredict:
                self.output_dir = os.path.dirname(imageNames[i])
                fileBaseName = ""
            else:
                fileBaseName = os.path.splitext(os.path.basename(imageNames[i]))[0]
                if os.path.exists(os.path.join(self.output_dir, fileBaseName + self.fileType)):
                    fileIdx = 0
                    fileBaseNameBUP = fileBaseName + str(fileIdx)
                    while os.path.exists(os.path.join(self.output_dir, fileBaseNameBUP + self.fileType)):
                        fileIdx = fileIdx + 1
                        fileBaseNameBUP = fileBaseName + str(fileIdx)
                    fileBaseName = fileBaseNameBUP

            if not self.isStageTypePredict:
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

            # atlas_utils.saveImageTensor(
            #     negFlowMinusWarpedPosFlow[i, None, ...],
            #     os.path.join(self.output_dir, fileBaseName + "DefFieldMinusAtlasDefField" + self.fileType),
            #     atlasOrigin,
            #     self.meshSpacing,
            #     self.meshDir,
            # )

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

            # atlas_utils.saveImageTensor(
            #     posFlowMinusWarpedNegFlow[i, None, ...],
            #     os.path.join(self.output_dir, fileBaseName + "AtlasDefFieldMinusImageDefField" + self.fileType),
            #     atlasOrigin,
            #     self.meshSpacing,
            #     self.meshDir,
            # )

            flowFieldSpacing = [(2.0 / (pos_flow.shape[i + 2] - 1)) for i in range(len(self.meshSpacing))]

            jacobiDetNegFlow = atlas_utils.jacobianDeterminant(neg_flow[i, None, ...], flowFieldSpacing)

            atlas_utils.saveImageTensor(
                jacobiDetNegFlow,
                os.path.join(self.output_dir, fileBaseName + "DefFieldJacobian" + self.fileType),
                atlasOrigin,
                self.meshSpacing,
                self.meshDir,
            )
            jacobiDetPosFlow = atlas_utils.jacobianDeterminant(pos_flow[i, None, ...], flowFieldSpacing)
            atlas_utils.saveImageTensor(
                jacobiDetPosFlow,
                os.path.join(self.output_dir, fileBaseName + "AtlasDefFieldJacobian" + self.fileType),
                atlasOrigin,
                self.meshSpacing,
                self.meshDir,
            )

            jacobyMeanValue = torch.zeros_like(distanceMapsAtlas[i, None, ...])
            labelMap = torch.floor(distanceMapsAtlas[i, None, ...] / self.maximalDistanceForDitanceMaps)
            labels = torch.unique(labelMap)
            for label in labels:
                jacobyMeanValue[labelMap == label] = jacobiDetPosFlow[labelMap == label].mean()

#            jacobiDetPosFlow = torch.nn.functional.avg_pool3d(
#                torch.nn.functional.avg_pool3d(jacobiDetPosFlow, kernel_size=3, stride=1, padding=1),
#                kernel_size=3,
#                stride=1,
#                padding=1,
#            )

            jacobiDetPosFlow = torch.nn.functional.avg_pool3d(jacobiDetPosFlow, kernel_size=3, stride=1, padding=1)
            diff = torch.abs(jacobiDetPosFlow) / torch.abs(jacobyMeanValue)
            diff[diff != 0.0] = torch.max(diff[diff != 0.0], 1.0 / diff[diff != 0.0])

            simgoid = torch.nn.Sigmoid()
            diff = simgoid(10*(diff-2.0))
            vpLossValues = diff

            atlas_utils.saveImageTensor(
                vpLossValues,
                os.path.join(self.output_dir, fileBaseName + "VolumePreservingLoss" + self.fileType),
                atlasOrigin,
                self.meshSpacing,
                self.meshDir,
            )

            atlas_utils.saveImageTensor(
                jacobyMeanValue,
                os.path.join(self.output_dir, fileBaseName + "JacobiMeanValues" + self.fileType),
                atlasOrigin,
                self.meshSpacing,
                self.meshDir,
            )
