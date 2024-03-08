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
from losses import LossFactory
import numpy as np
import csv


class PredictionEvaluationWriter(BasePredictionWriter):
    def __init__(self, config, write_interval):
        super().__init__(write_interval)
        self.output_dir = config.getParam("outputPath")
        self.csvDelimiter = config.getParam("csvDelimiter")

        labelLoss = config.getParam("labelLoss")
        if labelLoss == "NCC":
            self.transformDistanceMaps = True
        else:
            self.transformDistanceMaps = False

        self.transformer = Transformation()
        _fileType = config.getParam("fileTypeToWrite")
        if _fileType is None:
            self.fileType = ".mha"  ##would prefer nrrd, but had difficulties with vector orientation in Slicer
        else:
            self.fileType = _fileType

        self.finalResultList = []
        self.header = None

    def write_on_epoch_end(self, trainer, pl_module, predictions, batch_indices):
        if self.header is not None:
            outputFile = os.path.join(self.output_dir, "EvaluationFigures.csv")
            with open(outputFile, "w", encoding="UTF8") as f:
                writer = csv.writer(f, delimiter=self.csvDelimiter)
                writer.writerow(self.header)
                for lineToWrite in self.finalResultList:
                    writer.writerow(lineToWrite)

    def write_on_batch_end(self, trainer, pl_module, prediction, batch_indices, batch, batch_idx, dataloader_idx):
        images, meshes, labels = pl_module.prepare_batch(batch)

        meshSpacing = atlas_utils.getMeshSpacing(meshes[0, ...])

        atlasImages = pl_module.getInputAtlasImage(images.shape[0])
        atlasMeshes = pl_module.getInputAtlasMesh(images.shape[0])
        atlasLabels = pl_module.getInputAtlasLabel(images.shape[0])

        if self.transformDistanceMaps:
            labels = atlas_utils.convertDistanceMapToLabelMap(labels)
            atlasLabels = atlas_utils.convertDistanceMapToLabelMap(atlasLabels)

        sampledLabels = self.transformer.sampleImage(labels, meshes, interpolationType="nearest")

        pos_flow = prediction[0]
        neg_flow = prediction[1]

        posDeformationFieldAtlas = self.transformer.combineMeshesAndFlowField(atlasMeshes, pos_flow)
        warpedAtlas = self.transformer.sampleImage(
            atlasImages,
            posDeformationFieldAtlas,
        )
        warpedAtlasLabels = self.transformer.sampleImage(
            atlasLabels, posDeformationFieldAtlas, interpolationType="nearest"
        )

        negDeformationFieldImages = self.transformer.combineMeshesAndFlowField(meshes, neg_flow)
        warpedLabels = self.transformer.sampleImage(labels, negDeformationFieldImages, interpolationType="nearest")

        imageNames = batch["imagePath"]

        atlasLabels = torch.argmax(atlasLabels, dim=1, keepdim=True)
        sampledLabels = torch.argmax(sampledLabels, dim=1, keepdim=True)
        warpedLabels = torch.argmax(warpedLabels, dim=1, keepdim=True)
        warpedAtlasLabels = torch.argmax(warpedAtlasLabels, dim=1, keepdim=True)

        diceLoss = LossFactory.lossMap["MultiClassSingleChannelDiceCalculator"]()

        for i in range(0, warpedAtlas.shape[0]):
            fileName = imageNames[i]
            originalDscValues = diceLoss.getDscValues(sampledLabels[i, None, ...], atlasLabels[0, None, ...])
            warpedLabelDscInAtlasSpace = diceLoss.getDscValues(warpedLabels[i, None, ...], atlasLabels[0, None, ...])

            warpedLabelsInImgSpace = diceLoss.getDscValues(sampledLabels[i, None, ...], warpedAtlasLabels[0, None, ...])

            jacobiDetNegFlow = atlas_utils.jacobian_determinant(neg_flow[i, None, ...], meshSpacing)

            fractionOfFoldingsNegFlow = np.sum(jacobiDetNegFlow < 0) / jacobiDetNegFlow.size

            jacobiDetPosFlow = atlas_utils.jacobian_determinant(
                pos_flow[i, None, ...], meshSpacing
            )  # Atlas to Image DefField
            fractionOfFoldingsPosFlow = np.sum(jacobiDetPosFlow < 0) / jacobiDetPosFlow.size

            result = (
                [
                    fileName,
                ]
                + originalDscValues.numpy().astype(str).tolist()
                + warpedLabelDscInAtlasSpace.numpy().astype(str).tolist()
                + warpedLabelsInImgSpace.numpy().astype(str).tolist()
                + [
                    str(fractionOfFoldingsNegFlow),
                ]
                + [
                    str(fractionOfFoldingsPosFlow),
                ]
            )

            if self.header == None:
                self.header = (
                    [
                        "Case",
                    ]
                    + ["Original Dsc"] * originalDscValues.__len__()
                    + ["Atlas Space Dsc"] * warpedLabelDscInAtlasSpace.__len__()
                    + ["Img Space Dsc"] * warpedLabelsInImgSpace.__len__()
                    + [
                        "Img To Atlas FoF",
                    ]
                    + [
                        "Atlas To Img FoF",
                    ]
                )

            self.finalResultList.append(result)
