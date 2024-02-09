"""
Created on Apr 28, 2023

@author: fechter
"""
from losses import LossFactory
from imageTransformation import Bilinear
import torch
from LossWrapper import LossWrapper


class LossCalculator:
    def __init__(self, config):
        self.lossWrapper = LossWrapper()
        similartiyLossName = config.getParam("similarityLoss")
        self.sim_factor = config.getParam("similarityFactor")
        self.labelSimilarityFactor = config.getParam("labelSimilarityFactor")
        if self.labelSimilarityFactor is None:
            self.labelSimilarityFactor = 0.0

        self.similarityLoss = LossFactory.lossMap[similartiyLossName]()

        self.diceLoss = config.getParam("labelLoss")
        if self.diceLoss is not None:
            self.diceLoss = LossFactory.lossMap["MultiClassSingleChannelDiceCalculator"]()
        else:
            self.diceLoss = LossFactory.lossMap["Dummy"]()

        self.reg_factor = config.getParam("regularizationFactor")
        if self.reg_factor != 0.0:
            regularizationLossName = config.getParam("regularizationLoss")
            if regularizationLossName is None:
                regularizationLossName = "BendingEnergy"
            self.regularizationLoss = LossFactory.lossMap[regularizationLossName]()
        else:
            self.regularizationLoss = LossFactory.lossMap["Dummy"]()

        self.imagePairSimilarityFactor = config.getParam("imagePairSimFactor")
        self.imageSpaceLabelSimFactor = config.getParam("imageSpaceLabelSimFactor")
        if self.imageSpaceLabelSimFactor is None:
            self.imageSpaceLabelSimFactor = 0.0

        self.atlasPairSimilarityFactor = config.getParam("atlasPairSimFactor")
        self.atlasSpaceLabelSimFactor = config.getParam("atlasSpaceLabelSimFactor")
        if self.atlasSpaceLabelSimFactor is None:
            self.atlasSpaceLabelSimFactor = 0.0

        self.smooth_factor = config.getParam("smoothingFactor")
        if self.smooth_factor is not None and self.smooth_factor != 0.0:
            self.smoothLoss = LossFactory.lossMap["GradLoss"](penalty="l2")
        else:
            self.smooth_factor = 0.0
            self.smoothLoss = LossFactory.lossMap["Dummy"]

        self.transformer = Bilinear(zero_boundary=True)

    def _getDefomredImages(
        self, posDeformationField, neg_flow, images, meshes, paddMode="border", interpolationType="bilinear"
    ):
        sec_src_imgs = torch.flip(images, dims=[0])
        negFlowAndMesh = self.transformer.combineMeshesAndFlowField(meshes, neg_flow)
        secNegFlowAndMesh = torch.flip(negFlowAndMesh, dims=[0])
        transforemdImageMeshToOtherImageSpace = self.transformer.sampleImage(secNegFlowAndMesh, posDeformationField)
        return self.transformer.sampleImage(
            sec_src_imgs, transforemdImageMeshToOtherImageSpace, paddMode=paddMode, interpolationType=interpolationType
        )

    def _getDiceloss(self, label0, label1):
        dscLoss = self.diceLoss(label0, label1)
        return dscLoss

    def _getImageSpaceSimilarityLoss(self, imgs0, imgs1):
        imgSpaceSimLoss = self.similarityLoss(imgs0, imgs1)
        return imgSpaceSimLoss / imgs0.shape[0]

    def getLoss(self):
        lossValues = self.getLosses()
        return sum(lossValues)

    def calculateLoss(self, pos_flow, neg_flow, images, meshes, atlasImages, atlasMeshes, atlasLabels, labels):
        # posDeformationField = self.transformer.getDeformationField(pos_flow)
        # deformedAtlasMeshes = self.transformer(atlasMeshes, posDeformationField)
        # deformedAtlas = self.transformer.sampleImage(atlasImages, deformedAtlasMeshes)

        posDeformationFieldAtlas = self.transformer.combineMeshesAndFlowField(atlasMeshes, pos_flow)
        posDeformationField = self.transformer.getDeformationField(pos_flow)
        negDeformationFieldImages = self.transformer.combineMeshesAndFlowField(
            meshes, neg_flow
        )  # self.transformer.getDeformationField(neg_flow)

        warpedAtlas = self.transformer.sampleImage(atlasImages, posDeformationFieldAtlas)

        sampledImages = self.transformer.sampleImage(images, meshes)
        sampledLabels = self.transformer.sampleImage(labels, meshes, interpolationType="nearest")

        self.lossWrapper.sim_loss = self._getImageSpaceSimilarityLoss(warpedAtlas, sampledImages)

        self.lossWrapper.reg_loss = self.regularizationLoss(pos_flow)

        if self.labelSimilarityFactor != 0.0:
            warpedAtlasLabels = self.transformer.sampleImage(
                atlasLabels, posDeformationFieldAtlas, interpolationType="nearest"
            )
            self.lossWrapper.atlasSpaceLabelLoss = self._getDiceloss(warpedAtlasLabels, sampledLabels)

        if self.imagePairSimilarityFactor != 0.0:
            deformedImages = self._getDefomredImages(posDeformationField, neg_flow, images, meshes)
            self.lossWrapper.pair_sim_loss = self._getImageSpaceSimilarityLoss(deformedImages, sampledImages)

        if self.imageSpaceLabelSimFactor != 0.0:
            deformedLabels = self._getDefomredImages(
                posDeformationField, neg_flow, labels, meshes, interpolationType="nearest"
            )
            self.lossWrapper.imgSpaceLabelLoss = self._getDiceloss(deformedLabels, sampledLabels)

        if self.atlasPairSimilarityFactor != 0.0:
            warpedImages = self.transformer.sampleImage(images, negDeformationFieldImages)
            batch_size = images.shape[0]
            self.lossWrapper.atlas_pair_sim_loss = self._getImageSpaceSimilarityLoss(
                warpedImages[: int(batch_size / 2)], warpedImages[int(batch_size / 2) :]
            )

        if self.atlasSpaceLabelSimFactor != 0.0:
            warpedLabels = self.transformer.sampleImage(labels, negDeformationFieldImages, interpolationType="nearest")
            batch_size = labels.shape[0]
            self.lossWrapper.atlasSpaceLabelLoss = self._getDiceloss(
                warpedLabels[: int(batch_size / 2)], warpedLabels[int(batch_size / 2) :]
            )

    def getLossesWithoutWeighting(self):
        return (
            self.lossWrapper.sim_loss,
            self.lossWrapper.reg_loss,
            self.lossWrapper.pair_sim_loss,
            self.lossWrapper.atlas_pair_sim_loss,
            self.lossWrapper.imgSpaceLabelLoss,
            self.lossWrapper.atlasSpaceLabelLoss,
        )

    def getLosses(self):
        (
            sim_loss,
            reg_loss,
            pair_sim_loss,
            atlas_pair_sim_loss,
            imgSpaceLabelLoss,
            atlasSpaceLabelLoss,
        ) = self.getLossesWithoutWeighting()

        sim_loss = sim_loss * self.sim_factor

        reg_loss = reg_loss * self.reg_factor

        pair_sim_loss = pair_sim_loss * self.imagePairSimilarityFactor
        atlas_pair_sim_loss = atlas_pair_sim_loss * self.atlasPairSimilarityFactor

        imgSpaceLabelLoss = imgSpaceLabelLoss * self.imageSpaceLabelSimFactor
        atlasSpaceLabelLoss = atlasSpaceLabelLoss * self.atlasSpaceLabelSimFactor

        return sim_loss, reg_loss, pair_sim_loss, atlas_pair_sim_loss, imgSpaceLabelLoss, atlasSpaceLabelLoss

    def getDiceLosses(self, pos_flow, neg_flow, labels, meshes):
        sampledLabels = self.transformer.sampleImage(labels, meshes, interpolationType="nearest")

        posDeformationField = self.transformer.getDeformationField(pos_flow)
        negDeformationFieldImages = self.transformer.combineMeshesAndFlowField(meshes, neg_flow)

        deformedLabels = self._getDefomredImages(
            posDeformationField, neg_flow, labels, meshes, interpolationType="nearest"
        )
        imgSpaceDiceloss = self._getDiceloss(deformedLabels, sampledLabels)

        warpedLabels = self.transformer.sampleImage(labels, negDeformationFieldImages, interpolationType="nearest")
        batch_size = labels.shape[0]
        atlasSpaceDiceLoss = self._getDiceloss(warpedLabels[: int(batch_size / 2)], warpedLabels[int(batch_size / 2) :])

        return imgSpaceDiceloss, atlasSpaceDiceLoss
