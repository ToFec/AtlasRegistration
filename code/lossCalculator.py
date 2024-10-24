"""
Created on Apr 28, 2023

@author: fechter
"""
from losses import LossFactory
from imageTransformation import Transformation
import torch
from LossWrapper import LossWrapper
import atlas_utils


class LossCalculator:
    def __init__(self, config):
        self.transformer = Transformation()

        self.lossWrapper = LossWrapper()
        similartiyLossName = config.getParam("similarityLoss")
        self.sim_factor = config.getParam("similarityFactor")
        self.labelSimilarityFactor = config.getParam("labelSimilarityFactor")
        if self.labelSimilarityFactor is None:
            self.labelSimilarityFactor = 0.0

        self.labelSimilarityFactorAtlasSpace = config.getParam("labelSimilarityFactorAtlasSpace")
        if self.labelSimilarityFactorAtlasSpace is None:
            self.labelSimilarityFactorAtlasSpace = 0.0

        if self.sim_factor is not None and self.sim_factor != 0.0:
            self.similarityLoss = LossFactory.lossMap[similartiyLossName]()
        else:
            self.sim_factor = 0.0
            self.similarityLoss = LossFactory.lossMap["Dummy"]()

        diceLoss = config.getParam("labelLoss")
        if diceLoss is not None and diceLoss in LossFactory.lossMap:
            self.diceLoss = LossFactory.lossMap[diceLoss]()
            ignoreBackground = config.getParam("ignoreBackground")
            if ignoreBackground:
                self.diceLoss.setIgnoreBackground(ignoreBackground)
        else:
            self.diceLoss = LossFactory.lossMap["Dummy"]()

        self.reg_factor = config.getParam("regularizationFactor")
        if self.reg_factor is not None and self.reg_factor != 0.0:
            regularizationLossName = config.getParam("regularizationLoss")
            if regularizationLossName is None:
                regularizationLossName = "BendingEnergy"
            self.regularizationLoss = LossFactory.lossMap[regularizationLossName]()
        else:
            self.reg_factor = 0.0
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

        self.defFieldInverseConsistencyLossFactor = config.getParam("defDieldInverseConsistencyLossFactor")

        if self.defFieldInverseConsistencyLossFactor is None:
            self.defFieldInverseConsistencyLossFactor = 0.0
            # self.defFieldInverseConsistencyLoss = LossFactory.lossMap["Dummy"]()
        # else:
        self.defFieldInverseConsistencyLoss = LossFactory.lossMap["MissingCorrespondences"](self.transformer)

    def _getDefomredImages(
        self, posDeformationField, neg_flow, images, meshes, paddMode="border", interpolationType="bilinear"
    ):
        # sec_src_imgs = torch.flip(images, dims=[0])
        sec_src_imgs = images[::-1]
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

        warpedAtlasLabels = self.transformer.sampleImage(atlasLabels, posDeformationFieldAtlas)
        # self.lossWrapper.labelSimilarityLoss = self._getImageSpaceSimilarityLoss(warpedAtlasLabels, sampledLabels)
        self.lossWrapper.labelSimilarityLoss = self._getDiceloss(sampledLabels, warpedAtlasLabels)

        warpedLabels = self.transformer.sampleImage(labels, negDeformationFieldImages)
        self.lossWrapper.labelSimilarityLossAtlasSpace = self._getDiceloss(atlasLabels, warpedLabels)

        deformedImages = self._getDefomredImages(posDeformationField, neg_flow, images, meshes)
        self.lossWrapper.pair_sim_loss = self._getImageSpaceSimilarityLoss(deformedImages, sampledImages)

        deformedLabels = self._getDefomredImages(posDeformationField, neg_flow, labels, meshes)
        self.lossWrapper.imgSpaceLabelLoss = self._getDiceloss(sampledLabels, deformedLabels)

        batch_size = meshes.shape[0]
        if (batch_size % 2) == 0:
            warpedImages = self.transformer.sampleImage(images, negDeformationFieldImages)
            self.lossWrapper.atlas_pair_sim_loss = self._getImageSpaceSimilarityLoss(
                warpedImages[: int(batch_size / 2)], warpedImages[int(batch_size / 2) :]
            )

            self.lossWrapper.atlasSpaceLabelLoss = self._getDiceloss(
                warpedLabels[: int(batch_size / 2)], warpedLabels[int(batch_size / 2) :]
            )
        else:
            self.lossWrapper.atlas_pair_sim_loss = torch.zeros_like(self.lossWrapper.reg_loss)
            self.lossWrapper.atlasSpaceLabelLoss = torch.zeros_like(self.lossWrapper.reg_loss)

        self.lossWrapper.defFieldInverseConsistencyLoss = self.defFieldInverseConsistencyLoss(pos_flow, neg_flow)

    def getLossesWithoutWeighting(self):
        return (
            self.lossWrapper.sim_loss,
            self.lossWrapper.reg_loss,
            self.lossWrapper.pair_sim_loss,
            self.lossWrapper.atlas_pair_sim_loss,
            self.lossWrapper.imgSpaceLabelLoss,
            self.lossWrapper.atlasSpaceLabelLoss,
            self.lossWrapper.labelSimilarityLoss,
            self.lossWrapper.labelSimilarityLossAtlasSpace,
            self.lossWrapper.defFieldInverseConsistencyLoss,
        )

    def getLosses(self):
        (
            sim_loss,
            reg_loss,
            pair_sim_loss,
            atlas_pair_sim_loss,
            imgSpaceLabelLoss,
            atlasSpaceLabelLoss,
            labelSimilarityLoss,
            labelSimilarityFactorAtlasSpace,
            defFieldInverseConsistencyLoss,
        ) = self.getLossesWithoutWeighting()

        sim_loss = sim_loss * self.sim_factor

        reg_loss = reg_loss * self.reg_factor

        pair_sim_loss = pair_sim_loss * self.imagePairSimilarityFactor
        atlas_pair_sim_loss = atlas_pair_sim_loss * self.atlasPairSimilarityFactor

        imgSpaceLabelLoss = imgSpaceLabelLoss * self.imageSpaceLabelSimFactor
        atlasSpaceLabelLoss = atlasSpaceLabelLoss * self.atlasSpaceLabelSimFactor

        labelSimilarityLoss = labelSimilarityLoss * self.labelSimilarityFactor

        labelSimilarityFactorAtlasSpace = labelSimilarityFactorAtlasSpace * self.labelSimilarityFactorAtlasSpace

        defFieldInverseConsistencyLoss = defFieldInverseConsistencyLoss * self.defFieldInverseConsistencyLossFactor

        return (
            sim_loss,
            reg_loss,
            pair_sim_loss,
            atlas_pair_sim_loss,
            imgSpaceLabelLoss,
            atlasSpaceLabelLoss,
            labelSimilarityLoss,
            labelSimilarityFactorAtlasSpace,
            defFieldInverseConsistencyLoss,
        )

    def getDiceLosses(self, pos_flow, neg_flow, labels, meshes):
        sampledLabels = self.transformer.sampleImage(labels, meshes, interpolationType="nearest")

        posDeformationField = self.transformer.getDeformationField(pos_flow)
        negDeformationFieldImages = self.transformer.combineMeshesAndFlowField(meshes, neg_flow)

        deformedLabels = self._getDefomredImages(posDeformationField, neg_flow, labels, meshes)
        imgSpaceDiceloss = self._getDiceloss(deformedLabels, sampledLabels)

        warpedLabels = self.transformer.sampleImage(labels, negDeformationFieldImages)
        batch_size = labels.shape[0]
        atlasSpaceDiceLoss = self._getDiceloss(warpedLabels[: int(batch_size / 2)], warpedLabels[int(batch_size / 2) :])

        return imgSpaceDiceloss, atlasSpaceDiceLoss
