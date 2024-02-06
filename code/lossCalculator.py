"""
Created on Apr 28, 2023

@author: fechter
"""
from losses import LossFactory
from imageTransformation import Bilinear
import torch


class LossCalculator:
    def __init__(self, config):
        similartiyLossName = config.getParam("similarityLoss")
        self.sim_factor = config.getParam("similarityFactor")
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

        self.atlasPairSimilarityFactor = config.getParam("atlasPairSimFactor")
        self.atlasSpaceLabelSimFactor = config.getParam("atlasSpaceLabelSimFactor")

        self.smooth_factor = config.getParam("smoothingFactor")
        if self.smooth_factor != 0.0:
            self.smoothLoss = LossFactory.lossMap["GradLoss"](penalty="l2")
        else:
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

    def getLoss(self, pos_flow, neg_flow, images, meshes, atlasImages, atlasMeshes, labels):
        sim_loss, reg_loss, pair_sim_loss, atlas_pair_sim_loss, imgSpaceLabelLoss, atlasSpaceLabelLoss = self.getLosses(
            pos_flow, neg_flow, images, meshes, atlasImages, atlasMeshes, labels
        )
        return sim_loss + reg_loss + pair_sim_loss + atlas_pair_sim_loss + imgSpaceLabelLoss + atlasSpaceLabelLoss

    def getLossWeights(self):
        return (
            self.sim_factor,
            self.reg_factor,
            self.imagePairSimilarityFactor,
            self.atlasPairSimilarityFactor,
            self.imageSpaceLabelSimFactor,
            self.atlasSpaceLabelSimFactor,
        )

    def getLossesWithoutWeighting(self, pos_flow, neg_flow, images, meshes, atlasImages, atlasMeshes, labels):
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

        sim_loss = self._getImageSpaceSimilarityLoss(warpedAtlas, sampledImages)

        reg_loss = self.regularizationLoss(pos_flow)

        pair_sim_loss = 0.0
        if self.imagePairSimilarityFactor != 0.0:
            deformedImages = self._getDefomredImages(posDeformationField, neg_flow, images, meshes)
            pair_sim_loss = self._getImageSpaceSimilarityLoss(deformedImages, sampledImages)

        imgSpaceLabelLoss = 0.0
        if self.imageSpaceLabelSimFactor != 0.0:
            deformedLabels = self._getDefomredImages(
                posDeformationField, neg_flow, labels, meshes, interpolationType="nearest"
            )
            sampledLabels = self.transformer.sampleImage(labels, meshes, interpolationType="nearest")
            imgSpaceLabelLoss = self._getDiceloss(deformedLabels, sampledLabels)

        atlas_pair_sim_loss = 0.0
        if self.atlasPairSimilarityFactor != 0.0:
            warpedImages = self.transformer.sampleImage(images, negDeformationFieldImages)
            batch_size = images.shape[0]
            atlas_pair_sim_loss = self._getImageSpaceSimilarityLoss(
                warpedImages[: int(batch_size / 2)], warpedImages[int(batch_size / 2) :]
            )

        atlasSpaceLabelLoss = 0.0
        if self.atlasSpaceLabelSimFactor != 0.0:
            warpedLabels = self.transformer.sampleImage(labels, negDeformationFieldImages, interpolationType="nearest")
            batch_size = labels.shape[0]
            atlasSpaceLabelLoss = self._getDiceloss(
                warpedLabels[: int(batch_size / 2)], warpedLabels[int(batch_size / 2) :]
            )

        return sim_loss, reg_loss, pair_sim_loss, atlas_pair_sim_loss, imgSpaceLabelLoss, atlasSpaceLabelLoss

    def getLosses(self, pos_flow, neg_flow, images, meshes, atlasImages, atlasMeshes, labels):
        (
            sim_loss,
            reg_loss,
            pair_sim_loss,
            atlas_pair_sim_loss,
            imgSpaceLabelLoss,
            atlasSpaceLabelLoss,
        ) = self.getLossesWithoutWeighting(pos_flow, neg_flow, images, meshes, atlasImages, atlasMeshes, labels)

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
