"""
Created on Apr 28, 2023

@author: fechter
"""
from losses import LossFactory
from imageTransformation import Transformation
import torch
from LossWrapper import LossWrapper


class LossCalculator:
    def __init__(self, config):
        self.transformer = Transformation()

        self.lossWrapper = LossWrapper()

        labelSimilarityFactor = config.getParam("labelSimilarityFactor")
        if labelSimilarityFactor is not None:
            self.lossWrapper.setLossFactor("labelSimilarityLoss", labelSimilarityFactor)

        labelSimilarityFactorAtlasSpace = config.getParam("labelSimilarityFactorAtlasSpace")
        if labelSimilarityFactorAtlasSpace is not None:
            self.lossWrapper.setLossFactor("labelSimilarityLossAtlasSpace", labelSimilarityFactorAtlasSpace)

        sim_factor = config.getParam("similarityFactor")
        if sim_factor is not None and sim_factor != 0.0:
            similartiyLossName = config.getParam("similarityLoss")
            self.similarityLoss = LossFactory.lossMap[similartiyLossName]()
            self.lossWrapper.setLossFactor("sim_loss", sim_factor)
        else:
            self.similarityLoss = LossFactory.lossMap["Dummy"]()

        diceLoss = config.getParam("labelLoss")
        if diceLoss is not None and diceLoss in LossFactory.lossMap:
            self.diceLoss = LossFactory.lossMap[diceLoss]()
            ignoreBackground = config.getParam("ignoreBackground")
            if ignoreBackground:
                self.diceLoss.setIgnoreBackground(ignoreBackground)
        else:
            self.diceLoss = LossFactory.lossMap["Dummy"]()

        reg_factor = config.getParam("regularizationFactor")
        if reg_factor is not None and reg_factor != 0.0:
            regularizationLossName = config.getParam("regularizationLoss")
            if regularizationLossName is None:
                regularizationLossName = "BendingEnergy"
            self.regularizationLoss = LossFactory.lossMap[regularizationLossName]()
            self.lossWrapper.setLossFactor("reg_loss", reg_factor)
        else:
            self.regularizationLoss = LossFactory.lossMap["Dummy"]()

        imagePairSimilarityFactor = config.getParam("imagePairSimFactor")
        if imagePairSimilarityFactor is not None:
            self.lossWrapper.setLossFactor("pair_sim_loss", imagePairSimilarityFactor)

        imageSpaceLabelSimFactor = config.getParam("imageSpaceLabelSimFactor")
        if imageSpaceLabelSimFactor is not None:
            self.lossWrapper.setLossFactor("imgSpaceLabelLoss", imageSpaceLabelSimFactor)

        atlasPairSimilarityFactor = config.getParam("atlasPairSimFactor")
        if atlasPairSimilarityFactor is not None:
            self.lossWrapper.setLossFactor("atlas_pair_sim_loss", atlasPairSimilarityFactor)

        atlasSpaceLabelSimFactor = config.getParam("atlasSpaceLabelSimFactor")
        if atlasSpaceLabelSimFactor is not None:
            self.lossWrapper.setLossFactor("atlasSpaceLabelLoss", atlasSpaceLabelSimFactor)

        smooth_factor = config.getParam("smoothingFactor")
        if smooth_factor is not None and smooth_factor != 0.0:
            self.smoothLoss = LossFactory.lossMap["GradLoss"](penalty="l2")
            self.lossWrapper.setLossFactor("smooth_loss", smooth_factor)
        else:
            self.smoothLoss = LossFactory.lossMap["Dummy"]

        defFieldInverseConsistencyLossFactor = config.getParam("defDieldInverseConsistencyLossFactor")
        if defFieldInverseConsistencyLossFactor is not None:
            self.lossWrapper.setLossFactor("defFieldInverseConsistencyLoss", defFieldInverseConsistencyLossFactor)
            # self.defFieldInverseConsistencyLoss = LossFactory.lossMap["Dummy"]()
        # else:
        self.defFieldInverseConsistencyLoss = LossFactory.lossMap["MissingCorrespondences"](self.transformer)

        jacobianLossFactor = config.getParam("jacobianLossFactor")
        if jacobianLossFactor is not None:
            self.lossWrapper.setLossFactor("jacobianLoss", jacobianLossFactor)
        registrationGridsize = config.getParam("registrationGridsize")
        self.defFieldJacobianLoss = LossFactory.lossMap["JacobianLoss"](registrationGridsize)

        maximalDistanceForDitanceMaps = None
        if config.getParam("convertToDistanceMaps") is not None and config.getParam("convertToDistanceMaps"):
            maximalDistanceForDitanceMaps = config.getParam("maxDistanceForDistanceMaps")
        volumePreservationLossFactor = config.getParam("volumePreservationLossFactor")
        if volumePreservationLossFactor is not None:
            self.lossWrapper.setLossFactor("volumePreservationLoss", volumePreservationLossFactor)
        self.volumePreservationLoss = LossFactory.lossMap["VolumePreservationLoss"](
            registrationGridsize, maximalDistanceForDitanceMaps
        )

    def _getDefomredImages(
        self, posDeformationField, neg_flow, images, meshes, paddMode="border", interpolationType="bilinear"
    ):
        if isinstance(images, list):
            sec_src_imgs = images[::-1]
        else:
            sec_src_imgs = torch.flip(images, dims=[0])
        negFlowAndMesh = self.transformer.combineMeshesAndFlowField(meshes, neg_flow)
        secNegFlowAndMesh = torch.flip(negFlowAndMesh, dims=[0])
        transforemdImageMeshToOtherImageSpace = self.transformer.sampleImage(secNegFlowAndMesh, posDeformationField)
        return self.transformer.sampleImage(
            sec_src_imgs, transforemdImageMeshToOtherImageSpace, paddMode=paddMode, interpolationType=interpolationType
        )

    def _getDiceloss(self, label0, label1, mask=None):
        dscLoss = self.diceLoss(label0, label1, mask)
        return dscLoss

    def _getImageSpaceSimilarityLoss(self, imgs0, imgs1, mask=None):
        imgSpaceSimLoss = self.similarityLoss(imgs0, imgs1, mask)
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
        warpedAtlasLabels = self.transformer.sampleImage(atlasLabels, posDeformationFieldAtlas)
        warpedLabels = self.transformer.sampleImage(labels, negDeformationFieldImages)
        warpedImages = self.transformer.sampleImage(images, negDeformationFieldImages)

        deformedImages = self._getDefomredImages(posDeformationField, neg_flow, images, meshes)
        deformedLabels = self._getDefomredImages(posDeformationField, neg_flow, labels, meshes)

        sampledImages = self.transformer.sampleImage(images, meshes)
        sampledLabels = self.transformer.sampleImage(labels, meshes, interpolationType="nearest")

        self.lossWrapper.setLoss("reg_loss", self.regularizationLoss(pos_flow))

        deviationFroMeanJacobyMask = None
        if self.lossWrapper.lossFactors["volumePreservationLoss"] != 0.0:
            deviationFroMeanJacobyMask = self.volumePreservationLoss.getDeviationFromMeanJacobyMask(
                pos_flow.detach(), atlasLabels
            )
            deviationFroMeanJacobyMask = 1.0 - deviationFroMeanJacobyMask

        self.lossWrapper.setLoss(
            "sim_loss", self._getImageSpaceSimilarityLoss(warpedAtlas, sampledImages, deviationFroMeanJacobyMask)
        )

        self.lossWrapper.setLoss(
            "labelSimilarityLoss", self._getDiceloss(sampledLabels, warpedAtlasLabels, deviationFroMeanJacobyMask)
        )

        self.lossWrapper.setLoss(
            "labelSimilarityLossAtlasSpace", self._getDiceloss(atlasLabels, warpedLabels, deviationFroMeanJacobyMask)
        )

        self.lossWrapper.setLoss(
            "pair_sim_loss",
            self._getImageSpaceSimilarityLoss(deformedImages, sampledImages, deviationFroMeanJacobyMask),
        )

        self.lossWrapper.setLoss(
            "imgSpaceLabelLoss", self._getDiceloss(sampledLabels, deformedLabels, deviationFroMeanJacobyMask)
        )

        batch_size = meshes.shape[0]
        if (batch_size % 2) == 0:
            self.lossWrapper.setLoss(
                "atlas_pair_sim_loss",
                self._getImageSpaceSimilarityLoss(
                    warpedImages[: int(batch_size / 2)], warpedImages[int(batch_size / 2) :], deviationFroMeanJacobyMask
                ),
            )

            self.lossWrapper.setLoss(
                "atlasSpaceLabelLoss",
                self._getDiceloss(
                    warpedLabels[: int(batch_size / 2)], warpedLabels[int(batch_size / 2) :], deviationFroMeanJacobyMask
                ),
            )
        else:
            self.lossWrapper.setLoss(
                "atlas_pair_sim_loss", torch.zeros_like(self.lossWrapper.getUnweightedLoss("reg_loss"))
            )
            self.lossWrapper.setLoss(
                "atlasSpaceLabelLoss", torch.zeros_like(self.lossWrapper.getUnweightedLoss("reg_loss"))
            )

        self.lossWrapper.setLoss(
            "defFieldInverseConsistencyLoss", self.defFieldInverseConsistencyLoss(pos_flow, neg_flow)
        )

        self.lossWrapper.setLoss("jacobianLoss", self.defFieldJacobianLoss(pos_flow))

        # self.lossWrapper.setLoss("volumePreservationLoss", self.volumePreservationLoss(pos_flow, atlasLabels))
        with torch.no_grad():
            self.lossWrapper.setLoss("volumePreservationLoss", self.volumePreservationLoss(pos_flow, atlasLabels))

    def getLosses(self):
        return self.lossWrapper

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
