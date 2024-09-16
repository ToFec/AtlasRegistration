"""
Created on May 10, 2023

@author: fechter
"""

import unittest
from config import Config
from atlasDataModule import AtlasDataModule
import atlas_utils as atlasUtils
from imageTransformation import Transformation
import torchio as tio
import torch
import SimpleITK as sitk
import locale


class Test(unittest.TestCase):
    def getConfig(self, batchSize) -> Config:
        config = Config()
        config.setParam("trainingDataFile", "./resources/DirectionTest/DataTestTrainingMethods.csv")
        config.setParam("numberOfWorkersDataLoader", 0)
        config.setParam("registrationGridsize", [96, 96, 80])
        config.setParam("registrationGridSpacing", [2.0, 2.0, 2.0])
        config.setParam("doRandomTrainValSetSplit", False)
        config.setParam("trainValRatio", 1.0)
        config.setParam("batchSize", batchSize)
        return config

    def _testApplicationItkRegistrationMatrix3(self):
        img = sitk.ReadImage("./resources/ITKRegMatrix2/orig_1.nii.gz")
        reg = sitk.ReadTransform("./resources/ITKRegMatrix2/affineRegistrationMatrix.txt")

        atlasUtils.applyRigidRegistrationToImgHeader(img, reg)
        sitk.WriteImage(img, "./resources/ITKRegMatrix2/t1Reg.nrrd")

    def _testApplicationItkRegistrationMatrix2(self):
        img = sitk.ReadImage("./resources/ITKRegMatrix/orig.nii.gz")
        reg = sitk.ReadTransform("./resources/ITKRegMatrix/affineRegITK.txt")

        # resampledImg = atlasUtils.resampleSitkImage(img, reg)
        # sitk.WriteImage(resampledImg, "./resources/ITKRegMatrix/origRes.nii.gz")

        atlasUtils.applyRigidRegistrationToImgHeader(img, reg)
        sitk.WriteImage(img, "./resources/ITKRegMatrix/t1Reg.nrrd")

    def _testApplicationItkRegistrationMatrix(self):
        img = sitk.ReadImage("./resources/AffineReg2/t1.nrrd")
        reg = sitk.ReadTransform("./resources/AffineReg2/reg.tfm")
        # reg = sitk.ReadTransform("./resources/AffineReg2/trans222.tfm")

        # resampledImg = atlasUtils.resampleSitkImage(img, reg)
        # sitk.WriteImage(resampledImg, "./resources/AffineReg2/origRes.nii.gz")

        atlasUtils.applyRigidRegistrationToImgHeader(img, reg)
        sitk.WriteImage(img, "./resources/AffineReg2/t1Reg.nrrd")

    def _testDeformationCombination(self):
        batchSize = 2
        data = AtlasDataModule(self.getConfig(batchSize))
        data.prepare_data()
        data.setup(stage="fit")
        defformation = "y30R"
        neg_flowITK = sitk.ReadImage("./resources/DirectionTest/" + defformation + ".mhd")
        neg_flow_orig = atlasUtils.loadDefField("./resources/DirectionTest/" + defformation + ".mhd")

        transformer = Transformation()
        for batch in data.train_dataloader():
            images = batch["image"][tio.DATA]
            meshes = batch["samplingMesh"]
            neg_flow = neg_flow_orig.expand(images.shape[0], -1, -1, -1, -1)
            negDeformationFieldImages = transformer.combineMeshesAndFlowField(meshes, neg_flow)
            warpedImages = transformer.sampleImage(images, negDeformationFieldImages)
            origImages = transformer.sampleImage(images, meshes)
            for i in range(images.shape[0]):
                atlasUtils.saveImageTensor(
                    warpedImages[i, None, ...],
                    "./resources/DirectionTest/" + defformation + str(i) + "Deformed.mhd",
                    neg_flowITK.GetOrigin(),
                    neg_flowITK.GetSpacing(),
                    neg_flowITK.GetDirection(),
                )
                atlasUtils.saveImageTensor(
                    origImages[i, None, ...],
                    "./resources/DirectionTest/" + defformation + str(i) + "NotDeformed.mhd",
                    neg_flowITK.GetOrigin(),
                    neg_flowITK.GetSpacing(),
                    neg_flowITK.GetDirection(),
                )
                atlasUtils.saveDefField(
                    "./resources/DirectionTest/" + defformation + str(i) + "Saved.mhd",
                    neg_flow[i, None, ...],
                    neg_flowITK.GetOrigin(),
                    neg_flowITK.GetSpacing(),
                    neg_flowITK.GetDirection(),
                )

    def _testDirectionDeformation(self):
        config = Config()

        config.setParam("trainingDataFile", "./resources/DirectionTest/DataTestTrainingMethods.csv")
        config.setParam("numberOfWorkersDataLoader", 0)
        config.setParam("registrationGridsize", [10, 100, 500])
        config.setParam("registrationGridSpacing", [0.5, 5.0, 10.0])
        config.setParam("doRandomTrainValSetSplit", False)

        defFieldITK = sitk.ReadImage("./resources/DirectionTest/zT.mhd")
        defField = atlasUtils.loadDefField("./resources/DirectionTest/zT.mhd")

        data = AtlasDataModule(config)
        data.prepare_data()
        data.setup(stage="fit")
        originalImage = data.train_set[0]["image"][tio.DATA]
        atlasImage, atlasMesh, _, _ = data.getInitalAtlas()

        locale.setlocale(locale.LC_NUMERIC, "en_US")
        transformer = Transformation()
        deformaiton = transformer.getDeformationField(defField)

        # tmpDeformed = transformer.sampleImage(atlasImage[0, None, :], deformaiton).detach()
        tmpDeformed = transformer.sampleImage(atlasImage, deformaiton)

        atlasUtils.saveDefField(
            "./resources/DirectionTest/DefFieldSaved.mhd",
            defField,
            defFieldITK.GetOrigin(),
            defFieldITK.GetSpacing(),
            defFieldITK.GetDirection(),
        )

        atlasUtils.saveImageTensor(
            originalImage[0, None, ...].detach(),
            "./resources/DirectionTest/ImageOrig.mhd",
            defFieldITK.GetOrigin(),
            defFieldITK.GetSpacing(),
            defFieldITK.GetDirection(),
        )

        atlasUtils.saveImageTensor(
            tmpDeformed[0, None, ...],
            "./resources/DirectionTest/ImageDeformed.mhd",
            defFieldITK.GetOrigin(),
            defFieldITK.GetSpacing(),
            defFieldITK.GetDirection(),
        )

    def _testAnistropicDeformation(self):
        config = Config()

        config.setParam("trainingDataFile", "./resources/AnisotropicDefTest/DataTestTrainingMethods.csv")
        config.setParam("numberOfWorkersDataLoader", 0)
        config.setParam("registrationGridsize", [10, 100, 500])
        config.setParam("registrationGridSpacing", [0.5, 5.0, 10.0])
        config.setParam("doRandomTrainValSetSplit", False)

        defFieldITK = sitk.ReadImage("./resources/AnisotropicDefTest/xT.mhd")
        defField = atlasUtils.loadDefField("./resources/AnisotropicDefTest/xT.mhd")

        data = AtlasDataModule(config)
        data.prepare_data()
        data.setup(stage="fit")
        originalImage = data.train_set[0]["image"][tio.DATA]
        atlasImage, atlasMesh, _, _ = data.getInitalAtlas()

        locale.setlocale(locale.LC_NUMERIC, "en_US")
        transformer = Transformation()
        deformaiton = transformer.getDeformationField(defField)

        tmpDeformed = transformer.sampleImage(atlasImage, deformaiton)

        atlasUtils.saveDefField(
            "./resources/AnisotropicDefTest/DefFieldSaved.mhd",
            defField,
            defFieldITK.GetOrigin(),
            defFieldITK.GetSpacing(),
            defFieldITK.GetDirection(),
        )

        atlasUtils.saveImageTensor(
            originalImage[0, None, ...].detach(),
            "./resources/AnisotropicDefTest/ImageOrig.mhd",
            defFieldITK.GetOrigin(),
            defFieldITK.GetSpacing(),
            defFieldITK.GetDirection(),
        )

        atlasUtils.saveImageTensor(
            tmpDeformed[0, None, ...],
            "./resources/AnisotropicDefTest/ImageDeformed.mhd",
            defFieldITK.GetOrigin(),
            defFieldITK.GetSpacing(),
            defFieldITK.GetDirection(),
        )

    def _testLoadDeformImageAndSaveDefField(self):
        config = Config()

        config.setParam("trainingDataFile", "./resources/TestDeformOnBrain/DataTestTrainingMethods.csv")
        config.setParam("numberOfWorkersDataLoader", 0)
        config.setParam("registrationGridsize", [192, 192, 160])
        config.setParam("registrationGridSpacing", [1.0, 1.0, 1.0])
        config.setParam("doRandomTrainValSetSplit", False)

        defFieldITK = sitk.ReadImage("./resources/TestDeformOnBrain/LargeTranslation.nrrd")
        defField = atlasUtils.loadDefField("./resources/TestDeformOnBrain/LargeTranslation.nrrd")

        data = AtlasDataModule(config)
        data.prepare_data()
        data.setup(stage="fit")
        originalImage = data.train_set[0]["image"][tio.DATA]
        atlasImage, atlasMesh, _, _ = data.getInitalAtlas()

        locale.setlocale(locale.LC_NUMERIC, "en_US")
        transformer = Transformation()
        deformaiton = transformer.getDeformationField(defField)

        tmpDeformed = transformer.sampleImage(atlasImage, deformaiton)

        atlasUtils.saveDefField(
            "./resources/TestDeformOnBrain/DefFieldSaved.nrrd",
            defField,
            defFieldITK.GetOrigin(),
            defFieldITK.GetSpacing(),
            defFieldITK.GetDirection(),
        )

        atlasUtils.saveImageTensor(
            originalImage[0, None, ...].detach(),
            "./resources/TestDeformOnBrain/ImageOrig.nrrd",
            defFieldITK.GetOrigin(),
            defFieldITK.GetSpacing(),
            defFieldITK.GetDirection(),
        )

        atlasUtils.saveImageTensor(
            tmpDeformed[0, None, ...],
            "./resources/TestDeformOnBrain/ImageDeformed.nrrd",
            defFieldITK.GetOrigin(),
            defFieldITK.GetSpacing(),
            defFieldITK.GetDirection(),
        )


if __name__ == "__main__":
    unittest.main()
