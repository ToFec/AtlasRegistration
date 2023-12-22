"""
Created on May 10, 2023

@author: fechter
"""

import unittest
from config import Config
from atlasDataModule import AtlasDataModule
import atlas_utils as atlasUtils
from imageTransformation import Bilinear
import torchio as tio
import torch
import SimpleITK as sitk
import locale


class Test(unittest.TestCase):
    def testDirectionDeformation(self):
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
        atlasImage, atlasMesh, _ = data.getInitalAtlas()

        locale.setlocale(locale.LC_NUMERIC, "en_US")
        transformer = Bilinear()
        deformaiton = transformer.getDeformationField(defField)

        # tmpDeformed = transformer.sampleImage(atlasImage[0, None, :], deformaiton).detach()
        tmpDeformed = transformer(atlasImage, deformaiton)

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
        atlasImage, atlasMesh, _ = data.getInitalAtlas()

        locale.setlocale(locale.LC_NUMERIC, "en_US")
        transformer = Bilinear()
        deformaiton = transformer.getDeformationField(defField)

        # tmpDeformed = transformer.sampleImage(atlasImage[0, None, :], deformaiton).detach()
        tmpDeformed = transformer(atlasImage, deformaiton)

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
        atlasImage, atlasMesh, _ = data.getInitalAtlas()

        locale.setlocale(locale.LC_NUMERIC, "en_US")
        transformer = Bilinear()
        deformaiton = transformer.getDeformationField(defField)

        # tmpDeformed = transformer.sampleImage(atlasImage[0, None, :], deformaiton).detach()
        tmpDeformed = transformer(atlasImage, deformaiton)

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
