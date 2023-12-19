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
    def testLoadDeformImageAndSaveDefField(self):
        config = Config()

        config.setParam("trainingDataFile", "./resources/TestDeformOnBrain/DataTestTrainingMethods.csv")
        config.setParam("numberOfWorkersDataLoader", 0)
        config.setParam("registrationGridsize", [192, 192, 160])
        config.setParam("registrationGridSpacing", [1.0, 1.0, 1.0])
        config.setParam("doRandomTrainValSetSplit", False)

        defFieldITK = sitk.ReadImage("./resources/TestDeformOnBrain/deformationFieldRes.nrrd")
        defField = atlasUtils.loadDefField("./resources/TestDeformOnBrain/deformationFieldRes.nrrd")

        data = AtlasDataModule(config)
        data.prepare_data()
        data.setup(stage="fit")
        originalImage = data.train_set[0]["image"][tio.DATA]
        atlasImage, atlasMesh, _ = data.getInitalAtlas()

        locale.setlocale(locale.LC_NUMERIC, "en_US")
        transformer = Bilinear()
        deformaiton = transformer.getDeformationField(defField)
        # deformaiton = atlasMesh[0, None, :] + defField

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
