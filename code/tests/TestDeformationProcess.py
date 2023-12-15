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

        config.setParam("trainingDataFile", "./resources/DataTestTrainingMethods.csv")
        config.setParam("numberOfWorkersDataLoader", 0)
        config.setParam("registrationGridsize", [64, 56, 60])
        config.setParam("registrationGridSpacing", [1.0, 1.0, 1.0])
        config.setParam("doRandomTrainValSetSplit", False)

        defFieldITK = sitk.ReadImage("./resources/DummyDeformationFieldInv.nrrd")
        defField = atlasUtils.loadDefField("./resources/DummyDeformationFieldInv.nrrd")

        data = AtlasDataModule(config)
        data.prepare_data()
        data.setup(stage="fit")

        tmp = data.val_set[0]["image"][tio.DATA]
        data.atlasImage = tmp.unsqueeze(0).detach().clone().type(torch.FloatTensor)
        data.atlasImage.requires_grad = True

        atlasMesh = data.val_set[0]["samplingMesh"]
        data.atlasMesh = atlasMesh.unsqueeze(0).detach().clone()

        atlasImage, atlasMesh, _ = data.getInitalAtlas()

        locale.setlocale(locale.LC_NUMERIC, "en_US")
        transformer = Bilinear()

        deformaiton = atlasMesh[0, None, :] + defField
        tmpDeformed = transformer.sampleImage(atlasImage[0, None, :], deformaiton).detach()

        atlasUtils.saveDefField(
            "./resources/DummyDeformationFieldInvSaved.nrrd",
            defField,
            defFieldITK.GetOrigin(),
            defFieldITK.GetSpacing(),
            defFieldITK.GetDirection(),
        )

        atlasUtils.saveImageTensor(
            atlasImage[0, None, ...].detach(),
            "./resources/DummyOrigToBeDeformationFieldInv.nrrd",
            defFieldITK.GetOrigin(),
            defFieldITK.GetSpacing(),
            defFieldITK.GetDirection(),
        )

        atlasUtils.saveImageTensor(
            tmpDeformed[0, None, ...],
            "./resources/DummyDeformedByDeformationFieldInv.nrrd",
            defFieldITK.GetOrigin(),
            defFieldITK.GetSpacing(),
            defFieldITK.GetDirection(),
        )


if __name__ == "__main__":
    unittest.main()
