"""
Created on May 5, 2023

@author: fechter
"""
import unittest

from config import Config
from atlasDataModule import AtlasDataModule
import torchio as tio
import torch
import os
import imageTransformation
import SimpleITK as sitk
import atlas_utils as au


class Test(unittest.TestCase):
    def testNormalisation(self):
        config = Config()
        config.setParam("registrationGridsize", [64, 56, 60])
        config.setParam("registrationGridSpacing", [1.0, 1.0, 1.0])
        config.setParam("initializeAtlasWithAverageImg", True)
        config.setParam("trainingDataFile", "./resources/DataTrainForAvgAtlasTest.csv")
        config.setParam("doNormalisation", False)
        data = AtlasDataModule(config)

        data.prepare_data()
        data.setup(stage="fit")

        for batch in data.train_dataloader():
            images, _, label = batch["image"][tio.DATA], batch["samplingMesh"], batch["label"][tio.DATA]
            images = images.type(torch.FloatTensor)
            for i in range(images.shape[0]):
                meanVal = images[i][label[i] > 0].mean()
                sdVal = images[i][label[i] > 0].std()
                self.assertNotAlmostEquals(0.0, meanVal.numpy(), 6)
                self.assertNotAlmostEquals(1.0, sdVal.numpy(), 6)

        if os.path.exists("./resources/rightMesh.pt"):
            os.remove("./resources/rightMesh.pt")
        if os.path.exists("./resources/leftMesh.pt"):
            os.remove("./resources/leftMesh.pt")
        if os.path.exists("./resources/frontMesh.pt"):
            os.remove("./resources/frontMesh.pt")
        if os.path.exists("./resources/backMesh.pt"):
            os.remove("./resources/backMesh.pt")

    def testNormalisation2(self):
        config = Config()
        config.setParam("registrationGridsize", [64, 56, 60])
        config.setParam("registrationGridSpacing", [1.0, 1.0, 1.0])
        config.setParam("initializeAtlasWithAverageImg", True)
        config.setParam("trainingDataFile", "./resources/DataTrainForAvgAtlasTest.csv")
        config.setParam("doNormalisation", True)
        data = AtlasDataModule(config)

        data.prepare_data()
        data.setup(stage="fit")

        for batch in data.train_dataloader():
            images, _, label = batch["image"][tio.DATA], batch["samplingMesh"], batch["label"][tio.DATA]
            for i in range(images.shape[0]):
                meanVal = images[i][label[i] > 0].mean()
                sdVal = images[i][label[i] > 0].std()
                self.assertAlmostEqual(0.0, meanVal.numpy(), 6)
                self.assertAlmostEqual(1.0, sdVal.numpy(), 6)

        if os.path.exists("./resources/rightMesh.pt"):
            os.remove("./resources/rightMesh.pt")
        if os.path.exists("./resources/leftMesh.pt"):
            os.remove("./resources/leftMesh.pt")
        if os.path.exists("./resources/frontMesh.pt"):
            os.remove("./resources/frontMesh.pt")
        if os.path.exists("./resources/backMesh.pt"):
            os.remove("./resources/backMesh.pt")

    def testDummyMaskGeneration(self):
        if os.path.exists("./resources/DummyRotatedMesh.pt"):
            os.remove("./resources/DummyRotatedMesh.pt")
        config = Config()
        config.setParam("registrationGridsize", [64, 56, 60])  # [24, 22, 20])
        config.setParam("registrationGridSpacing", [1.0, 1.0, 1.0])
        data = AtlasDataModule(config)
        data.prepare_data()
        data.setup(stage="fit")
        firstTrainSubj = data.train_subjects[0]
        tioImage = firstTrainSubj["image"]
        mesh = firstTrainSubj["samplingMesh"]

        labelData = data._craeteLabelImage(tioImage, mesh)

        sitkReferenceImg = sitk.ReadImage("./resources/DummyRotatedDefaultMask.nrrd")
        sitkReferenceArray = sitk.GetArrayFromImage(sitkReferenceImg)

        for i in range(labelData.shape[0]):
            calculatedImageArray = labelData[i].detach().squeeze(0).permute([2, 1, 0])
            self.assertTrue(torch.max(torch.abs(calculatedImageArray - sitkReferenceArray)).numpy() < 0.00001)

        if os.path.exists("./resources/DummyRotatedMesh.pt"):
            os.remove("./resources/DummyRotatedMesh.pt")

    def testAverageAtlasGeneration(self):
        config = Config()
        config.setParam("registrationGridsize", [64, 56, 60])
        config.setParam("registrationGridSpacing", [1.0, 1.0, 1.0])
        config.setParam("initializeAtlasWithAverageImg", True)
        config.setParam("trainingDataFile", "./resources/DataTrainForAvgAtlasTest.csv")
        config.setParam("doRandomTrainValSetSplit", False)
        config.setParam("trainValRatio", 1.0)
        data = AtlasDataModule(config)

        data.prepare_data()
        data.setup(stage="fit")

        sitkReferenceImg = sitk.ReadImage("./resources/AverageAtlasImgTest.nrrd")
        sitkReferenceArray = sitk.GetArrayFromImage(sitkReferenceImg)

        atlasImg, atlasMesh, _, _ = data.getInitalAtlas()
        transformer = imageTransformation.Transformation()
        atlasSampledImg = transformer.sampleImage(atlasImg, atlasMesh)
        for i in range(atlasSampledImg.shape[0]):
            calculatedImageArray = atlasSampledImg[i].detach().squeeze(0).permute([2, 1, 0])
            self.assertTrue(torch.max(torch.abs(calculatedImageArray - sitkReferenceArray)).numpy() < 0.00001)

        if os.path.exists("./resources/rightMesh.pt"):
            os.remove("./resources/rightMesh.pt")
        if os.path.exists("./resources/leftMesh.pt"):
            os.remove("./resources/leftMesh.pt")
        if os.path.exists("./resources/frontMesh.pt"):
            os.remove("./resources/frontMesh.pt")
        if os.path.exists("./resources/backMesh.pt"):
            os.remove("./resources/backMesh.pt")

    def testGridSampling(self):
        config = Config()
        config.setParam("trainingDataFile", "./resources/DataTestTrainingMethods.csv")
        config.setParam("registrationGridsize", [64, 56, 60])
        config.setParam("registrationGridSpacing", [1.0, 1.0, 1.0])
        config.setParam("doRandomTrainValSetSplit", False)
        config.setParam("trainValRatio", 1.0)
        config.setParam("doDataAugmentation", 1.0)
        data = AtlasDataModule(config)
        data.prepare_data()
        data.setup(stage="fit")
        firstTrainSubj = data.train_subjects[0]
        tioImage = firstTrainSubj["image"]
        imageData = tioImage[tio.DATA]
        imageData = imageData.unsqueeze(0)
        mesh = firstTrainSubj["samplingMesh"]
        mesh = mesh.unsqueeze(0)
        transformer = imageTransformation.Transformation()
        sampledImg = transformer.sampleImage(imageData, mesh)

        sitkReferenceImg = sitk.ReadImage("./resources/Dummy.nrrd")
        sitkReferenceArray = sitk.GetArrayFromImage(sitkReferenceImg)

        for i in range(sampledImg.shape[0]):
            calculatedImageArray = sampledImg[i].detach().squeeze(0).permute([2, 1, 0])
            self.assertTrue(torch.max(torch.abs(calculatedImageArray - sitkReferenceArray)).numpy() < 0.00001)

        imgShape = list(data.train_subjects[0]["samplingMesh"].shape)
        imgShape[0] = 1
        imgShape = [1] + imgShape
        transformer = imageTransformation.Transformation(imgShape)
        mesh = transformer.identityTransform
        mesh = mesh.unsqueeze(0)
        sampledImg = transformer.sampleImage(imageData, mesh)
        for i in range(sampledImg.shape[0]):
            calculatedImageArray = sampledImg[i].detach().squeeze(0).permute([2, 1, 0])
            self.assertTrue(torch.max(torch.abs(calculatedImageArray - sitkReferenceArray)).numpy() < 0.00001)

        if os.path.exists("./resources/DummyMesh.pt"):
            os.remove("./resources/DummyMesh.pt")
        if os.path.exists("./resources/DummyDeformedMesh.pt"):
            os.remove("./resources/DummyDeformedMesh.pt")
        if os.path.exists("./resources/DummyRotatedMesh.pt"):
            os.remove("./resources/DummyRotatedMesh.pt")

    def testGridGeneration2(self):
        config = Config()
        config.setParam("trainingDataFile", "./resources/DataTrain1.csv")
        config.setParam("registrationGridsize", [64, 56, 60])
        config.setParam("registrationGridSpacing", [1.0, 1.0, 1.0])
        data = AtlasDataModule(config)
        data.prepare_data()
        data.setup(stage="fit")
        firstTrainSubj = data.train_subjects[0]
        tioImage = firstTrainSubj["image"]
        imageData = tioImage[tio.DATA]
        imageData = imageData.unsqueeze(0)
        mesh = firstTrainSubj["samplingMesh"]

        mesh = mesh.unsqueeze(0)
        transformer = imageTransformation.Transformation()
        sampledImgData = transformer.sampleImage(imageData.type(torch.FloatTensor), mesh)

        sitkReferenceImg = sitk.ReadImage("./resources/Noise.nrrd")
        sitkReferenceArray = sitk.GetArrayFromImage(sitkReferenceImg)
        for i in range(sampledImgData.shape[0]):
            calculatedImageArray = sampledImgData[i].detach().squeeze(0).permute([2, 1, 0])
            self.assertTrue(torch.max(torch.abs(calculatedImageArray - sitkReferenceArray)).numpy() < 0.001)

        if os.path.exists("./resources/NoiseMesh.pt"):
            os.remove("./resources/NoiseMesh.pt")

    def testGridGeneration(self):
        config = Config()
        config.setParam("trainingDataFile", "./resources/DataTrain.csv")
        config.setParam("registrationGridsize", [64, 56, 60])
        config.setParam("registrationGridSpacing", [1.0, 1.0, 1.0])
        data = AtlasDataModule(config)
        data.prepare_data()
        data.setup(stage="fit")
        firstTrainSubj = data.train_subjects[0]
        tioImage = firstTrainSubj["image"]
        imageData = tioImage[tio.DATA]
        imageData = imageData.unsqueeze(0)
        mesh = firstTrainSubj["samplingMesh"]

        # meshOrigin = firstTrainSubj['meshOrigin']
        # meshSpacing = config.getParam("registrationGridSpacing")
        # meshDir = [1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0]
        # sampledImgData = torch.nn.functional.grid_sample(imageData.type(torch.FloatTensor), mesh,padding_mode="zeros", align_corners=True)
        # sitkImage = sitk.GetImageFromArray(sampledImgData.squeeze(0).squeeze(0).permute([2,1,0]))
        # sitkImage.SetOrigin(meshOrigin.tolist())
        # sitkImage.SetDirection(meshDir)
        # sitkImage.SetSpacing(meshSpacing)
        # sitk.WriteImage(sitkImage, "gridTest.nrrd")

        referenceMesh = torch.load("./resources/ReferenceMeshForDummyRotated.pt")

        self.assertEqual(torch.mean(mesh), torch.mean(referenceMesh))

        if os.path.exists("./resources/DummyRotatedMesh.pt"):
            os.remove("./resources/DummyRotatedMesh.pt")

    def testSignedDistanceMapGeneration(self):
        sitkLabel = sitk.ReadImage("./resources/DscLoss/Label1.nii.gz", sitk.sitkFloat32)
        sigendDistanceMapTensor = au.createSignedDistanceMap(sitkLabel)


if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testGridGeneration']
    unittest.main()
