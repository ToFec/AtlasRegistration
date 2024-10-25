"""
Created on May 10, 2023

@author: fechter
"""

import unittest
from config import Config
from atlasDataModule import AtlasDataModule
import atlas_utils as atlasUtils
from lossCalculator import LossCalculator
from atlasModule import AtlasModule
from atlas_models import SVF_resid
from imageTransformation import Transformation
import numpy as np
import torchio as tio
import torch
import SimpleITK as sitk
import locale
import os
from SimpleITK.extra import ReadImage
from TrainAtlas import runTraining


class Test(unittest.TestCase):
    def testTrainProcess(self):
        configFile = "./resources/AverageTestAtlasConfig.json"
        config = Config(configFile)
        config.setParam("epochs", 4)
        # config.setParam("accelerator", "cpu")
        config.setParam("numberOfWorkersDataLoader", 1)

        runTraining(config)

    def testLossFunction(self):
        config = Config()

        config.setParam("trainingDataFile", "./resources/DataTestTrainingMethods.csv")
        config.setParam("numberOfWorkersDataLoader", 0)
        config.setParam("registrationGridsize", [64, 56, 60])
        config.setParam("registrationGridSpacing", [1.0, 1.0, 1.0])
        config.setParam("doRandomTrainValSetSplit", False)
        config.setParam("imagePairSimFactor", 1.0)
        config.setParam("atlasPairSimFactor", 1.0)
        config.setParam("similarityFactor", 1.0)
        config.setParam("regularizationFactor", 1.0)

        data = AtlasDataModule(config)
        data.prepare_data()
        data.setup(stage="fit")

        tmp = data.val_set[0]["image"][tio.DATA]
        data.atlasImage = tmp.unsqueeze(0).detach().clone().type(torch.FloatTensor)
        data.atlasImage.requires_grad = True

        atlasMesh = data.val_set[0]["samplingMesh"]
        data.atlasMesh = atlasMesh.unsqueeze(0).detach().clone()

        lossCalculator = LossCalculator(config)

        atlasImages, atlasMeshes, _, atlasLabels = data.getInitalAtlas()

        neg_flow = atlasUtils.loadDefField("./resources/DummyDeformationField.nrrd")
        neg_flow = torch.cat((neg_flow, neg_flow))

        pos_flow = atlasUtils.loadDefField("./resources/DummyDeformationFieldInv.nrrd")
        pos_flow = torch.cat((pos_flow, pos_flow))

        locale.setlocale(locale.LC_NUMERIC, "en_US")

        for batch in data.train_dataloader():
            images, meshes, labels = batch["image"][tio.DATA], batch["samplingMesh"], batch["label"][tio.DATA]

            lossCalculator.calculateLoss(
                pos_flow,
                neg_flow,
                images,
                meshes,
                atlasImages.expand(images.shape[0], -1, -1, -1, -1),
                atlasMeshes.expand(images.shape[0], -1, -1, -1, -1),
                atlasLabels.expand(images.shape[0], -1, -1, -1, -1),
                labels,
            )
            lossValue = lossCalculator.getLoss()
            self.assertAlmostEqual(lossValue.detach().numpy(), 0.0014, delta=0.00005)

        if os.path.exists("./resources/DummyDeformedMesh.pt"):
            os.remove("./resources/DummyDeformedMesh.pt")
        if os.path.exists("./resources/DummyMesh.pt"):
            os.remove("./resources/DummyMesh.pt")
        if os.path.exists("./resources/DummyRotatedMesh.pt"):
            os.remove("./resources/DummyRotatedMesh.pt")

    def testCombinedForwardBackwardTransform(self):
        config = Config()

        config.setParam("trainingDataFile", "./resources/DataTestTrainingMethods.csv")
        config.setParam("numberOfWorkersDataLoader", 0)
        config.setParam("registrationGridsize", [64, 56, 60])
        config.setParam("registrationGridSpacing", [1.0, 1.0, 1.0])
        config.setParam("doRandomTrainValSetSplit", False)
        config.setParam("imagePairSimFactor", 1.0)
        config.setParam("atlasPairSimFactor", 1.0)
        config.setParam("similarityFactor", 1.0)
        config.setParam("regularizationFactor", 1.0)

        data = AtlasDataModule(config)
        data.prepare_data()
        data.setup(stage="fit")

        tmp = data.val_set[0]["image"][tio.DATA]
        data.atlasImage = tmp.detach().clone().type(torch.FloatTensor)
        data.atlasImage.requires_grad = True

        atlasMesh = data.val_set[0]["samplingMesh"]
        data.atlasMesh = atlasMesh.unsqueeze(0).detach().clone()

        neg_flow = atlasUtils.loadDefField("./resources/DummyDeformationField.nrrd")
        neg_flow = torch.cat((neg_flow, neg_flow))

        pos_flow = atlasUtils.loadDefField("./resources/DummyDeformationFieldInv.nrrd")
        pos_flow = torch.cat((pos_flow, pos_flow))

        locale.setlocale(locale.LC_NUMERIC, "en_US")
        transformer = Transformation()
        lossCalculator = LossCalculator(config)

        sitkReferenceImg = sitk.ReadImage("./resources/Dummy.nrrd")
        sitkReferenceArray = sitk.GetArrayFromImage(sitkReferenceImg)

        for batch in data.train_dataloader():
            images, meshes = batch["image"][tio.DATA], batch["samplingMesh"]

            posDeformationField = transformer.getDeformationField(pos_flow)
            deformedImages = lossCalculator._getDefomredImages(posDeformationField, neg_flow, images, meshes)

            for i in range(deformedImages.shape[0]):
                calculatedImageArray = deformedImages[i].detach().squeeze(0).permute([2, 1, 0])
                self.assertTrue(torch.sum(calculatedImageArray - sitkReferenceArray).numpy() < -840.0)
                self.assertTrue(torch.sum(calculatedImageArray - sitkReferenceArray).numpy() > -841.0)

        if os.path.exists("./resources/DummyDeformedMesh.pt"):
            os.remove("./resources/DummyDeformedMesh.pt")
        if os.path.exists("./resources/DummyMesh.pt"):
            os.remove("./resources/DummyMesh.pt")
        if os.path.exists("./resources/DummyRotatedMesh.pt"):
            os.remove("./resources/DummyRotatedMesh.pt")

    def testLoadAndSaveOfDefField(self):
        defFieldITK = sitk.ReadImage("./resources/DummyDeformationFieldInv.nrrd")
        defField = atlasUtils.loadDefField("./resources/DummyDeformationFieldInv.nrrd")

        atlasUtils.saveDefField(
            "./resources/DummyDeformationFieldInvSaved.nrrd",
            defField,
            defFieldITK.GetOrigin(),
            defFieldITK.GetSpacing(),
            defFieldITK.GetDirection(),
        )

    def testBackwardDeformation(self):
        config = Config()

        config.setParam("trainingDataFile", "./resources/DataTestTrainingMethods.csv")
        config.setParam("numberOfWorkersDataLoader", 0)
        config.setParam("registrationGridsize", [64, 56, 60])
        config.setParam("registrationGridSpacing", [1.0, 1.0, 1.0])
        config.setParam("doRandomTrainValSetSplit", False)

        data = AtlasDataModule(config)
        data.prepare_data()
        data.setup(stage="fit")

        tmp = data.val_set[0]["image"][tio.DATA]
        data.atlasImage = tmp.unsqueeze(0).detach().clone().type(torch.FloatTensor)
        data.atlasImage.requires_grad = True

        atlasMesh = data.val_set[0]["samplingMesh"]
        data.atlasMesh = atlasMesh.unsqueeze(0).detach().clone()

        atlasImage, atlasMesh, _, _ = data.getInitalAtlas()

        defField = atlasUtils.loadDefField("./resources/DummyDeformationFieldInv.nrrd")

        locale.setlocale(locale.LC_NUMERIC, "en_US")
        transformer = Transformation()

        deformaiton = transformer.combineMeshesAndFlowField(atlasMesh[0, None, :], defField)
        tmpDeformed = transformer.sampleImage(atlasImage[0, None, :], deformaiton)

        calculatedImageArray = tmpDeformed.detach().squeeze(0).squeeze(0).permute([2, 1, 0])

        # meshOrigin = data.val_set[0]['meshOrigin']
        # meshSpacing = config.getParam("registrationGridSpacing")
        # meshDir = [1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0]
        # sitkImage = sitk.GetImageFromArray(tmpDeformed.detach().squeeze(0).squeeze(0).permute([2,1,0]))
        # sitkImage.SetOrigin(meshOrigin.tolist())
        # sitkImage.SetDirection(meshDir)
        # sitkImage.SetSpacing(meshSpacing)
        # sitk.WriteImage(sitkImage, "./resources/DummyDeformedInvTest.nrrd")

        sitkReferenceImg = sitk.ReadImage("./resources/Dummy.nrrd")
        sitkReferenceArray = sitk.GetArrayFromImage(sitkReferenceImg)

        self.assertTrue(torch.sum(calculatedImageArray - sitkReferenceArray).numpy() < -117.0)
        self.assertTrue(torch.sum(calculatedImageArray - sitkReferenceArray).numpy() > -118.0)

        if os.path.exists("./resources/DummyDeformedMesh.pt"):
            os.remove("./resources/DummyDeformedMesh.pt")
        if os.path.exists("./resources/DummyMesh.pt"):
            os.remove("./resources/DummyMesh.pt")
        if os.path.exists("./resources/DummyRotatedMesh.pt"):
            os.remove("./resources/DummyRotatedMesh.pt")

    def testForwardDeformation(self):
        config = Config()

        config.setParam("trainingDataFile", "./resources/DataTestTrainingMethods.csv")
        config.setParam("numberOfWorkersDataLoader", 0)
        config.setParam("registrationGridsize", [64, 56, 60])
        config.setParam("registrationGridSpacing", [1.0, 1.0, 1.0])
        config.setParam("doRandomTrainValSetSplit", False)

        data = AtlasDataModule(config)
        data.prepare_data()
        data.setup(stage="fit")

        defField = atlasUtils.loadDefField("./resources/DummyDeformationField.nrrd")

        locale.setlocale(locale.LC_NUMERIC, "en_US")
        transformer = Transformation()
        for batch in data.train_dataloader():
            images, meshes = batch["image"][tio.DATA], batch["samplingMesh"]

            deformaiton = transformer.combineMeshesAndFlowField(meshes[0, None, :], defField)
            tmpDeformed = transformer.sampleImage(images[0, None, :], deformaiton)

            # meshOrigin = data.train_set[0]['meshOrigin']
            # meshSpacing = config.getParam("registrationGridSpacing")
            # meshDir = [1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0]
            # sitkImage = sitk.GetImageFromArray(tmpDeformed.squeeze(0).squeeze(0).permute([2,1,0]))
            # sitkImage.SetOrigin(meshOrigin.tolist())
            # sitkImage.SetDirection(meshDir)
            # sitkImage.SetSpacing(meshSpacing)
            # sitk.WriteImage(sitkImage, "./resources/DummyDeformedTest.nrrd")
            #

            calculatedImageArray = tmpDeformed.detach().squeeze(0).squeeze(0).permute([2, 1, 0])

            sitkReferenceImg = sitk.ReadImage("./resources/DummyDeformed.nrrd")
            sitkReferenceArray = sitk.GetArrayFromImage(sitkReferenceImg)

            self.assertTrue(torch.sum(calculatedImageArray - sitkReferenceArray).numpy() > -3802.0)
            self.assertTrue(torch.sum(calculatedImageArray - sitkReferenceArray).numpy() < -3801.0)

        if os.path.exists("./resources/DummyDeformedMesh.pt"):
            os.remove("./resources/DummyDeformedMesh.pt")
        if os.path.exists("./resources/DummyMesh.pt"):
            os.remove("./resources/DummyMesh.pt")
        if os.path.exists("./resources/DummyRotatedMesh.pt"):
            os.remove("./resources/DummyRotatedMesh.pt")

    def testBatchMethods(self):
        config = Config()

        config.setParam("trainingDataFile", "./resources/DataTestTrainingMethods.csv")
        config.setParam("numberOfWorkersDataLoader", 0)
        config.setParam("registrationGridsize", [64, 56, 60])
        config.setParam("registrationGridSpacing", [1.0, 1.0, 1.0])
        config.setParam("doRandomTrainValSetSplit", False)

        network = SVF_resid()
        newShape = network.getShapeForModel(config.getParam("registrationGridsize"))
        # config.setParam("registrationGridsize", newShape.tolist())

        data = AtlasDataModule(config)
        data.prepare_data()
        data.setup(stage="fit")

        loss = LossCalculator(config)
        networkOptim = atlasUtils.getOptimizer(config.getParam("optimizer"))
        atlasOptim = atlasUtils.getOptimizer(config.getParam("optimizer"))

        atlasImage, atlasMesh, atlasOrigin, atlasLabel = data.getInitalAtlas()

        model = AtlasModule(
            network,
            atlasImage,
            atlasLabel,
            atlasMesh,
            atlasOrigin,
            loss,
            networkLearning_rate=config.getParam("learningRate"),
            atlasLearning_rate=config.getParam("atlasLearningRate"),
            networkOptimizer_class=networkOptim,
            atlasOptimizer_class=atlasOptim,
            useLrScheduler=config.getParam("lrScheduler"),
        )

        model.setup("fit")
        model.configure_optimizers()
        model.atlasImages, model.atlasMeshes, _, _ = data.getInitalAtlas()

        defField = atlasUtils.loadDefField("./resources/DummyDeformationField.nrrd")
        # defField = atlasUtils.loadDefField("./resources/DummyDeformationFieldInv.nrrd")

        locale.setlocale(locale.LC_NUMERIC, "en_US")
        # for batch in data.train_dataloader():
        #     _, meshes, _ = model.prepare_batch(batch)

        # networkInputImages = model.transformer.sampleImage(images,meshes)
        # netWorkInputAtlasImages = model.transformer.sampleImage(model.atlasImages, model.atlasMeshes)
        # pos_flow, neg_flow = model.infer_batch(networkInputImages, netWorkInputAtlasImages)

        # loss = model.criterion.getLoss(pos_flow, neg_flow, images, meshes, model.atlasImages, model.atlasMeshes)

        # deformaiton = model.transformer.combineMeshesAndFlowField(meshes[0, None, :], defField)
        # tmpDeformed = model.transformer.sampleImage(images[0, None, :], deformaiton)

        # meshOrigin = data.train_set[0]['meshOrigin']
        # meshSpacing = config.getParam("registrationGridSpacing")
        # meshDir = [1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0]
        # sitkImage = sitk.GetImageFromArray(tmpDeformed.squeeze(0).squeeze(0).permute([2,1,0]))
        # sitkImage.SetOrigin(meshOrigin.tolist())
        # sitkImage.SetDirection(meshDir)
        # sitkImage.SetSpacing(meshSpacing)
        # sitk.WriteImage(sitkImage, "./resources/gridTest.nrrd")

        if os.path.exists("./resources/DummyDeformedMesh.pt"):
            os.remove("./resources/DummyDeformedMesh.pt")
        if os.path.exists("./resources/DummyMesh.pt"):
            os.remove("./resources/DummyMesh.pt")
        if os.path.exists("./resources/DummyRotatedMesh.pt"):
            os.remove("./resources/DummyRotatedMesh.pt")

    def asdftestTraining(self):
        atlasUtils.setSeeds(1234)
        config = Config()
        data = AtlasDataModule(config)
        config.setParam("registrationGridsize", [64, 56, 60])
        config.setParam("registrationGridSpacing", [1.0, 1.0, 1.0])
        data.prepare_data()
        data.setup(stage="fit")

        initialAtlas = data.getInitalAtlas()

        loss = LossCalculator(config)
        optimizer = atlasUtils.getOptimizer(config.getParam("optimizer"))
        network = SVF_resid(img_sz=np.array(config.getParam("registrationGridsize")))

        # model = AtlasModule(
        #     net=network,
        #     criterion=loss,
        #     learning_rate=config.getParam('learningRate'),
        #     optimizer_class=optimizer,
        #     useLrScheduler=config.getParam('lrScheduler'),
        #     initialAtlas
        # )
        #
        # stringForStoringVariables="AtlasRegistrationTest"
        # checkpoint_callback = pl.callbacks.ModelCheckpoint(dirpath='./checkpoints/',
        #                                                    filename= stringForStoringVariables + '-{epoch:02d}-{val_loss:.2f}',
        #                                                    every_n_epochs=config.getParam("saveEveryEpoch"),
        #                                                    monitor="val_loss",
        #                                                    mode="min",
        #                                                    save_top_k=3)
        # callBackFunctions=[]
        # callBackFunctions.append(checkpoint_callback)
        #
        # lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='step')
        # callBackFunctions.append(lr_monitor)
        #
        # logger = TensorBoardLogger("tb_logs", name=stringForStoringVariables)
        #
        # trainer = pl.Trainer(
        #       accelerator=config.getParam("accelerator"),
        #       precision=32,
        #       callbacks=callBackFunctions,
        #       auto_lr_find=config.getParam('tuneLR'),
        #       # profiler="simple",
        #       logger=logger,
        #       deterministic=True,
        #       check_val_every_n_epoch=5
        #   )
        #
        # trainer.tune(model,datamodule=data)
        #
        # trainer.logger._default_hp_metric = False
        #
        # start = datetime.now()
        #
        # print('Training started at', start)
        # trainer.fit(model=model, datamodule=data)
        # print('Training duration:', datetime.now() - start)


if __name__ == "__main__":
    unittest.main()
