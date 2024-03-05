"""
Created on Dec 19, 2023

@author: fechter
"""
import unittest
from config import Config
from atlasDataModule import AtlasDataModule
from losses import LossFactory
import torchio as tio
import torch


class Test(unittest.TestCase):
    def getConfig(self, batchSize) -> Config:
        config = Config()
        config.setParam("trainingDataFile", "./resources/DscLoss/Data.csv")
        config.setParam("numberOfWorkersDataLoader", 0)
        config.setParam("registrationGridsize", [256, 256, 256])
        config.setParam("registrationGridSpacing", [1.0, 1.0, 1.0])
        config.setParam("doRandomTrainValSetSplit", False)
        config.setParam("trainValRatio", 1.0)
        config.setParam("batchSize", batchSize)
        return config

    def _testDiceLossAllClasses(self):
        batchSize = 2
        data = AtlasDataModule(self.getConfig(batchSize))
        data.prepare_data()
        data.setup(stage="fit")

        diceLoss = LossFactory.lossMap["MultiClassSingleChannelDiceCalculator"]()
        for batch in data.train_dataloader():
            labels = batch["label"][tio.DATA]
            diceLoss = diceLoss(labels[: int(batchSize / 2)], labels[int(batchSize / 2) :])
            self.assertAlmostEqual(diceLoss.item(), 0.0648, 3)

    def testDiceLossAllClasses2(self):
        batchSize = 2
        data = AtlasDataModule(self.getConfig(batchSize))
        data.prepare_data()
        data.setup(stage="fit")

        diceLoss = LossFactory.lossMap["MultiClassMultiChannelDiceCalculator"]()
        for batch in data.train_dataloader():
            labels = batch["label"][tio.DATA]
            diceLoss = diceLoss(labels[: int(batchSize / 2)], labels[int(batchSize / 2) :])
            self.assertAlmostEqual(diceLoss.item(), 0.5501, 3)

    def testDiceLossSomeClasses(self):
        batchSize = 2
        data = AtlasDataModule(self.getConfig(batchSize))
        data.prepare_data()
        data.setup(stage="fit")

        diceLoss = LossFactory.lossMap["MultiClassSingleChannelDiceCalculator"]()
        for batch in data.train_dataloader():
            labels = batch["label"][tio.DATA]
            labels = torch.argmax(labels, dim=1, keepdim=True)
            diceLoss = diceLoss(labels[: int(batchSize / 2)], labels[int(batchSize / 2) :], torch.tensor(0))
            self.assertAlmostEqual(diceLoss.item(), 0.6825, 3)

    def testDiceLossSingleClass(self):
        batchSize = 2
        data = AtlasDataModule(self.getConfig(batchSize))
        data.prepare_data()
        data.setup(stage="fit")

        diceLoss = LossFactory.lossMap["MultiClassSingleChannelDiceCalculator"]()
        for batch in data.train_dataloader():
            labels = batch["label"][tio.DATA]
            labels = torch.argmax(labels, dim=1, keepdim=True)
            diceLoss = diceLoss(labels[: int(batchSize / 2)], labels[int(batchSize / 2) :], torch.tensor([0, 1, 2, 3]))
            self.assertAlmostEqual(diceLoss.item(), 0.8784, 3)


if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testDiceLoss']
    unittest.main()
