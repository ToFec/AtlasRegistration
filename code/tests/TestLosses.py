"""
Created on Apr 9, 2024

@author: fechter
"""
import unittest
from config import Config
from atlasDataModule import AtlasDataModule
from losses import LossFactory
import torchio as tio


class Test(unittest.TestCase):
    def getConfig(self, batchSize) -> Config:
        config = Config()
        config.setParam("trainingDataFile", "./resources/DscLoss/Data2.csv")
        config.setParam("numberOfWorkersDataLoader", 0)
        config.setParam("registrationGridsize", [256, 256, 256])
        config.setParam("registrationGridSpacing", [1.0, 1.0, 1.0])
        config.setParam("doRandomTrainValSetSplit", False)
        config.setParam("trainValRatio", 1.0)
        config.setParam("batchSize", batchSize)
        return config

    def testNCC(self):
        batchSize = 2
        data = AtlasDataModule(self.getConfig(batchSize))
        data.prepare_data()
        data.setup(stage="fit")

        nccLoss1 = LossFactory.lossMap["NCC"]()
        nccLoss2 = LossFactory.lossMap["NCC2"]()
        for batch in data.train_dataloader():
            labels = batch["image"][tio.DATA]

            loss1 = nccLoss1(labels[: int(batchSize / 2)], labels[: int(batchSize / 2)])
            loss2 = nccLoss2(labels[: int(batchSize / 2)], labels[: int(batchSize / 2)])
            self.assertAlmostEqual(loss1.item(), loss2.item(), 3)

            loss1 = nccLoss1(labels[: int(batchSize / 2)], labels[int(batchSize / 2) :])
            loss2 = nccLoss2(labels[: int(batchSize / 2)], labels[int(batchSize / 2) :])
            self.assertAlmostEqual(loss1.item(), loss2.item(), 3)

            loss1 = nccLoss1(labels[int(batchSize / 2) :], labels[: int(batchSize / 2)])
            loss2 = nccLoss2(labels[int(batchSize / 2) :], labels[: int(batchSize / 2)])
            self.assertAlmostEqual(loss1.item(), loss2.item(), 3)


if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
