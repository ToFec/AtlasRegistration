"""
Created on Dec 19, 2023

@author: fechter
"""
import unittest
from config import Config
from atlasDataModule import AtlasDataModule
from losses import LossFactory
import torchio as tio


class Test(unittest.TestCase):
    def testDiceLoss(self):
        config = Config()
        config.setParam("trainingDataFile", "./resources/DscLoss/Data.csv")
        config.setParam("numberOfWorkersDataLoader", 0)
        config.setParam("registrationGridsize", [256, 256, 256])
        config.setParam("registrationGridSpacing", [1.0, 1.0, 1.0])
        config.setParam("doRandomTrainValSetSplit", False)
        config.setParam("trainValRatio", 1.0)
        batchSize = 2
        config.setParam("batchSize", batchSize)

        data = AtlasDataModule(config)
        data.prepare_data()
        data.setup(stage="fit")

        diceLoss = LossFactory.lossMap["DiceLossMultiClass"]()
        for batch in data.train_dataloader():
            labels = batch["label"][tio.DATA]
            diceLoss = diceLoss(labels, labels)
            # diceLoss = diceLoss(labels[: int(batchSize / 2)], labels[int(batchSize / 2) :])


if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testDiceLoss']
    unittest.main()
