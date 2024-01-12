"""
Created on Jan 12, 2024

@author: fechter
"""
import unittest
from config import Config
import atlas_utils
from losses import LossFactory


class TestRegularisationLoss(unittest.TestCase):
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

    def testTranslation(self):
        defField = atlas_utils.loadDefField("./resources/DirectionTest/xyz5T.mhd")
        defField = defField.contiguous()
        regLossLoss = LossFactory.lossMap["BendingEnergy"]()
        loss = regLossLoss(defField)

        self.assertAlmostEqual(loss.item(), 0.0, 3)

    def testRotation(self):
        defField = atlas_utils.loadDefField("./resources/DirectionTest/y30R.mhd")
        defField = defField.contiguous()
        regLossLoss = LossFactory.lossMap["BendingEnergy"]()
        loss = regLossLoss(defField)

        self.assertAlmostEqual(loss.item(), 0.0, 3)

    def testOtherDeformation(self):
        defField = atlas_utils.loadDefField("./resources/DummyDeformationField.nrrd")
        defField = defField.contiguous()
        regLossLoss = LossFactory.lossMap["BendingEnergy"]()
        loss = regLossLoss(defField)

        self.assertAlmostEqual(loss.item(), 0.0, 3)


if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
