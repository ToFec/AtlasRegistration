'''
Created on May 5, 2023

@author: fechter
'''
import unittest

from config import Config
from atlasDataModule import AtlasDataModule
import torchio as tio
import torch

class Test(unittest.TestCase):


    def testGridGeneration(self):
      config = Config()
      data = AtlasDataModule(config)
      data.prepare_data()
      data.setup(stage="fit")
      firstTrainSubj = data.train_subjects[0]
      tioImage = firstTrainSubj['image']
      imageData = tioImage[tio.DATA]
      imageData = imageData.unsqueeze(0)
      mesh = firstTrainSubj['samplingMesh']
      mesh = mesh.permute([0, 2, 3, 4, 1])
      sampledImgData = torch.nn.functional.grid_sample(imageData, mesh,padding_mode="zeros")
      tioImage = tio.ScalarImage(tensor=sampledImgData.squeeze(0))
      tioImage.save("gridTest.nrrd", squeeze=True)


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testGridGeneration']
    unittest.main()