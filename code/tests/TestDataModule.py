'''
Created on May 5, 2023

@author: fechter
'''
import unittest

from config import Config
from atlasDataModule import AtlasDataModule
import torchio as tio
import torch
import os
import SimpleITK as sitk

class Test(unittest.TestCase):


    def testGridGeneration(self):
      
      config = Config()
      data = AtlasDataModule(config)
      config.setParam("registrationGridsize", [64, 56, 60])
      config.setParam("registrationGridSpacing", [1.0, 1.0, 1.0])
      data.prepare_data()
      data.setup(stage="fit")
      firstTrainSubj = data.train_subjects[0]
      tioImage = firstTrainSubj['image']
      imageData = tioImage[tio.DATA]
      imageData = imageData.unsqueeze(0)
      mesh = firstTrainSubj['samplingMesh']
      
      # meshOrigin = firstTrainSubj['meshOrigin']
      # meshSpacing = config.getParam("registrationGridSpacing")
      # meshDir = [1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0]
      # sampledImgData = torch.nn.functional.grid_sample(imageData.type(torch.FloatTensor), mesh,padding_mode="zeros", align_corners=True)
      # sitkImage = sitk.GetImageFromArray(sampledImgData.squeeze(0).squeeze(0).permute([2,1,0]))
      # sitkImage.SetOrigin(meshOrigin.tolist())
      # sitkImage.SetDirection(meshDir)
      # sitkImage.SetSpacing(meshSpacing)
      # sitk.WriteImage(sitkImage, "gridTest.nrrd")
      
      referenceMesh = torch.load('./resources/ReferenceMeshForDummyRotated.pt')
      
      self.assertEqual(torch.mean(mesh), torch.mean(referenceMesh))
      
      if os.path.exists('./resources/DummyRotatedMesh.pt'):
        os.remove('./resources/DummyRotatedMesh.pt')


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testGridGeneration']
    unittest.main()