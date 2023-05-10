'''
Created on Apr 28, 2023

@author: fechter
'''

import pytorch_lightning as pl
import csv
import os
import torchio as tio
from torch.utils.data import random_split, DataLoader
from typing import Optional
import numpy as np
import SimpleITK as sitk
import torch

from config import Config


class AtlasDataModule(pl.LightningDataModule):
    dataModuleName="AtlasDataModule"
    
    def __init__(self, config : Config):
      super().__init__()
      self.datasetTrainingFile = config.getParam("trainingDataFile")
      self.datasetTestFile = config.getParam("testDataFile")
      self.train_subjects = []
      self.test_subjects = []
      self.batchSize = config.getParam("batchSize")
      self.train_val_ratio = config.getParam("trainValRatio")
      self.num_workers = config.getParam("numberOfWorkersDataLoader")
      self.shuffle = False
      self.imgFileNameColIdx = config.getParam("imageColIdxInTrainFile")
      self.labelFileNameColIdx = config.getParam("labelColIdxInTrainFile")
      
      
      
      self.registrationGridsize = config.getParam("registrationGridsize")
      self.registrationGridSpacing = config.getParam("registrationGridSpacing")
      
      if config.getParam("csvDelimiter"):
        self.delimiter = config.getParam("csvDelimiter")
      else:
        self.delimiter = ";"
      
      if config.getParam("sampleSizeIncrease"):
        self.sampleSizeIncrease = config.getParam("sampleSizeIncrease")
      else:
        self.sampleSizeIncrease = 1
      
      if config.getParam("sampleSizeIncreaseValidation"):
        self.sampleSizeIncreaseValidation = config.getParam("sampleSizeIncreaseValidation")
      else:
        self.sampleSizeIncreaseValidation = 1
      
    # pytorch lightning hook
    def prepare_data(self):
      pass
      
    def getInitalAtlas(self):
      return self.atlasImages.repeat(self.trainer.datamodule.batchSize, 1, 1, 1, 1)
    
    def _prepare_data(self):
      
      self.iterateFile(self.datasetTrainingFile, self.train_subjects)
      self.iterateFile(self.datasetTestFile, self.test_subjects)
      
    def iterateFile(self, inputFile, container):
      with open(inputFile) as csvDataFile:
        csvReader = csv.reader(csvDataFile,delimiter=self.delimiter)
        for row in csvReader:
          imageFileName = row[self.imgFileNameColIdx]
           
          if (os.path.exists(imageFileName)):
            labelFileName = None
            if self.labelFileNameColIdx > -1 and len(row) > self.labelFileNameColIdx:
              labelFileName = row[self.labelFileNameColIdx]
            subject = self.getSubject(imageFileName, labelFileName)
            container.append(subject)  
    
    def getSubject(self, imageFileName, labelFileName):
      subject = None
      if (os.path.exists(imageFileName)):
        scalarImage = tio.ScalarImage(imageFileName)
        labelImage = None
        if labelFileName and os.path.exists(labelFileName):
          labelImage = tio.LabelMap(labelFileName)
        
        meshName = os.path.splitext(imageFileName)[0] + "Mesh.pt"
        if os.path.exists(meshName):
          samplieMesh, sampleMeshOrigin = torch.load(meshName)
        else:
          samplieMesh, sampleMeshOrigin = self.getSampleMesh(scalarImage, labelImage)
          torch.save([samplieMesh,sampleMeshOrigin], meshName)
          
        subject = tio.Subject(image = scalarImage, label = labelImage, samplingMesh = samplieMesh, meshOrigin = sampleMeshOrigin)
      return subject
    
    def _getAugmentationTransform(self):
        augment = tio.Compose([
            tio.RandomAffine(
              scales=(0.9, 1.1),
              translation=(4,4,4),
              degrees=15,
              p=0.3),
            tio.RandomFlip(axes=('LR'),p=0.3)
        ])
        return augment             
     
    def _dataSplit(self):
      num_subjects = len(self.train_subjects)
      num_train_subjects = int(round(num_subjects * self.train_val_ratio))
      num_val_subjects = num_subjects - num_train_subjects
      splits = num_train_subjects, num_val_subjects
      train_subjects, val_subjects = random_split(self.train_subjects, splits)
      return train_subjects, val_subjects
    
    def _setAtlasImage(self):
      if len(self.train_subjects) > 0:
        tmp = self.train_subjects[0]['image'][tio.DATA]
        self.atlasImage = tmp.unsqueeze(0).detach().clone().type(torch.FloatTensor)
        self.atlasImage.requires_grad = True
      else:
        self.atlasImage = None
      
    
    def getSampleMesh(self, scalarImage, labelImage):
      
      sitkScalarImage = scalarImage.as_sitk()
      
      if labelImage:
        sitkLabelImage = labelImage.as_sitk()
        label_statistic = sitk.LabelIntensityStatisticsImageFilter()
        label_statistic.Execute(sitkLabelImage, sitkLabelImage > 0)
        centerPoint = label_statistic.GetCentroid(1)
      else:
        centerPoint = sitkScalarImage.TransformContinuousIndexToPhysicalPoint(np.asarray(sitkScalarImage.GetSize())/2.0)
      
      centerPoint = centerPoint - (np.multiply(self.registrationGridsize, self.registrationGridSpacing)/2.0)
      
      
      imgSize = torch.asarray(sitkScalarImage.GetSize())
      dirMatrix = torch.inverse(torch.Tensor(sitkScalarImage.GetDirection()).reshape([3,3]))
      orig = torch.Tensor(sitkScalarImage.GetOrigin())
      spacing = torch.Tensor(sitkScalarImage.GetSpacing())
      
      
      gridVecWorldC = [(torch.arange(s) * self.registrationGridSpacing[idx]) + centerPoint[idx] for idx, s in enumerate(self.registrationGridsize)]
      gridWorldC = torch.meshgrid(*gridVecWorldC)
      gridShape = gridWorldC[0].shape + (len(gridWorldC),)
      flatGridWorldC = [s.flatten() for s in gridWorldC]
      flatGridWorldC = torch.stack(flatGridWorldC, 1)
      gridOrigin = flatGridWorldC[0,:]
      
      flatImgC = torch.matmul(dirMatrix,((flatGridWorldC - orig)/spacing)[:,:,None])
      flatImgC = flatImgC.squeeze()
      flatImgC = (flatImgC / (imgSize - 1.0))*2.0-1.0
      
      gridImgC = flatImgC.reshape(gridShape).flip(-1)
      
      gridImgC = torch.unsqueeze(gridImgC, 0)
      gridImgC = gridImgC.type(torch.FloatTensor)
      
      return gridImgC, gridOrigin
      
     
    # pytorch lightning hook
    def setup(self, stage: Optional[str] = None):
      
      self._prepare_data()
      self._setAtlasImage()
      
      if stage == "fit" or stage is None:

        train_subjects, val_subjects = self._dataSplit()
        self.train_sampler = None
        self.validation_sampler = None
        self.shuffle = True
          
        transform = self._getAugmentationTransform()
        self.train_set = tio.SubjectsDataset(train_subjects, transform=transform)
        self.val_set = tio.SubjectsDataset(val_subjects, transform=transform)
        
        print("size of training set: ", len(self.train_set))
        print("size of validation set: ", len(self.val_set))
        
      if stage == "test" or stage is None:
        self.test_set = tio.SubjectsDataset(self.test_subjects)
     
    # pytorch lightning hook         
    def train_dataloader(self):
        return DataLoader(self.train_set, self.batchSize, num_workers=self.num_workers, shuffle=self.shuffle, sampler=self.train_sampler)

    # pytorch lightning hook
    def val_dataloader(self):
        return DataLoader(self.val_set, self.batchSize, num_workers=self.num_workers, sampler=self.validation_sampler)

    # pytorch lightning hook
    def test_dataloader(self):
        return DataLoader(self.test_set, self.batchSize, num_workers=self.num_workers)      