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
from imageTransformation import Transformation

import imageTransformation


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
      self.randomSplit = config.getParam("doRandomTrainValSetSplit")
      self.doAugmentation = config.getParam("doDataAugmentation")
      self.doNormalisation = config.getParam("doNormalisation")
      self.initializeAtlasWithAverageImg = config.getParam("initializeAtlasWithAverageImg")
      
      
      
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
      return self.atlasImage.repeat(self.batchSize, 1, 1, 1, 1), self.atlasMesh.repeat(self.batchSize, 1, 1, 1, 1)
    
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
        
        subjectDict = {"image": scalarImage}
        
        labelImage = None
        if labelFileName and os.path.exists(labelFileName):
          labelImage = tio.LabelMap(labelFileName)
        
        meshName = os.path.splitext(imageFileName)[0] + "Mesh.pt"
        if os.path.exists(meshName):
          sampleMesh, sampleMeshOrigin = torch.load(meshName)
        else:
          sampleMesh, sampleMeshOrigin = self.getSampleMesh(scalarImage, labelImage)
          torch.save([sampleMesh,sampleMeshOrigin], meshName)
        
        if labelImage is None:
          labelData = self._craeteLabelImage(scalarImage, sampleMesh)
          labelImage = tio.LabelMap(tensor=labelData)
        subjectDict["label"] = labelImage
        
        subjectDict["samplingMesh"] = sampleMesh
        subjectDict["meshOrigin"] = sampleMeshOrigin
          
        subject = tio.Subject(subjectDict)
      return subject
    
    def _craeteLabelImage(self,scalarImage, mesh):
      imageData = scalarImage[tio.DATA]
      
      mesh = (mesh + 1.0) / 2.0
      tmp = (mesh >= 0.0).all(axis=0)
      mesh = mesh[:,tmp]
      tmp = (mesh <= 1.0).all(axis=0)
      mesh = mesh[:,tmp]
      for dim in range(mesh.shape[0]):
        mesh[dim] = mesh[dim] * (imageData.shape[-3+dim] - 1.0)
      
      meshFloor = torch.round(mesh).type(torch.int32)
      labelData = torch.zeros_like(imageData)
      labelData[:,meshFloor[0,:],meshFloor[1,:],meshFloor[2,:]] = 1.0
      labelData = labelData.unsqueeze(0)
      labelData = torch.nn.functional.conv3d(labelData, weight=torch.ones([1,1,3,3,3],dtype=labelData.dtype), stride=1,padding=1)
      labelData = labelData.squeeze(0)
      labelData[labelData < 14] = 0
      labelData[labelData >= 14] = 1
      labelData = labelData.type(torch.int8)
      
      
      return labelData
      

    
    def _getAugmentationTransform(self):
      augmentations = []
      if self.doAugmentation:
        augment = tio.Compose([
            tio.RandomAffine(
              scales=(0.9, 1.1),
              translation=(4,4,4),
              degrees=15,
              p=0.3),
            tio.RandomFlip(axes=('LR'),p=0.3)
        ])
        augmentations.append(augment)
        
      if self.doNormalisation:
        transform = tio.ZNormalization(masking_method='label')
        augmentations.append(transform)
      return tio.Compose(augmentations)             
     
    def _dataSplit(self):
      num_subjects = len(self.train_subjects)
      num_train_subjects = int(round(num_subjects * self.train_val_ratio))
      num_val_subjects = num_subjects - num_train_subjects
      
      if self.randomSplit:
        splits = num_train_subjects, num_val_subjects
        train_subjects, val_subjects = random_split(self.train_subjects, splits)
      else:
        train_subjects = self.train_subjects[0:num_train_subjects]
        val_subjects = self.train_subjects[num_train_subjects:num_subjects]
      return train_subjects, val_subjects
    
    def _setAtlasImage(self):
      if len(self.train_subjects) > 0:
        if self.initializeAtlasWithAverageImg:
          imgShape = list(self.train_subjects[0]['samplingMesh'].shape)
          imgShape[0] = 1
          imgShape = [1] + imgShape
          transformer = Transformation(imgShape)
          avgImg = torch.zeros(imgShape)
          for train_subject in self.train_subjects:
            tmpImg = train_subject['image'][tio.DATA].unsqueeze(0).type(torch.FloatTensor)
            tmpMesh = train_subject['samplingMesh'].unsqueeze(0)
            sampledData = transformer.sampleImage(tmpImg, tmpMesh)
            avgImg = avgImg + sampledData
          avgImg = avgImg / len(self.train_subjects)
          self.atlasImage = avgImg[0]
          self.atlasImage.requires_grad = True
          
          self.atlasMesh = transformer.identityTransform
        else:
          tmp = self.train_subjects[0]['image'][tio.DATA]
          self.atlasImage = tmp.detach().clone().type(torch.FloatTensor)
          self.atlasImage.requires_grad = True
          
          atlasMesh = self.train_subjects[0]['samplingMesh']
          self.atlasMesh = atlasMesh.unsqueeze(0).detach().clone()
      else:
        self.atlasImage = None
        self.atlasMesh = None
    
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
      
      gridImgC = flatImgC.reshape(gridShape)#.flip(-1)
      gridImgC = torch.moveaxis(gridImgC,-1,0)
      
      #gridImgC = torch.unsqueeze(gridImgC, 0)
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
        if len(val_subjects) > 0:
          self.val_set = tio.SubjectsDataset(val_subjects, transform=transform)
        else:
          self.val_set = []
          
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