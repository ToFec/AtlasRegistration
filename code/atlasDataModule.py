'''
Created on Apr 28, 2023

@author: fechter
'''

import pytorch_lightning as pl
import csv
from os.path import exists as file_exists
import torchio as tio
from torch.utils.data import random_split, DataLoader
from typing import Optional

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
      self.outcomeColIdx = config.getParam("labelColIdxInTrainFile")
      
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
           
          if (file_exists(imageFileName)):
            labelFileName = None
            if row[self.labelFileNameColIdx] > -1:
              labelFileName = row[self.labelFileNameColIdx]
            subject = self.getSubject(imageFileName, labelFileName)
            container.append(subject)  
    
    def getSubject(self, imageFileName, labelFileName):
      subject = None
      if (file_exists(imageFileName)):
        scalarImage = tio.ScalarImage(imageFileName)
        labelImage = None
        if labelFileName and file_exists(labelFileName):
          labelImage = tio.LabelMap(labelFileName)
          
        samplieMesh = self.getSampleMesh(scalarImage, labelImage)
          
        subject = tio.Subject(image = scalarImage, label = labelImage, samplingMesh = samplieMesh)
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
        self.atlasImage = tmp.unsqueeze(0).detach().clone()
        self.atlasImage.requires_grad = True
      else:
        self.atlasImage = None
      
    
    def getSampleMesh(self, scalarImage, labelImage):
      from scipy import ndimage
      import numpy as np
      
      self.
      
      vectors = [torch.arange(0, s) for s in size]
      grids = torch.meshgrid(vectors)
      grid = torch.stack(grids)
      grid = torch.unsqueeze(grid, 0)
      grid = grid.type(torch.FloatTensor)
      
      ##multiply grid indices by gridSpacing / imageSpacing
      
      ## add center of brainmaks to coordinates to center grid on brain
      
      ## resample image and mask with map_coordinates method?
      
      ##TODO: when calculating loss use grid too, gridNew = grid + defField; and sample image every time image info is accessed
      ## -> metod should be moved to atlasModule  
      
      
      #other coce
      
      c,h,w = img.shape
      x, y = torch.arange(h)/(h-1), torch.arange(w)/(w-1)
      grid = torch.dstack(torch.meshgrid(x, y))*2-1
      
      sampled = F.grid_sample(img[None], grid[None])
      
      ## other code end
      
      a = np.arange(12.).reshape((4, 3))
      
      a
      array([[  0.,   1.,   2.],
             [  3.,   4.,   5.],
             [  6.,   7.,   8.],
             [  9.,  10.,  11.]])
      
      ndimage.map_coordinates(a, [[0.5, 2], [0.5, 1]], order=1)
      array([ 2.,  7.])
      
     
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