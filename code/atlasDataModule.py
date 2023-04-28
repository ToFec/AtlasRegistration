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
      
    
    
    def _prepare_data(self):
      
      with open(self.datasetTrainingFile) as csvDataFile:
        csvReader = csv.reader(csvDataFile,delimiter=self.delimiter)
        for row in csvReader:
          imageFileName = row[self.imgFileNameColIdx]
           
          if (file_exists(imageFileName)):
            labelImage = None
            if row[self.labelFileNameColIdx] > -1:
              labelFileName = row[self.labelFileNameColIdx]
              labelImage = tio.LabelMap(labelFileName)
            subject = tio.Subject(image = tio.ScalarImage(imageFileName), label = labelImage)
            self.train_subjects.append(subject)        
      
      
      with open(self.datasetTestFile) as csvDataFile:
        csvReader = csv.reader(csvDataFile,delimiter=self.delimiter)
        for row in csvReader:
          imageFileName = row[self.imgFileNameColIdx]
          if (file_exists(imageFileName)):
            labelImage = None
            if row[self.labelFileNameColIdx] > -1:
              labelFileName = row[self.labelFileNameColIdx]
              labelImage = tio.LabelMap(labelFileName)
            subject = tio.Subject(image = tio.ScalarImage(imageFileName), label = labelImage)
            self.test_subjects.append(subject)              

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
        self.atlasImage = tmp[tio.DATA].unsqueeze(0)
        self.atlasImage.requires_grad = True

      else:
        self.atlasImage = None
      
      
     
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