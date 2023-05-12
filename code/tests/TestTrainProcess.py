'''
Created on May 10, 2023

@author: fechter
'''
import unittest
from config import Config
from atlasDataModule import AtlasDataModule
import atlas_utils as atlasUtils
from lossCalculator import LossCalculator
from atlasModule import AtlasModule
from atlas_models import SVF_resid
import numpy as np
import torchio as tio
import torch
import SimpleITK as sitk
import locale
#import pytorch_lightning as pl
#from pytorch_lightning.loggers import TensorBoardLogger
#import datetime

class Test(unittest.TestCase):



    def testBatchMethods(self):
      
      config = Config()
      
      config.setParam("trainingDataFile", "./resources/DataTestTrainingMethods.csv")
      config.setParam("numberOfWorkersDataLoader",0)
      config.setParam("registrationGridsize", [64, 56, 60])
      config.setParam("registrationGridSpacing", [1.0, 1.0, 1.0])
      config.setParam("doRandomTrainValSetSplit", False)
      
      
      network = SVF_resid()
      newShape = network.getShapeForModel(config.getParam("registrationGridsize"))
      config.setParam("registrationGridsize", newShape.tolist())
      
      
      data = AtlasDataModule(config)
      data.prepare_data()
      data.setup(stage="fit") 
      
      tmp = data.val_set[0]['image'][tio.DATA]
      data.atlasImage = tmp.detach().clone().type(torch.FloatTensor)
      data.atlasImage.requires_grad = True
      
      atlasMesh = data.val_set[0]['samplingMesh']
      data.atlasMesh = atlasMesh.unsqueeze(0).detach().clone()
      
      loss = LossCalculator(config)
      networkOptim = atlasUtils.getOptimizer(config.getParam('optimizer'))
      atlasOptim = atlasUtils.getOptimizer(config.getParam('optimizer'))

      model = AtlasModule(
          network,
          loss,
          networkLearning_rate=config.getParam('learningRate'),
          atlasLearning_rate=config.getParam('learningRate'),
          networkOptimizer_class=networkOptim,
          atlasOptimizer_class=atlasOptim,
          useLrScheduler=config.getParam('lrScheduler')
      )      
      
      model.setup("fit")
      model.configure_optimizers()
      model.atlasImages, model.atlasMeshes = data.getInitalAtlas()
      
      #defField = atlasUtils.loadDefField("./resources/DummyDeformationField.nrrd")
      defField = atlasUtils.loadDefField("./resources/DummyDeformationFieldInv.nrrd")
      # locale.setlocale(locale.LC_NUMERIC, "en_US")
      for batch in data.train_dataloader():
        images, meshes = model.prepare_batch(batch)
        
        
        networkInputImages = model.transformer.sampleImage(images,meshes)
        netWorkInputAtlasImages = model.transformer.sampleImage(model.atlasImages, model.atlasMeshes)
        pos_flow, neg_flow = model.infer_batch(networkInputImages, netWorkInputAtlasImages)
      
        loss = model.criterion.getLoss(pos_flow, neg_flow, images, meshes, model.atlasImages, model.atlasMeshes)
        
        # deformaiton = meshes[0,None,:] + defField        
        # tmpDeformed = model.transformer.sampleImage(images[0,None,:],deformaiton)
        #
        # meshOrigin = data.train_set[0]['meshOrigin']
        # meshSpacing = config.getParam("registrationGridSpacing")
        # meshDir = [1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0]
        # sitkImage = sitk.GetImageFromArray(tmpDeformed.squeeze(0).squeeze(0).permute([2,1,0]))
        # sitkImage.SetOrigin(meshOrigin.tolist())
        # sitkImage.SetDirection(meshDir)
        # sitkImage.SetSpacing(meshSpacing)
        # sitk.WriteImage(sitkImage, "./resources/gridTest.nrrd")        
        

    def asdftestTraining(self):
      atlasUtils.setSeeds(1234)
      config = Config()
      data = AtlasDataModule(config)
      config.setParam("registrationGridsize", [64, 56, 60])
      config.setParam("registrationGridSpacing", [1.0, 1.0, 1.0])
      data.prepare_data()
      data.setup(stage="fit")
      
      initialAtlas = data.getInitalAtlas()
    
      loss = LossCalculator(config)
      optimizer = atlasUtils.getOptimizer(config.getParam('optimizer'))
      network = SVF_resid(img_sz=np.array(config.getParam("registrationGridsize")))
    
      # model = AtlasModule(
      #     net=network,
      #     criterion=loss,
      #     learning_rate=config.getParam('learningRate'),
      #     optimizer_class=optimizer,
      #     useLrScheduler=config.getParam('lrScheduler'),
      #     initialAtlas
      # )
      #
      # stringForStoringVariables="AtlasRegistrationTest"
      # checkpoint_callback = pl.callbacks.ModelCheckpoint(dirpath='./checkpoints/',
      #                                                    filename= stringForStoringVariables + '-{epoch:02d}-{val_loss:.2f}',
      #                                                    every_n_epochs=config.getParam("saveEveryEpoch"),
      #                                                    monitor="val_loss",
      #                                                    mode="min",
      #                                                    save_top_k=3)
      # callBackFunctions=[]
      # callBackFunctions.append(checkpoint_callback)
      #
      # lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='step')
      # callBackFunctions.append(lr_monitor)  
      #
      # logger = TensorBoardLogger("tb_logs", name=stringForStoringVariables)
      #
      # trainer = pl.Trainer(
      #       accelerator=config.getParam("accelerator"),
      #       precision=32,
      #       callbacks=callBackFunctions,
      #       auto_lr_find=config.getParam('tuneLR'),
      #       # profiler="simple",
      #       logger=logger,
      #       deterministic=True,
      #       check_val_every_n_epoch=5
      #   )
      #
      # trainer.tune(model,datamodule=data)
      #
      # trainer.logger._default_hp_metric = False
      #
      # start = datetime.now()
      #
      # print('Training started at', start)
      # trainer.fit(model=model, datamodule=data)
      # print('Training duration:', datetime.now() - start)      
        


if __name__ == "__main__":
    unittest.main()