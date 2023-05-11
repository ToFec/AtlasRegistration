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
#import pytorch_lightning as pl
#from pytorch_lightning.loggers import TensorBoardLogger
#import datetime

class Test(unittest.TestCase):


    def testBatchMethods(self):
      config = Config()
      data = AtlasDataModule(config)
      config.setParam("numberOfWorkersDataLoader",0)
      config.setParam("registrationGridsize", [64, 56, 60])
      config.setParam("registrationGridSpacing", [1.0, 1.0, 1.0])
      data.prepare_data()
      data.setup(stage="fit") 
      
      loss = LossCalculator(config)
      networkOptim = atlasUtils.getOptimizer(config.getParam('optimizer'))
      atlasOptim = atlasUtils.getOptimizer(config.getParam('optimizer'))
      network = SVF_resid(img_sz=np.array(config.getParam("registrationGridsize")))
    
      model = AtlasModule(
          network,
          loss,
          networkLearning_rate=config.getParam('learningRate'),
          atlasLearning_rate=config.getParam('learningRate'),
          networkOptimizer_class=networkOptim,
          atlasOptimizer_class=atlasOptim,
          useLrScheduler=config.getParam('lrScheduler')
      )      
      
      for batch in data.train_dataloader():
        inputs, meshes = model.prepare_batch(batch)

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