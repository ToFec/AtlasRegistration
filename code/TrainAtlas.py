import datetime as dt
import os
import sys
from atlasDataModule import AtlasDataModule
from atlasModule import AtlasModule
sys.path.append(os.path.realpath(".."))
import argparse

import atlas_models as atlasUtils

from atlas_models import SVF_resid
from config import Config
from lossCalculator import LossCalculator

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from ImageLogger import ImageLogger

def getCheckPointString(config):
    seed = config.getParam("seed")

    if seed is not None:
      atlasUtils.setSeeds(seed)
      
    max_epochs = config.getParam("epochs")
    batch_size = config.getParam("batchSize")
    lr = config.getParam("learningRate")
    atlas_lr = config.getParam("learningRate")
    loss_name = config.getParam("similarityLoss")

    reg_factor = config.getParam("regularizationFactor")
    sim_factor = config.getParam("similarityFactor")
    pair_sim_factor = config.getParam("imagePairSimFactor")
    smooth_factor = config.getParam("smoothingFactor")
          
    stringForStoringVariables = "atlasRegistration" + str(loss_name) \
                + '_seed_' + str(seed) \
                + '_reg_' + str(reg_factor) \
                + '_atlas_sim_' + str(sim_factor) \
                + '_pair_sim_' + str(pair_sim_factor) \
                + '_smooth_' + str(smooth_factor) \
                + '_epoch_' + str(max_epochs) \
                + '_batchsize_' + str(batch_size) \
                + '_network_lr_' + str(lr) \
                + '_atlas_lr_' + str(atlas_lr)  
    return stringForStoringVariables

def getModelAndData(config, stageType):
    seed = config.getParam("seed")

    if seed is not None:
      atlasUtils.setSeeds(seed)
      
    stringForStoringVariables = getCheckPointString(config)
    
    f = open(os.path.join(config.getParam("checkPointPath"),stringForStoringVariables + ".txt"), "r")
    checkPointPath = f.read()
    
    model = AtlasModule.load_from_checkpoint(checkPointPath)
    
    network = model.net
    newShape = network.getShapeForModel(config.getParam("registrationGridsize"))
    config.setParam("registrationGridsize", newShape.tolist())

    data = AtlasDataModule(config)
    data.prepare_data()
    data.setup(stage=stageType)
    
    return model, data
  


def runTests(config):
  model, data = getModelAndData(config, "test")
  
  logger = TensorBoardLogger("tb_logs", name="test_" + getCheckPointString(config))
  trainer = pl.Trainer(logger=logger)
    
  start = dt.datetime.now()
    
  print('Training started at', start)
  trainer.test(model=model, datamodule=data)
  print('Training duration:', dt.datetime.now() - start)  

def runPrediction(config):
  model, data = getModelAndData(config, "test")
  trainer = pl.Trainer()
  start = dt.datetime.now()
    
  print('Training started at', start)
  predictions = trainer.predict(model=model, dataloaders=data.predict_dataloader())
  print('Training duration:', dt.datetime.now() - start)  

def runTraining(config):
  
    seed = config.getParam("seed")

    if seed is not None:
      atlasUtils.setSeeds(seed)

    max_epochs = config.getParam("epochs")

    network = SVF_resid()
    newShape = network.getShapeForModel(config.getParam("registrationGridsize"))
    config.setParam("registrationGridsize", newShape.tolist())

    loss = LossCalculator(config)
    optimizer = atlasUtils.getOptimizer(config.getParam('optimizer'))
    
    data = AtlasDataModule(config)
    data.prepare_data()
    data.setup(stage="fit")
    
    atlasImage, atlasMesh = data.getInitalAtlas()
    
    
    model = AtlasModule(
        network,
        atlasImage,
        atlasMesh,
        loss,
        networkLearning_rate=config.getParam('learningRate'),
        atlasLearning_rate=config.getParam('atlasLearningRate'),
        networkOptimizer_class=optimizer,
        atlasOptimizer_class=optimizer,
        useLrScheduler=config.getParam('lrScheduler')
    )     
        
    
    
    callBackFunctions=[]
    
    stringForStoringVariables =  getCheckPointString(config)
                      
    checkpoint_callback = pl.callbacks.ModelCheckpoint(dirpath=config.getParam("checkPointPath"),
                                                       filename=stringForStoringVariables + '-{epoch:02d}-{val_loss:.2f}',
                                                       every_n_epochs=config.getParam("saveEveryEpoch"),
                                                       monitor="val_loss",
                                                       mode="min",
                                                       save_top_k=3)
    callBackFunctions.append(checkpoint_callback)
    
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='step')
    callBackFunctions.append(lr_monitor)  
    
    logger = TensorBoardLogger("tb_logs", name=stringForStoringVariables)
    meshDir = [1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0]
    meshOrigin = [0.0,0.0,0.0]
    meshSpacing = config.getParam("registrationGridSpacing")
    imageLogger = ImageLogger("tb_logs", name=stringForStoringVariables, imageOrigin=meshOrigin, imageSpacing=meshSpacing, imageDirections=meshDir, version=logger.version)
    
    trainer = pl.Trainer(
          accelerator=config.getParam("accelerator"),
          devices="auto", 
         # strategy="auto",
          precision=32,
          callbacks=callBackFunctions,
          auto_lr_find=config.getParam('tuneLR'),
          logger=[logger, imageLogger],
          deterministic="warn",
          check_val_every_n_epoch=5,
          max_epochs=max_epochs
      )
      
    trainer.tune(model,datamodule=data)
      
    trainer.logger._default_hp_metric = False
      
    start = dt.datetime.now()
      
    print('Training started at', start)
    trainer.fit(model=model, datamodule=data)
    print('Training duration:', dt.datetime.now() - start)
    
    if checkpoint_callback.best_model_path:
      f = open(os.path.join(config.getParam("checkPointPath"),stringForStoringVariables + ".txt"), "w")
      f.write(checkpoint_callback.best_model_path)
      f.close()


parser = argparse.ArgumentParser(description='Atlas Registration')
parser.add_argument("-c", "--configFile", dest="configFile", help="configuration file")
parser.add_argument("-t", "--test", dest="runTests", action="store_true", help="run tests with best model")
parser.add_argument("-p", "--predict", dest="predict", action="store_true")


if __name__ == "__main__":

    args = parser.parse_args()

    configFile = args.configFile
    if configFile:
      config = Config(configFile)
    else:
      config = Config()    
    if args.runTests:
      runTests(config)
    elif args.predict:
      runPrediction(config)
    else:
      runTraining(config)  
    
    
    