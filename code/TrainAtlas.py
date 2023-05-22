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

def runTraining(config):
  
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

    network = SVF_resid()
    newShape = network.getShapeForModel(config.getParam("registrationGridsize"))
    config.setParam("registrationGridsize", newShape.tolist())

    loss = LossCalculator(config)
    optimizer = atlasUtils.getOptimizer(config.getParam('optimizer'))
    
    data = AtlasDataModule(config)
    data.prepare_data()
    data.setup(stage="fit")
    
    
    model = AtlasModule(
        network,
        loss,
        networkLearning_rate=config.getParam('learningRate'),
        atlasLearning_rate=config.getParam('atlasLearningRate'),
        networkOptimizer_class=optimizer,
        atlasOptimizer_class=optimizer,
        useLrScheduler=config.getParam('lrScheduler')
    )     
        
    
    
    callBackFunctions=[]
    
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
                      
    checkpoint_callback = pl.callbacks.ModelCheckpoint(dirpath='./checkpoints/',
                                                       filename= stringForStoringVariables + '-{epoch:02d}-{val_loss:.2f}',
                                                       every_n_epochs=config.getParam("saveEveryEpoch"),
                                                       monitor="val_loss",
                                                       mode="min",
                                                       save_top_k=3)
    callBackFunctions.append(checkpoint_callback)
    
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='step')
    callBackFunctions.append(lr_monitor)  
    
    logger = TensorBoardLogger("tb_logs", name=stringForStoringVariables)
    
    trainer = pl.Trainer(
          accelerator=config.getParam("accelerator"),
          devices="auto", 
         # strategy="auto",
          precision=32,
          callbacks=callBackFunctions,
          auto_lr_find=config.getParam('tuneLR'),
          logger=logger,
          deterministic=True,
          check_val_every_n_epoch=5,
          
      )
      
    trainer.tune(model,datamodule=data)
      
    trainer.logger._default_hp_metric = False
      
    start = dt.datetime.now()
      
    print('Training started at', start)
    trainer.fit(model=model, datamodule=data)
    print('Training duration:', dt.datetime.now() - start)  


parser = argparse.ArgumentParser(description='Atlas Registration')
parser.add_argument("-c", "--configFile", dest="configFile", help="configuration file")


if __name__ == "__main__":

    args = parser.parse_args()

    configFile = args.configFile
    if configFile:
      config = Config(configFile)
    else:
      config = Config()    
     
    runTraining(config)  
    
    
    