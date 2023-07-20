'''
Created on Apr 28, 2023

@author: fechter
'''
from ImageLogger import ImageLogger
''

import torchio as tio
import torch
import pytorch_lightning as pl
import pl_bolts
from imageTransformation import Transformation

class AtlasModule(pl.LightningModule):

    def __init__(self, net, atlasImage, atlasMesh, atlasOrigin, loss, networkLearning_rate, atlasLearning_rate, networkOptimizer_class, atlasOptimizer_class, useLrScheduler):
        super().__init__()
        self.automatic_optimization = False
        self.nlr = networkLearning_rate
        self.alr = atlasLearning_rate
        self.net = net
        self.criterion = loss
        self.networkOptimizer_class = networkOptimizer_class
        self.atlasOptimizer_class = atlasOptimizer_class
        self.lrScheduler = useLrScheduler
        self.transformer = Transformation()
        self.save_hyperparameters(logger = False)
        self.atlasImage = torch.nn.Parameter(atlasImage)
        # self.atlasImage.requires_grad = True
        self.register_buffer("atlasMesh", atlasMesh, True)
        self.register_buffer("atlasOrigin", atlasOrigin, True)
        # self.atlasMesh = torch.nn.Parameter(atlasMesh)
        # self.atlasMesh.requires_grad = False
        # self.register_parameter("tmp", self.tmp)
        # self.register_parameter("atlasMesh", atlasMesh)    

        

    # def on_save_checkpoint(self, checkpoint):
    #   checkpoint['atlasImage'] = self.atlasImage
    #   checkpoint['atlasMesh'] = self.atlasMesh
      
    # def on_load_checkpoint(self, checkpoint):
    #   self.atlasImage = checkpoint['atlasImage']
    #   self.atlasMesh = checkpoint['atlasMesh']      

    # def setup(self,stage):
    #   if stage == 'fit':
    #     atlasImage, atlasMesh = self.trainer.datamodule.getInitalAtlas()
    #     atlasImage.requires_grad = True
    #     self.atlasMesh = atlasMesh.to(self.device)
    #     self.atlasImage = atlasImage.to(self.device)
    #
    # def on_fit_start(self)->None:
    #   pl.LightningModule.on_fit_start(self)
    #   self.atlasMesh = self.atlasMesh.to(self.device)
    #   self.atlasImage = self.atlasImage.to(self.device)
      
    def getInputAtlasMesh(self, batch_size):
      return self.atlasMesh.expand(batch_size,-1,-1,-1,-1)
    
    def getInputAtlasImage(self,batch_size, detached=False):
      if detached:
        return self.atlasImage.expand(batch_size,-1,-1,-1,-1).detach()
      else:
        return self.atlasImage.expand(batch_size,-1,-1,-1,-1)

    def configure_optimizers(self):       
        networkOptimizer = self.networkOptimizer_class(self.net.parameters(), lr=self.nlr)
        # atlasOptimizer = self.atlasOptimizer_class(self.parameters(), lr=(self.alr or self.learning_rate))
        atlasOptimizer = self.atlasOptimizer_class([self.atlasImage], lr=self.alr)
        
        if self.lrScheduler:
          lr_scheduler_net = {
            'scheduler': pl_bolts.optimizers.lr_scheduler.LinearWarmupCosineAnnealingLR(networkOptimizer,warmup_epochs=10,max_epochs=100,warmup_start_lr=1e-6, eta_min=1e-6),
            'name' : 'WarumUpCosineLR'
          }
          lr_scheduler_atlas = {
            'scheduler': pl_bolts.optimizers.lr_scheduler.LinearWarmupCosineAnnealingLR(atlasOptimizer,warmup_epochs=10,max_epochs=100,warmup_start_lr=1e-6, eta_min=1e-6),
            'name' : 'WarumUpCosineLR'
          }
        else:
          lr_scheduler_net = {
            'scheduler': torch.optim.lr_scheduler.ConstantLR(networkOptimizer,factor= 1.0, total_iters = 0),
            'name' : 'ConstantLR'
          }
          lr_scheduler_atlas = {
            'scheduler': torch.optim.lr_scheduler.ConstantLR(atlasOptimizer,factor= 1.0, total_iters = 0),
            'name' : 'ConstantLR'
          }
          
        return (
            {"optimizer": networkOptimizer,"lr_scheduler": lr_scheduler_net},
            {"optimizer": atlasOptimizer, "lr_scheduler": lr_scheduler_atlas},
        )
          
        

    def _createNetworkInput(self, images, meshes):
        networkImageToRegInput = self.transformer.sampleImage(images,meshes)
        networkAtlasInput = self.transformer.sampleImage(self.getInputAtlasImage(networkImageToRegInput.shape[0], detached=True), self.getInputAtlasMesh(networkImageToRegInput.shape[0]))
        return networkImageToRegInput, networkAtlasInput
      
    def prepare_batch(self, batch):
        images= batch['image'][tio.DATA]
        meshes =  batch['samplingMesh']
        return images, meshes

    def infer_batch(self, images, atlasImages):
        atlasAndImages = torch.cat((atlasImages, images), 1)
        pos_flow, neg_flow = self.net(atlasAndImages)
        return pos_flow, neg_flow


    def gatherInfoOfTrainingValidationStep(self, pos_flow, neg_flow, images, meshes):
      sim_loss, reg_loss, pair_sim_loss, atlas_pair_sim_loss = self.criterion.getLosses(pos_flow, neg_flow, images, meshes, self.getInputAtlasImage(images.shape[0]), self.getInputAtlasMesh(images.shape[0]))
      loss = sim_loss + reg_loss + pair_sim_loss + atlas_pair_sim_loss
      return {"loss": loss, "sim_loss": sim_loss, "reg_loss": reg_loss, "pair_sim_loss": pair_sim_loss, "atlas_pair_sim_loss": atlas_pair_sim_loss}  
    
    def training_step(self, batch, batch_idx):
      
      optNetwork, _ = self.optimizers(use_pl_optimizer=True)

      images, meshes = self.prepare_batch(batch)
      networkImageToRegInput, networkAtlasInput = self._createNetworkInput(images, meshes)
      pos_flow, neg_flow = self.infer_batch(networkImageToRegInput, networkAtlasInput)
      
      stepInfo = self.gatherInfoOfTrainingValidationStep(pos_flow, neg_flow, images, meshes)
      loss = stepInfo["loss"]
      optNetwork.zero_grad()
      self.manual_backward(loss)
      optNetwork.step()
    
      return stepInfo

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
      images, meshes = self.prepare_batch(batch)
      networkImageToRegInput, networkAtlasInput = self._createNetworkInput(images, meshes)
      pos_flow, neg_flow = self.infer_batch(networkImageToRegInput, networkAtlasInput)
      return pos_flow, neg_flow

    def validation_step(self, batch, batch_idx):
      images, meshes = self.prepare_batch(batch)
      networkImageToRegInput, networkAtlasInput = self._createNetworkInput(images, meshes)
      pos_flow, neg_flow = self.infer_batch(networkImageToRegInput, networkAtlasInput)
      stepInfo = self.gatherInfoOfTrainingValidationStep(pos_flow, neg_flow, images, meshes)

      self.log('val_loss', stepInfo["loss"])
      self.log('val_sim_loss', stepInfo["sim_loss"])
      self.log('val_reg_loss', stepInfo["reg_loss"])
      self.log('val_pair_sim_loss', stepInfo["pair_sim_loss"])
      self.log('val_atlas_pair_sim_loss', stepInfo["atlas_pair_sim_loss"])
      
      imgSpaceDsc, atlasSpacedsc = self.criterion.getDiceLosses(pos_flow, neg_flow, batch['label'][tio.DATA], meshes)
      
      self.log('val_img_space_dsc', 1 - imgSpaceDsc)
      self.log('val_atlas_space_dsc', 1 - atlasSpacedsc)
      
      return stepInfo
        
      
    def test_step(self, batch, batch_idx):
      images, meshes = self.prepare_batch(batch)
      networkImageToRegInput, networkAtlasInput = self._createNetworkInput(images, meshes)
      pos_flow, neg_flow = self.infer_batch(networkImageToRegInput, networkAtlasInput)
      stepInfo = self.gatherInfoOfTrainingValidationStep(pos_flow, neg_flow, images, meshes)
      
      loss = stepInfo["loss"]
      self.log('test_loss', loss)
      self.log('test_sim_loss', stepInfo["sim_loss"])
      self.log('test_reg_loss', stepInfo["reg_loss"])
      self.log('test_pair_sim_loss', stepInfo["pair_sim_loss"])
      self.log('test_atlas_pair_sim_loss', stepInfo["atlas_pair_sim_loss"])
      
      imgSpaceDsc, atlasSpacedsc = self.criterion.getDiceLosses(pos_flow, neg_flow, batch['label'][tio.DATA], meshes)
      
      self.log('test_img_space_dsc', 1 - imgSpaceDsc)
      self.log('test_atlas_space_dsc', 1 - atlasSpacedsc)
      
      return stepInfo      

     
    def epochEndLogging(self,outputs, trainValString): 
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
          
        if(self.current_epoch==1):
          networkAtlasInput = self.transformer.sampleImage(self.getInputAtlasImage(1), self.getInputAtlasMesh(1))
          exampleInputArray = torch.cat((networkAtlasInput, networkAtlasInput), 1)
          self.logger.experiment.add_graph(self.net,exampleInputArray)
        
        self.logger.experiment.add_scalar("Loss/" + trainValString,avg_loss,self.current_epoch)
        
        networkAtlasInput = self.transformer.sampleImage(self.getInputAtlasImage(1), self.getInputAtlasMesh(1))
        
        networkAtlasInput = torch.Tensor.cpu(networkAtlasInput.detach())
        self.logger.experiment.add_image("AtlasCenterSlice",networkAtlasInput[0,0,int(networkAtlasInput.shape[2]/2),...],self.current_epoch,dataformats="HW")
        self.log_dict({"test":123})
        for logger in self.loggers:
          if isinstance(logger, ImageLogger):
            logger.saveImage(networkAtlasInput, "AtlasImage", self.current_epoch)
          
      
    def training_epoch_end(self,outputs):
      ###
      ##in case of heterogeneous/speckled atlas, a smooth loss can be used here
      ####
      _ , optAtlas = self.optimizers(use_pl_optimizer=True)
      optAtlas.step()
      optAtlas.zero_grad()
      self.epochEndLogging(outputs, "Train")
        
      
    def validation_epoch_end(self,outputs):
      self.epochEndLogging(outputs, "Validation")
        
      