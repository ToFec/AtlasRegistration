'''
Created on Apr 28, 2023

@author: fechter
'''
''

import torchio as tio
import torch
import pytorch_lightning as pl
import pl_bolts
from imageTransformation import Transformation

class AtlasModule(pl.LightningModule):

    def __init__(self, net, loss, networkLearning_rate, atlasLearning_rate, networkOptimizer_class, atlasOptimizer_class, useLrScheduler):
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
        self.save_hyperparameters()
        

    def on_save_checkpoint(self, checkpoint):
      checkpoint['atlasImage'] = self.atlasImage
      checkpoint['atlasMesh'] = self.atlasMesh
      
    def on_load_checkpoint(self, checkpoint):
      self.atlasImage = checkpoint['atlasImage']
      self.atlasMesh = checkpoint['atlasMesh']      

    def on_fit_start(self)->None:
      pl.LightningModule.on_fit_start(self)
      atlasImage, atlasMesh = self.trainer.datamodule.getInitalAtlas()
      atlasImage.requires_grad = True
      self.atlasMesh = atlasMesh.to(self.device)
      self.atlasImage = atlasImage.to(self.device)
      
    def _atlasMesh(self, batch_size):
      return self.atlasMesh.expand(batch_size,-1,-1,-1,-1)
    
    def _atlasImage(self,batch_size):
      return self.atlasImage.expand(batch_size,-1,-1,-1,-1)

    def configure_optimizers(self):       
        networkOptimizer = self.networkOptimizer_class(self.net.parameters(), lr=(self.nlr or self.learning_rate))
        atlasOptimizer = self.atlasOptimizer_class(self.parameters(), lr=(self.alr or self.learning_rate))
        
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
          
        

    def prepare_batch(self, batch):
        return batch['image'][tio.DATA], batch['samplingMesh']

    def infer_batch(self, images, atlasImages):
        atlasAndImages = torch.cat((atlasImages, images), 1)
        pos_flow, neg_flow = self.net(atlasAndImages)
        return pos_flow, neg_flow

    def gatherInfoOfTrainingValidationStep(self, batch, batch_idx):
      images, meshes = self.prepare_batch(batch)
      networkImageToRegInput = self.transformer.sampleImage(images,meshes)
      networkAtlasInput = self.transformer.sampleImage(self._atlasImage(networkImageToRegInput.shape[0]), self._atlasMesh(networkImageToRegInput.shape[0]))
      pos_flow, neg_flow = self.infer_batch(networkImageToRegInput, networkAtlasInput)
      
      loss = self.criterion.getLoss(pos_flow, neg_flow, images, meshes, self._atlasImage(images.shape[0]), self._atlasMesh(images.shape[0]))
      return {"loss": loss}  
    
    def training_step(self, batch, batch_idx):
      
      optNetwork, _ = self.optimizers(use_pl_optimizer=True)

      stepInfo = self.gatherInfoOfTrainingValidationStep(batch, batch_idx)
      loss = stepInfo["loss"]
      optNetwork.zero_grad()
      self.manual_backward(loss)
      optNetwork.step()
    
      return stepInfo
        

    def validation_step(self, batch, batch_idx):
      stepInfo = self.gatherInfoOfTrainingValidationStep(batch, batch_idx)
      loss = stepInfo["loss"]
      self.log('val_loss', loss)
      return stepInfo
        
      
    def test_step(self, batch, batch_idx):
      stepInfo = self.gatherInfoOfTrainingValidationStep(batch, batch_idx)
      loss = stepInfo["loss"]
      self.log('test_loss', loss)
      return stepInfo      

     
    def epochEndLogging(self,outputs, trainValString): 
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
          
        if(self.current_epoch==1):
          exampleInputArray = torch.cat((self._atlasImage(1), self._atlasImage(1)), 1)
          self.logger.experiment.add_graph(self.net,exampleInputArray)
        
        self.logger.experiment.add_scalar("Loss/" + trainValString,avg_loss,self.current_epoch)
        
        networkAtlasInput = self.transformer.sampleImage(self._atlasImage(1), self._atlasMesh(1))
        
        networkAtlasInput = torch.Tensor.cpu(networkAtlasInput[0,0,int(networkAtlasInput.shape[2]/2),...])
        self.logger.experiment.add_image("AtlasCenterSlice",networkAtlasInput,self.current_epoch,dataformats="HW")
      
    def training_epoch_end(self,outputs):
      _ , optAtlas = self.optimizers(use_pl_optimizer=True)
      optAtlas.step()
      optAtlas.zero_grad()
      self.epochEndLogging(outputs, "Train")
        
      
    def validation_epoch_end(self,outputs):
      self.epochEndLogging(outputs, "Validation")
        
      