'''
Created on Apr 28, 2023

@author: fechter
'''
''

import torchio as tio
import torch
import pytorch_lightning as pl
import pl_bolts

class AtlasModule(pl.LightningModule):

    def __init__(self, net, loss, networkLearning_rate, atlasLearning_rate, networkOptimizer_class, atlasOptimizer_class, useLrScheduler, initialAtlasImg):
        super().__init__()
        self.automatic_optimization = False
        self.save_hyperparameters()
        self.nlr = networkLearning_rate
        self.alr = atlasLearning_rate
        self.net = net
        self.criterion = loss
        self.networkOptimizer_class = networkOptimizer_class
        self.atlasOptimizer_class = atlasOptimizer_class
        self.lrScheduler = useLrScheduler
        self.atlasImages = initialAtlasImg

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
        return batch['image'][tio.DATA], batch['label'][tio.DATA]

    def infer_batch(self, images, atlasImages):
        atlasAndImages = torch.cat((atlasImages, images), 1)
        pos_flow, neg_flow = self.net(atlasAndImages)
        return pos_flow, neg_flow


    def gatherInfoOfTrainingValidationStep(self, batch, batch_idx):
      images, _ = self.prepare_batch(batch)
      pos_flow, neg_flow = self.infer_batch(images, self.atlasImages)
      
      loss = self.criterion.getLoss(pos_flow, neg_flow, images,  self.atlasImages)
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
      return self.gatherInfoOfTrainingValidationStep(batch, batch_idx)  
        
      
    def test_step(self, batch, batch_idx):
        images, _ = self.prepare_batch(batch)
        y_hat, y = self.infer_batch(images)
        loss = self.criterion(y_hat, y)
        self.log('test_loss', loss)
        return loss      
     
    def epochEndLogging(self,outputs, trainValString): 
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
          
        if(self.current_epoch==1):
          self.logger.experiment.add_graph(self.net,self.exampleInputArray)
        
        self.logger.experiment.add_scalar("Loss/" + trainValString,avg_loss,self.current_epoch)
      
    def training_epoch_end(self,outputs):
      _ , optAtlas = self.optimizers(use_pl_optimizer=True)
      optAtlas.step()
      optAtlas.zero_grad()
      self.epochEndLogging(outputs, "Train")
        
      
    def validation_epoch_end(self,outputs):
      self.epochEndLogging(outputs, "Validation")
        
      