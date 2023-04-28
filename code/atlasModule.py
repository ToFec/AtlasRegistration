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

    def __init__(self, net, loss, learning_rate, optimizer_class, useLrScheduler):
        super().__init__()
        self.save_hyperparameters()
        self.lr = learning_rate
        self.net = net
        self.criterion = loss
        self.optimizer_class = optimizer_class
        self.lrScheduler = useLrScheduler

    def on_fit_start(self):
      initialAtlas = self.trainer.datamodule.atlasImage
      self.atlasImages = initialAtlas.repeat(self.trainer.datamodule.batchSize, 1, 1, 1, 1)

    def configure_optimizers(self):       
        optimizer = self.optimizer_class(self.parameters(), lr=(self.lr or self.learning_rate))
        if self.lrScheduler:
          lr_scheduler = {
            'scheduler': pl_bolts.optimizers.lr_scheduler.LinearWarmupCosineAnnealingLR(optimizer,warmup_epochs=10,max_epochs=100,warmup_start_lr=1e-6, eta_min=1e-6),
            'name' : 'WarumUpCosineLR'
          }
        else:
          lr_scheduler = {
            'scheduler': torch.optim.lr_scheduler.ConstantLR(optimizer,factor= 1.0, total_iters = 0),
            'name' : 'ConstantLR'
          }
        return [optimizer], [lr_scheduler]
        

    def prepare_batch(self, batch):
        return batch['image'][tio.DATA], batch['label'][tio.DATA]

    def infer_batch(self, images):
        atlasAndImages = torch.cat((self.atlasImages, images), 1)
        pos_flow, neg_flow = self.net(atlasAndImages)
        return pos_flow, neg_flow


    def gatherInfoOfTrainingValidationStep(self, batch, batch_idx):
      images, _ = self.prepare_batch(batch)
      pos_flow, neg_flow = self.infer_batch(images)
      
      loss = self.criterion(y_hat, y)
      
      with torch.no_grad():
        tmp = torch.zeros_like(y,requires_grad=False)
        tmp[y_hat > 0] = 1
        correct = tmp.eq(y).sum().item()
        
        sensi = tmp[y > 0].eq(y[y > 0]).sum().item()
        sensiTotal = torch.count_nonzero(y > 0).item()
        spezi = tmp[y < 1].eq(y[y < 1]).sum().item()
        speziTotal = torch.count_nonzero(y < 1).item()
        
      total=len(y)
      
      batch_dictionary={
        "loss": loss, "correct": correct, "total": total, "sensitivity": sensi, "specificity" : spezi, "sensiTotal": sensiTotal, "speziTotal": speziTotal}
      
      return batch_dictionary  
    
    def training_step(self, batch, batch_idx):
      return self.gatherInfoOfTrainingValidationStep(batch, batch_idx)  
        

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
        correct = sum([x["correct"] for  x in outputs])
        total=sum([x["total"] for  x in outputs])
        
        sensi = sum([x["sensitivity"] for  x in outputs])
        sensiTotal=sum([x["sensiTotal"] for  x in outputs])
        if sensiTotal > 0:
          sensitivity = sensi /sensiTotal
        else:
          sensitivity = 0
        
        
        speci = sum([x["specificity"] for  x in outputs])
        speziTotal=sum([x["speziTotal"] for  x in outputs])
        if speziTotal > 0:
          specificity = speci /speziTotal
        else:
          specificity = 0
          
        if(self.current_epoch==1):
          self.logger.experiment.add_graph(self.net,self.exampleInputArray)
        
        self.logger.experiment.add_scalar("Loss/" + trainValString,avg_loss,self.current_epoch)
        self.logger.experiment.add_scalar("Accuracy/" + trainValString,correct/total,self.current_epoch)
        self.logger.experiment.add_scalar("Sensitivity/" + trainValString,sensitivity,self.current_epoch)
        self.logger.experiment.add_scalar("Specificity/" + trainValString,specificity,self.current_epoch)
        self.logger.experiment.add_scalar("#Class0/" + trainValString,speziTotal,self.current_epoch)
        self.logger.experiment.add_scalar("#Class1/" + trainValString,sensiTotal,self.current_epoch)      
      
    def training_epoch_end(self,outputs):
      self.epochEndLogging(outputs, "Train")
        
      
    def validation_epoch_end(self,outputs):
      self.epochEndLogging(outputs, "Validation")
        
    def _calculateLoss(self):
      
      sim_loss = get_sim_loss(svf_warped_atlas_imgs, src_imgs, loss_name)
      reg_loss = get_reg_loss(pos_flow)
      
      if atlas_pair_sim_factor != 0.0:
          atlas_pair_sim_loss = get_pair_sim_loss(svf_warped_src_imgs, loss_name)
  
      if image_pair_sim_factor != 0.0:
          image_pair_sim_loss = get_pair_sim_loss_image_space(svf_warped_src_imgs_in_image_space, sec_src_imgs, loss_name)
      y_hat, y = self.infer_batch(batch)
      loss = self.criterion(y_hat, y)
      loss = sim_factor * sim_loss + reg_factor * reg_loss + pair_sim_factor * pair_sim_loss
      return loss
      