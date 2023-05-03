'''
Created on Apr 28, 2023

@author: fechter
'''
from losses import LossFactory
from imageTransformation import Bilinear
import torch

class LossCalculator():

    def __init__(self, config):
        
        similartiyLossName = config.getParam("similarityLoss")
        self.sim_factor = config.getParam("similarityFactor")
        self.similarityLoss = LossFactory.lossMap[similartiyLossName]()
          
        self.reg_factor = config.getParam("regularizationFactor")
        if self.reg_factor != 0.0:
          self.regularizationLoss = LossFactory.lossMap["BendingEnergy"]()
        else:
          self.regularizationLoss = LossFactory.lossMap["Dummy"]()
        
        self.imagePairSimilarityFactor = config.getParam("imagePairSimFactor")
        
        self.atlasPairSimilarityFactor = config.getParam("atlasPairSimFactor")
        
        self.smooth_factor = config.getParam("smoothingFactor")
        if self.smooth_factor != 0.0:
          self.smoothLoss = LossFactory.lossMap["GradLoss"](penalty='l2')
        else:
          self.smoothLoss = LossFactory.lossMap["Dummy"]
          
    
        self.transformer = Bilinear()
    
    def _getDefomredImages(self, posDeformationField, neg_flow, images):##TODO: statt die bilder und das posDeformationField zu flippen, flippe ich das neg_flow field; testen, ob immer noch passt
      # sec_pos_deform_field = torch.flip(posDeformationField, dims=[0])
      # sec_src_imgs = torch.flip(images, dims=[0])
      # svf_warped_src_imgs_in_image_space = self.transformer(images, (self.transformer((neg_flow, sec_pos_deform_field) + sec_pos_deform_field)))
      sec_neg_flow = torch.flip(neg_flow)
      warpedSrcImgsInImgSpace = self.transformer(images, (self.transformer((sec_neg_flow, posDeformationField) + posDeformationField)))
      return warpedSrcImgsInImgSpace
    
    def _getImageSpaceSimilarityLoss(self, imgs0, imgs1):
      imgSpaceSimLoss = self.similarityLoss(imgs0, imgs1)
      return imgSpaceSimLoss / imgs0.shape[0]
    

    def getLoss(self, pos_flow, neg_flow, images, atlasImages):
      
      posDeformationField = self.transformer.getDeformationField(pos_flow)
      negDeformationField = self.transformer.getDeformationField(neg_flow)
      
      warpedAtlas = self.transformer(atlasImages, posDeformationField)
      
      sim_loss = self._getImageSpaceSimilarityLoss(warpedAtlas, images) * self.sim_factor
      
      reg_loss = self.regularizationLoss(pos_flow) * self.reg_factor
      
      loss = sim_loss + reg_loss
      
      if self.imagePairSimilarityFactor != 0.0:
        deformedImages = self._getDefomredImages(posDeformationField, neg_flow, images)
        pair_sim_loss = self._getImageSpaceSimilarityLoss(deformedImages, images) * self.imagePairSimilarityFactor
        loss = loss + pair_sim_loss
        
      if self.atlasPairSimilarityFactor != 0.0: ##TODO: vergleicht nicht alle bild kombinationen, ausreichend oder umprogrammiren? 
        warpedImages = self.transformer(images, negDeformationField)
        batch_size = images.shape[0]
        atlas_pair_sim_loss = self._getImageSpaceSimilarityLoss(warpedImages[:int(batch_size/2)], images[:int(batch_size/2)]) * self.atlasPairSimilarityFactor
        loss = loss + atlas_pair_sim_loss
        
      return loss
        
        
        
        
      
      
      
      
      