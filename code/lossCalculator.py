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
    
    def _getDefomredImages(self, posDeformationField, neg_flow, images, meshes):
      sec_src_imgs = torch.flip(images, dims=[0])
      negFlowAndMesh = neg_flow + meshes
      secNegFlowAndMesh = torch.flip(negFlowAndMesh, dims=[0])
      transforemdImageMeshToOtherImageSpace = self.transformer(secNegFlowAndMesh, posDeformationField)
      return self.transformer.sampleImage(sec_src_imgs, transforemdImageMeshToOtherImageSpace)

    
    def _getImageSpaceSimilarityLoss(self, imgs0, imgs1):
      imgSpaceSimLoss = self.similarityLoss(imgs0, imgs1)
      return imgSpaceSimLoss# / imgs0.shape[0]
    


    def getLoss(self, pos_flow, neg_flow, images, meshes, atlasImages, atlasMeshes):
      sim_loss, reg_loss, pair_sim_loss, atlas_pair_sim_loss = self.getLosses(pos_flow, neg_flow, images, meshes, atlasImages, atlasMeshes)
      return sim_loss + reg_loss + pair_sim_loss + atlas_pair_sim_loss
    
    def getLosses(self, pos_flow, neg_flow, images, meshes, atlasImages, atlasMeshes):
      
      # posDeformationField = self.transformer.getDeformationField(pos_flow)
      # deformedAtlasMeshes = self.transformer(atlasMeshes, posDeformationField)
      # deformedAtlas = self.transformer.sampleImage(atlasImages, deformedAtlasMeshes)
      
      
      posDeformationFieldAtlas = atlasMeshes + pos_flow
      posDeformationField = self.transformer.getDeformationField(pos_flow)
      negDeformationFieldImages = meshes + neg_flow#self.transformer.getDeformationField(neg_flow)
      
      warpedAtlas = self.transformer(atlasImages, posDeformationFieldAtlas)
      
      sampledImages = self.transformer.sampleImage(images, meshes)
      
      sim_loss = self._getImageSpaceSimilarityLoss(warpedAtlas, sampledImages) * self.sim_factor
      
      reg_loss = self.regularizationLoss(pos_flow) * self.reg_factor
      
      pair_sim_loss = 0.0
      atlas_pair_sim_loss = 0.0
      
      if self.imagePairSimilarityFactor != 0.0:
        deformedImages = self._getDefomredImages(posDeformationField, neg_flow, images, meshes)
        pair_sim_loss = self._getImageSpaceSimilarityLoss(deformedImages, images) * self.imagePairSimilarityFactor
        
      if self.atlasPairSimilarityFactor != 0.0: ##TODO: vergleicht nicht alle bild kombinationen, ausreichend oder umprogrammiren? 
        warpedImages = self.transformer(images, negDeformationFieldImages)
        batch_size = images.shape[0]
        atlas_pair_sim_loss = self._getImageSpaceSimilarityLoss(warpedImages[:int(batch_size/2)], warpedImages[int(batch_size/2):]) * self.atlasPairSimilarityFactor
        
      return sim_loss, reg_loss, pair_sim_loss, atlas_pair_sim_loss
        
        
        
        
      
      
      
      
      