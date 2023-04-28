'''
Created on Apr 28, 2023

@author: fechter
'''
from losses import LossFactory


class LossCalculator():

    def __init__(self, config):
        loss_name = config.getParam("similarityLoss")
        self.similarityLoss = LossFactory.lossMap[loss_name]
        
        reg_factor = config.getParam("regularizationFactor")
        sim_factor = config.getParam("similarityFactor")
        pair_sim_factor = config.getParam("imagePairSimFactor")
        smooth_factor = config.getParam("smoothingFactor")
        

    def getLoss(self, pos_flow, neg_flow, images, atlasImages):
      
      sim_loss = self.similarityLoss(svf_warped_atlas_imgs, src_imgs, loss_name)