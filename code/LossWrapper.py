"""
Created on Feb 9, 2024

@author: fechter
"""


class LossWrapper(object):
    def __init__(self):
        self.sim_loss = 0.0
        self.labelSimilarityLoss = 0.0
        self.labelSimilarityLossAtlasSpace = 0.0
        self.reg_loss = 0.0
        self.pair_sim_loss = 0.0
        self.atlas_pair_sim_loss = 0.0
        self.imgSpaceLabelLoss = 0.0
        self.atlasSpaceLabelLoss = 0.0
        self.defFieldInverseConsistencyLoss = 0.0
