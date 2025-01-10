"""
Created on Feb 9, 2024

@author: fechter
"""


class LossWrapper(object):
    def __init__(self):
        self.lossDict = {}
        self.lossFactors = {}

    def setLoss(self, lossName, value):
        self.lossDict[lossName] = value
        if lossName not in self.lossFactors.keys():
            self.lossFactors[lossName] = 0.0

    def getLossFactor(self, lossName):
        if lossName in self.lossFactors.keys():
            return self.lossFactors[lossName]
        else:
            return 0.0

    def setLossFactor(self, lossName, value):
        self.lossFactors[lossName] = value

    def getUnweightedLoss(self, lossName):
        if lossName in self.lossDict.keys():
            return self.lossDict[lossName]
        else:
            return 0.0

    def getWeightedLoss(self, lossName):
        if lossName in self.lossDict.keys():
            return self.lossDict[lossName] * self.lossFactors[lossName]
        else:
            return 0.0

    def getLossNames(self):
        return list(self.lossDict.keys())
