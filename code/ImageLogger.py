'''
Created on Jun 30, 2023

@author: fechter
'''
from pytorch_lightning.loggers.logger import Logger, rank_zero_experiment
from pytorch_lightning.utilities import rank_zero_only


class ImageLogger(Logger):
  
    def __init__(self, save_dir, name):
        super().__init__()
        self._save_dir = save_dir
        self._name = name or ""  
  
    @property
    def name(self):
        return "ImageLogger"

    @property
    def version(self):
        return "0.1"

    @rank_zero_only
    def log_hyperparams(self, params):
        # params is an argparse.Namespace
        # your code to record hyperparameters goes here
        pass

    @rank_zero_only
    def log_metrics(self, metrics, step):
        # metrics is a dictionary of metric names and values
        # your code to record metrics goes here
        pass

    @rank_zero_only
    def save(self):
        # Optional. Any code necessary to save logger data goes here
        pass
      
    def saveImage(self):
        meshOrigin = data.train_set[0]['meshOrigin']
        meshSpacing = config.getParam("registrationGridSpacing")
        meshDir = [1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0]
        sitkImage = sitk.GetImageFromArray(tmpDeformed.squeeze(0).squeeze(0).permute([2,1,0]))
        sitkImage.SetOrigin(meshOrigin.tolist())
        sitkImage.SetDirection(meshDir)
        sitkImage.SetSpacing(meshSpacing)
        sitk.WriteImage(sitkImage, "./resources/gridTest.nrrd")

    @rank_zero_only
    def finalize(self, status):
        # Optional. Any code that needs to be run after training
        # finishes goes here
        pass
