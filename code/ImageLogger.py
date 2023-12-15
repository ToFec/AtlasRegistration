"""
Created on Jun 30, 2023

@author: fechter
"""
from pytorch_lightning.loggers.logger import Logger, rank_zero_experiment
from pytorch_lightning.utilities import rank_zero_only

import os
import atlas_utils


class ImageLogger(Logger):
    def __init__(self, save_dir, name, imageOrigin, imageSpacing, imageDirections, version=1):
        super().__init__()
        self._save_dir = save_dir
        self._name = name or ""
        self._imageOrigin = imageOrigin
        self._imageSpacing = imageSpacing
        self._imageDirection = imageDirections
        self._version = version

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

    def saveImage(self, imageData, imageName, epoch):
        imageName = os.path.join(
            self._save_dir, self._name, f"version_{self._version}", imageName + str(epoch) + ".nrrd"
        )
        atlas_utils.saveImageTensor(imageData, imageName, self._imageOrigin, self._imageSpacing, self._imageDirection)

    @rank_zero_only
    def finalize(self, status):
        # Optional. Any code that needs to be run after training
        # finishes goes here
        pass
