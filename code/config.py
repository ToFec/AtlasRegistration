"""
Created on Apr 11, 2022

@author: fechter
"""

import os, json, logging
from pathlib import Path


class Config(object):
    def __init__(self, configFile=None):
        if configFile is None:
            configFile = Path(__file__).parents[1].joinpath("resources", "Config.json").as_posix()
        if not os.path.isfile(configFile):
            raise Exception("Configfile does not exist!")

        with open(configFile) as f:
            self.params = json.load(f)
            self.comments = {}
            paramKeys = list(self.params.keys())
            for k in paramKeys:  # remove comment entries
                if k.startswith("__"):
                    self.comments[k] = self.params[k]
                    self.params.pop(k)
                if k.startswith("f_"):
                    func = eval(self.params[k])
                    self.params.pop(k)
                    self.params[k[2:]] = func

    def getParams(self):
        return self.params

    def setParams(self, params: dict):
        self.params = params

    def setParam(self, key, value):
        self.params[key] = value

    def getParam(self, key):
        if key in self.params:
            return self.params.get(key)
        else:
            logging.warn("Configuration does not contain a value for " + key)
            return None
