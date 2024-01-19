"""
Created on Jul 18, 2023

@author: fechter
"""
import atlas_models


def getNetwork(config):
    networkName = config.getParam("networkClassName")
    doBatchNormalisation = config.getParam("batchNormalisation")
    if hasattr(atlas_models, networkName):
        networkClass = getattr(atlas_models, networkName)
    else:
        print("NeworkFactory: cannot find network with name ", networkName)
        print("NeworkFactory: using Dummy instead")
        networkClass = getattr(atlas_models, "Dummy")

    params = {}
    if doBatchNormalisation is not None:
        params["bn"] = doBatchNormalisation
    networkInstance = networkClass(params)
    return networkInstance
