'''
Created on Jul 10, 2023

@author: fechter
'''
from pytorch_lightning.callbacks import BasePredictionWriter
import os

from imageTransformation import Bilinear
import atlas_utils

class DeformationFieldAndDeformedImageWriter(BasePredictionWriter):
    def __init__(self, config, write_interval):
        super().__init__(write_interval)
        self.output_dir = config.getParam("outputPath")
        self.meshDir = [1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0]
        self.meshSpacing = config.getParam("registrationGridSpacing")
        self.transformer = Bilinear()


    def write_on_batch_end(self, trainer, pl_module, prediction, batch_indices, batch, batch_idx, dataloader_idx):
      
      images, meshes = pl_module.prepare_batch(batch)
      atlasImages = pl_module.getInputAtlasImage(images.shape[0])
      atlasMeshes = pl_module.getInputAtlasMesh(images.shape[0])
      pos_flow = prediction[0]
      neg_flow = prediction[1]
      posDeformationFieldAtlas = atlasMeshes + pos_flow
      warpedAtlas = self.transformer(atlasImages, posDeformationFieldAtlas)
      imageNames = batch['image']['path']
      meshOrigin = batch['meshOrigin']
      atlasOrigin = pl_module.atlasOrigin.tolist()
      atlas_utils.saveImageTensor(atlasImages[0,None,...], os.path.join(self.output_dir, "Atlas.nrrd"), atlasOrigin, self.meshSpacing, self.meshDir)
      
      for i in range(0,warpedAtlas.shape[0]):
        fileBaseName = os.path.splitext(os.path.basename(imageNames[i]))[0]
        atlas_utils.saveImageTensor(warpedAtlas[i,None,...], os.path.join(self.output_dir, fileBaseName + "AtlasDef.nrrd"), meshOrigin[i].tolist(), self.meshSpacing, self.meshDir)
        
        atlas_utils.saveDefField(os.path.join(self.output_dir, fileBaseName + "AtlasDefField.nrrd"), pos_flow[i,None,...], meshOrigin[i].tolist(), self.meshSpacing, self.meshDir)
      
      
      negDeformationFieldImages = meshes + neg_flow
      warpedImages = self.transformer(images, negDeformationFieldImages)
      for i in range(0,warpedImages.shape[0]):
        fileBaseName = os.path.splitext(os.path.basename(imageNames[i]))[0]
        atlas_utils.saveImageTensor(warpedImages[i,None,...], os.path.join(self.output_dir, fileBaseName + "Def.nrrd"), atlasOrigin, self.meshSpacing, self.meshDir)
        atlas_utils.saveDefField(os.path.join(self.output_dir, fileBaseName + "DefField.nrrd"), neg_flow[i,None,...],atlasOrigin, self.meshSpacing, self.meshDir)

    def write_on_epoch_end(self, trainer, pl_module, predictions, batch_indices):
      pass