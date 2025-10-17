"""
Created on Apr 28, 2023

@author: fechter
"""
from ImageLogger import ImageLogger


import torchio as tio
import torch

torch.cuda.empty_cache()

import pytorch_lightning as pl
import pl_bolts
from imageTransformation import Transformation


class AtlasModule(pl.LightningModule):
    def __init__(
        self,
        net,
        atlasImage,
        atlasLabel,
        atlasMesh,
        atlasOrigin,
        loss,
        networkLearning_rate,
        atlasLearning_rate,
        networkOptimizer_class,
        atlasOptimizer_class,
        useLrScheduler,
        logTemporaryDeformationFields=False,
    ):
        super().__init__()
        self.automatic_optimization = False
        self.nlr = networkLearning_rate
        self.alr = atlasLearning_rate
        self.net = net
        self.criterion = loss
        self.networkOptimizer_class = networkOptimizer_class
        self.atlasOptimizer_class = atlasOptimizer_class
        self.lrScheduler = useLrScheduler
        self.transformer = Transformation()
        self.save_hyperparameters(logger=False)
        if self.alr > 0.0:
            self.atlasImage = torch.nn.Parameter(atlasImage)
        else:
            self.register_buffer("atlasImage", atlasImage, True)
        self.register_buffer("atlasLabel", atlasLabel, True)
        self.register_buffer("atlasMesh", atlasMesh, True)
        self.register_buffer("atlasOrigin", atlasOrigin, True)
        self.logTemporaryDeformationFields = logTemporaryDeformationFields

    def setAtlasInformation(self, atlasImage, atlasLabel, atlasMesh, atlasOrigin, atlasLearning_rate):
        self.alr = atlasLearning_rate
        if self.alr > 0.0:
            self.atlasImage = torch.nn.Parameter(atlasImage)
        else:
            self.register_buffer("atlasImage", atlasImage, True)
        self.register_buffer("atlasLabel", atlasLabel, True)
        self.register_buffer("atlasMesh", atlasMesh, True)
        self.register_buffer("atlasOrigin", atlasOrigin, True)

    def getInputAtlasMesh(self, batch_size):
        return self.atlasMesh.expand(batch_size, -1, -1, -1, -1)

    def getInputAtlasLabel(self, batch_size):
        return self.atlasLabel.expand(batch_size, -1, -1, -1, -1)

    def getInputAtlasImage(self, batch_size, detached=False):
        if detached:
            return self.atlasImage.expand(batch_size, -1, -1, -1, -1).detach()
        else:
            return self.atlasImage.expand(batch_size, -1, -1, -1, -1)

    def configure_optimizers(self):
        networkOptimizer = self.networkOptimizer_class(self.net.parameters(), lr=self.nlr)
        # atlasOptimizer = self.atlasOptimizer_class(self.parameters(), lr=(self.alr or self.learning_rate))
        atlasOptimizer = self.atlasOptimizer_class([self.atlasImage], lr=self.alr)

        if self.lrScheduler:
            lr_scheduler_net = {
                "scheduler": pl_bolts.optimizers.lr_scheduler.LinearWarmupCosineAnnealingLR(
                    networkOptimizer, warmup_epochs=10, max_epochs=100, warmup_start_lr=1e-6, eta_min=1e-6
                ),
                "name": "WarumUpCosineLR",
            }
            lr_scheduler_atlas = {
                "scheduler": pl_bolts.optimizers.lr_scheduler.LinearWarmupCosineAnnealingLR(
                    atlasOptimizer, warmup_epochs=10, max_epochs=100, warmup_start_lr=1e-6, eta_min=1e-6
                ),
                "name": "WarumUpCosineLR",
            }
        else:
            lr_scheduler_net = {
                "scheduler": torch.optim.lr_scheduler.ConstantLR(networkOptimizer, factor=1.0, total_iters=0),
                "name": "ConstantLR",
            }
            lr_scheduler_atlas = {
                "scheduler": torch.optim.lr_scheduler.ConstantLR(atlasOptimizer, factor=1.0, total_iters=0),
                "name": "ConstantLR",
            }

        return (
            {"optimizer": networkOptimizer, "lr_scheduler": lr_scheduler_net},
            {"optimizer": atlasOptimizer, "lr_scheduler": lr_scheduler_atlas},
        )

    def _createNetworkInput(self, images, meshes):
        networkImageToRegInput = self.transformer.sampleImage(images, meshes)
        networkAtlasInput = self.transformer.sampleImage(
            self.getInputAtlasImage(networkImageToRegInput.shape[0], detached=True),
            self.getInputAtlasMesh(networkImageToRegInput.shape[0]),
        )
        return networkImageToRegInput, networkAtlasInput

    def prepare_batch(self, batch):
        # images = batch["image"][tio.DATA]
        images = [image[tio.DATA] for image in batch["image"]]
        meshes = batch["samplingMesh"]
        # labels = batch["label"][tio.DATA]
        labels = [image[tio.DATA] for image in batch["label"]]
        return images, meshes, labels

    def infer_batch(self, images, atlasImages):
        atlasAndImages = torch.cat((atlasImages, images), 1)
        pos_flow, neg_flow = self.net(atlasAndImages)
        return pos_flow, neg_flow

    def gatherInfoOfTrainingValidationStep(self, pos_flow, neg_flow, images, meshes, labels):
        self.criterion.calculateLoss(
            pos_flow,
            neg_flow,
            images,
            meshes,
            self.getInputAtlasImage(meshes.shape[0]),
            self.getInputAtlasMesh(meshes.shape[0]),
            self.getInputAtlasLabel(meshes.shape[0]),
            labels,
        )
        lossWrapper = self.criterion.getLosses()
        totalLoss = torch.zeros_like(lossWrapper.getUnweightedLoss("reg_loss"))
        dictToReturn = {}
        for loss in lossWrapper.getLossNames():
            currLoss = lossWrapper.getWeightedLoss(loss)
            totalLoss = totalLoss + currLoss
            dictToReturn[loss] = currLoss

        dictToReturn["loss"] = totalLoss
        return dictToReturn

    def training_step(self, batch, batch_idx):
        optNetwork, _ = self.optimizers(use_pl_optimizer=True)
        optNetwork.zero_grad()
        images, meshes, labels = self.prepare_batch(batch)
        networkImageToRegInput, networkAtlasInput = self._createNetworkInput(images, meshes)

        pos_flow, neg_flow = self.infer_batch(networkImageToRegInput, networkAtlasInput)

        stepInfo = self.gatherInfoOfTrainingValidationStep(pos_flow, neg_flow, images, meshes, labels)
        loss = stepInfo["loss"]

        self.manual_backward(loss)

        #print(optNetwork)
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), 0.5)

        optNetwork.step()
        return stepInfo

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        images, meshes, _ = self.prepare_batch(batch)
        networkImageToRegInput, networkAtlasInput = self._createNetworkInput(images, meshes)
        pos_flow, neg_flow = self.infer_batch(networkImageToRegInput, networkAtlasInput)
        return pos_flow, neg_flow

    def validation_step(self, batch, batch_idx):
        images, meshes, labels = self.prepare_batch(batch)
        networkImageToRegInput, networkAtlasInput = self._createNetworkInput(images, meshes)
        pos_flow, neg_flow = self.infer_batch(networkImageToRegInput, networkAtlasInput)
        self.criterion.calculateLoss(
            pos_flow,
            neg_flow,
            images,
            meshes,
            self.getInputAtlasImage(meshes.shape[0]),
            self.getInputAtlasMesh(meshes.shape[0]),
            self.getInputAtlasLabel(meshes.shape[0]),
            labels,
        )
        lossWrapper = self.criterion.getLosses()

        totalLoss = torch.zeros_like(lossWrapper.getUnweightedLoss("reg_loss"))
        totalLossUW = torch.zeros_like(totalLoss)
        for loss in lossWrapper.getLossNames():
            currLoss = lossWrapper.getWeightedLoss(loss)
            currLossUW = lossWrapper.getUnweightedLoss(loss)
            totalLoss = totalLoss + currLoss
            totalLossUW = totalLossUW + currLossUW

            self.log(f"val_{loss}", currLoss)
            self.log(f"val_{loss}_uw", currLossUW)

        self.log("val_loss_uw", totalLossUW)
        self.log("val_loss", totalLoss)

        if self.logTemporaryDeformationFields:
            for logger in self.loggers:
                if isinstance(logger, ImageLogger):
                    defFieldToSave = torch.Tensor.cpu(neg_flow[0, None, ...].detach())
                    logger.saveImage(defFieldToSave, "DeformationField0", self.current_epoch)
        return {
            "loss": totalLoss,
            "val_atlas_space_label_loss_uw": lossWrapper.getWeightedLoss("atlasSpaceLabelLoss"),
            "val_label_sim_loss_atlas_space_UW": lossWrapper.getWeightedLoss("labelSimilarityLossAtlasSpace"),
        }

    def test_step(self, batch, batch_idx):
        images, meshes, labels = self.prepare_batch(batch)
        networkImageToRegInput, networkAtlasInput = self._createNetworkInput(images, meshes)
        pos_flow, neg_flow = self.infer_batch(networkImageToRegInput, networkAtlasInput)
        stepInfo = self.gatherInfoOfTrainingValidationStep(pos_flow, neg_flow, images, meshes, labels)

        for lossName in stepInfo.keys():
            self.log(f"test_{lossName}", stepInfo[lossName])
        return pos_flow, neg_flow, stepInfo

    def epochEndLogging(self, outputs, trainValString):
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()

        if self.current_epoch == 1:
            networkAtlasInput = self.transformer.sampleImage(self.getInputAtlasImage(1), self.getInputAtlasMesh(1))
            exampleInputArray = torch.cat((networkAtlasInput, networkAtlasInput), 1)
            self.logger.experiment.add_graph(self.net, exampleInputArray)

        self.logger.experiment.add_scalar("Loss/" + trainValString, avg_loss, self.current_epoch)

        if self.alr > 0.0:
            networkAtlasInput = self.transformer.sampleImage(self.getInputAtlasImage(1), self.getInputAtlasMesh(1))

            networkAtlasInput = torch.Tensor.cpu(networkAtlasInput.detach())
            self.logger.experiment.add_image(
                "AtlasCenterSlice",
                networkAtlasInput[0, 0, int(networkAtlasInput.shape[2] / 2), ...],
                self.current_epoch,
                dataformats="HW",
            )
            for logger in self.loggers:
                if isinstance(logger, ImageLogger):
                    logger.saveImage(networkAtlasInput, "AtlasImage", self.current_epoch)

    def training_epoch_end(self, outputs):
        ###
        ##in case of heterogeneous/speckled atlas, a smooth loss can be used here
        ####
        _, optAtlas = self.optimizers(use_pl_optimizer=True)
        optAtlas.step()
        optAtlas.zero_grad()
        self.epochEndLogging(outputs, "Train")

    def validation_epoch_end(self, outputs):
        self.epochEndLogging(outputs, "Validation")

        avg_val_label_sim_loss_atlas_space_UW = torch.stack(
            [x["val_label_sim_loss_atlas_space_UW"] for x in outputs]
        ).mean()
        avg_val_atlas_space_label_loss_uw = torch.stack([x["val_atlas_space_label_loss_uw"] for x in outputs]).mean()
        self.log("ptl/label_sim_loss_atlas_space", avg_val_label_sim_loss_atlas_space_UW)
        self.log("ptl/atlas_space_label_loss", avg_val_atlas_space_label_loss_uw)
