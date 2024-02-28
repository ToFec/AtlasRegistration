"""
Created on Apr 28, 2023

@author: fechter
"""

import pytorch_lightning as pl
import csv
import os
import torchio as tio
from torch.utils.data import random_split, DataLoader
from typing import Optional
import numpy as np
import SimpleITK as sitk
import torch
import atlas_utils as atlasUtils

from config import Config
from imageTransformation import Transformation


class AtlasDataModule(pl.LightningDataModule):
    dataModuleName = "AtlasDataModule"

    def __init__(self, config: Config):
        super().__init__()
        self.datasetTrainingFile = config.getParam("trainingDataFile")
        self.datasetTestFile = config.getParam("testDataFile")
        self.train_subjects = []
        self.test_subjects = []
        self.batchSize = config.getParam("batchSize")
        self.train_val_ratio = config.getParam("trainValRatio")
        self.num_workers = config.getParam("numberOfWorkersDataLoader")
        self.persistentWorkers = config.getParam("numberOfWorkersDataLoader")
        self.pinMemory = config.getParam("pinMemory")
        self.shuffle = False
        self.imgFileNameColIdx = config.getParam("imageColIdxInTrainFile")
        self.labelFileNameColIdx = config.getParam("labelColIdxInTrainFile")
        self.randomSplit = config.getParam("doRandomTrainValSetSplit")
        self.doAugmentation = config.getParam("doDataAugmentation")
        self.doNormalisation = config.getParam("doNormalisation")
        self.initializeAtlasWithAverageImg = config.getParam("initializeAtlasWithAverageImg")
        self.atlasDataToLoad = config.getParam("atlasImage")
        self.atlasLabelToLoad = config.getParam("atlasLabel")
        self.loadImagesAsDataType = config.getParam("loadImagesAsDataType")

        if config.getParam("labelLoss") == "NCC":
            self.createDistanceMapFromlabel = True
        else:
            self.createDistanceMapFromlabel = False

        self.registrationGridsize = config.getParam("registrationGridsize")
        self.registrationGridSpacing = config.getParam("registrationGridSpacing")

        if config.getParam("csvDelimiter"):
            self.delimiter = config.getParam("csvDelimiter")
        else:
            self.delimiter = ";"
        self.atlasImage = None
        self.atlasLabel = None
        self.atlasOrigin = None
        self.atlasMesh = None
        self.train_sampler = None
        self.val_set = None
        self.train_set = None
        self.validation_sampler = None
        self.test_set = None

    # pytorch lightning hook
    def prepare_data(self):
        pass

    def getInitalAtlas(self):
        return self.atlasImage, self.atlasMesh, self.atlasOrigin, self.atlasLabel

    def _prepare_data(self):
        if len(self.train_subjects) == 0:
            self.iterateFile(self.datasetTrainingFile, self.train_subjects)
        if len(self.test_subjects) == 0:
            self.iterateFile(self.datasetTestFile, self.test_subjects)

    def iterateFile(self, inputFile, container):
        with open(inputFile) as csvDataFile:
            csvReader = csv.reader(csvDataFile, delimiter=self.delimiter)
            for row in csvReader:
                imageFileName = row[self.imgFileNameColIdx]

                if os.path.exists(imageFileName):
                    labelFileName = None
                    if self.labelFileNameColIdx > -1 and len(row) > self.labelFileNameColIdx:
                        labelFileName = row[self.labelFileNameColIdx]
                    subject = self.getSubject(imageFileName, labelFileName)
                    container.append(subject)

    def getSubject(self, imageFileName, labelFileName):
        subject = None
        if os.path.exists(imageFileName):
            sitkImage = sitk.ReadImage(imageFileName, sitk.GetPixelIDValueFromString(self.loadImagesAsDataType))
            scalarImage = tio.ScalarImage.from_sitk(sitkImage)
            subjectDict = {"image": scalarImage}
            subjectDict["imagePath"] = imageFileName

            labelImage = None
            if labelFileName and os.path.exists(labelFileName):
                sitkLabel = sitk.ReadImage(labelFileName, sitk.sitkInt64)
                labelImage = tio.LabelMap.from_sitk(sitkLabel)

            meshName = os.path.splitext(imageFileName)[0] + "Mesh.pt"
            meshParamsMatch = False
            if os.path.exists(meshName):
                sampleMesh, sampleMeshOrigin, sampleMeshSpacing = torch.load(meshName)
                if (
                    list(sampleMesh.shape[1:]) == self.registrationGridsize
                    and sampleMeshSpacing == self.registrationGridSpacing
                ):
                    meshParamsMatch = True
            if not meshParamsMatch:
                sampleMesh, sampleMeshOrigin = self.getSampleMesh(scalarImage, labelImage)
                torch.save([sampleMesh, sampleMeshOrigin, self.registrationGridSpacing], meshName)

            if labelImage is None:
                labelData = self._craeteLabelImage(scalarImage, sampleMesh)
                labelImage = tio.LabelMap(tensor=labelData, affine=scalarImage["affine"])

            if self.createDistanceMapFromlabel:
                sitkLabel = sitk.GetImageFromArray(labelImage.data.squeeze().swapaxes(0, -1))
                sitkLabel.CopyInformation(sitkImage)
                distnaceMapTensor = torch.from_numpy(atlasUtils.createSignedDistanceMap(sitkLabel).swapaxes(1, -1))
                labelImage.data = distnaceMapTensor.to(torch.float32)
            else:
                newData = torch.nn.functional.one_hot(labelImage.data.squeeze()).movedim(-1, 0)
                labelImage.data = newData.to(torch.float32)

            subjectDict["label"] = labelImage

            subjectDict["samplingMesh"] = sampleMesh
            subjectDict["meshOrigin"] = sampleMeshOrigin
            subject = tio.Subject(subjectDict)
        return subject

    def getStandardSpaceToSubjectTransformMatrix(self, mesh):
        normVec0 = torch.nn.functional.normalize((mesh[:, -1, 0, 0] - mesh[:, 0, 0, 0])[None, ...])
        normVec1 = torch.nn.functional.normalize((mesh[:, 0, -1, 0] - mesh[:, 0, 0, 0])[None, ...])
        normVec2 = torch.nn.functional.normalize((mesh[:, 0, 0, -1] - mesh[:, 0, 0, 0])[None, ...])
        orientationMatrix = torch.inverse(torch.cat((normVec0, normVec1, normVec2)))
        scaling = torch.zeros_like(orientationMatrix)
        scaling[0, 0] = torch.linalg.vector_norm((mesh[:, 0, 0, 0] - mesh[:, -1, 0, 0])[None, ...]) / 2.0
        scaling[1, 1] = torch.linalg.vector_norm((mesh[:, 0, 0, 0] - mesh[:, 0, -1, 0])[None, ...]) / 2.0
        scaling[2, 2] = torch.linalg.vector_norm((mesh[:, 0, 0, 0] - mesh[:, 0, 0, -1])[None, ...]) / 2.0
        combinedMatrix = torch.matmul(orientationMatrix, scaling)
        return combinedMatrix

    ##this method gives only reliable results when mesh spacing is close to voxelsize
    def _craeteLabelImageData(self, imageData, mesh):
        mesh = (mesh + 1.0) / 2.0
        tmp = (mesh >= 0.0).all(axis=0)
        mesh = mesh[:, tmp]
        tmp = (mesh <= 1.0).all(axis=0)
        mesh = mesh[:, tmp]
        for dim in range(mesh.shape[0]):
            mesh[dim] = mesh[dim] * (imageData.shape[-3 + dim] - 1.0)

        meshFloor = torch.floor(mesh).type(torch.int32)
        meshCeil = torch.ceil(mesh).type(torch.int32)
        labelData = torch.zeros_like(imageData)
        labelData[:, meshFloor[0, :], meshFloor[1, :], meshFloor[2, :]] = 1.0
        labelData[:, meshCeil[0, :], meshCeil[1, :], meshCeil[2, :]] = 1.0
        labelData = labelData.unsqueeze(0)
        labelData = torch.nn.functional.conv3d(
            labelData, weight=torch.ones([1, 1, 3, 3, 3], dtype=labelData.dtype), stride=1, padding=1
        )
        labelData = labelData.squeeze(0)
        labelData[labelData < 14] = 0
        labelData[labelData >= 14] = 1
        labelData = labelData.type(torch.long)

        return labelData

    def _craeteLabelImage(self, scalarImage, mesh):
        imageData = scalarImage[tio.DATA]
        return self._craeteLabelImageData(imageData, mesh)

    def _getAugmentationTransform(self):
        augmentations = []
        if self.doAugmentation:
            augment = tio.Compose(
                [
                    tio.RandomAffine(scales=(0.9, 1.1), translation=(4, 4, 4), degrees=15, p=0.3),
                    tio.RandomFlip(axes=("LR"), p=0.3),
                ]
            )
            augmentations.append(augment)

        if self.doNormalisation:
            transform = tio.ZNormalization()  # (masking_method="label")
            augmentations.append(transform)
        return tio.Compose(augmentations)

    def _dataSplit(self):
        num_subjects = len(self.train_subjects)
        num_train_subjects = int(round(num_subjects * self.train_val_ratio))
        num_val_subjects = num_subjects - num_train_subjects

        if self.randomSplit:
            splits = num_train_subjects, num_val_subjects
            train_subjects, val_subjects = random_split(self.train_subjects, splits)
        else:
            train_subjects = self.train_subjects[0:num_train_subjects]
            val_subjects = self.train_subjects[num_train_subjects:num_subjects]
        return train_subjects, val_subjects

    def _setAtlasImage(self):
        if len(self.train_subjects) > 0:
            atlasLabel = None
            if self.initializeAtlasWithAverageImg:
                imgShape = list(self.train_subjects[0]["samplingMesh"].shape)
                imgShape[0] = 1
                imgShape = [1] + imgShape
                transformer = Transformation(imgShape)
                avgImg = torch.zeros(imgShape)
                for train_subject in self.train_subjects:
                    tmpImg = train_subject["image"][tio.DATA].unsqueeze(0).type(torch.FloatTensor)
                    tmpMesh = train_subject["samplingMesh"].unsqueeze(0)
                    sampledData = transformer.sampleImage(tmpImg, tmpMesh)
                    avgImg = avgImg + sampledData
                avgImg = avgImg / len(self.train_subjects)
                self.atlasImage = avgImg[0]
                self.atlasOrigin = torch.zeros(3)  # [0.0, 0.0, 0.0]
                self.atlasMesh = transformer.identityTransform
            elif self.atlasDataToLoad is not None and os.path.exists(self.atlasDataToLoad):
                subject = self.getSubject(self.atlasDataToLoad, self.atlasLabelToLoad)

                imgShape = list(subject["samplingMesh"].shape)
                imgShape[0] = 1
                imgShape = [1] + imgShape
                transformer = Transformation(imgShape)

                tmpImg = subject["image"][tio.DATA].unsqueeze(0).type(torch.FloatTensor)
                sampledData = transformer.sampleImage(tmpImg, subject["samplingMesh"].unsqueeze(0))

                if self.atlasLabelToLoad is not None:
                    labels = subject["label"][tio.DATA]
                    atlasLabel = transformer.sampleImage(
                        labels.unsqueeze(0), subject["samplingMesh"].unsqueeze(0), interpolationType="nearest"
                    )[0]

                self.atlasImage = sampledData[0]
                self.atlasOrigin = subject["meshOrigin"]
                self.atlasMesh = transformer.identityTransform
            else:
                subject = self.train_subjects[0]
                imgData = subject["image"][tio.DATA].detach().clone().type(torch.FloatTensor)
                imgShape = list(subject["samplingMesh"].shape)
                imgShape[0] = 1
                imgShape = [1] + imgShape
                transformer = Transformation(imgShape)

                labels = subject["label"][tio.DATA].detach().clone()

                self.atlasMesh = subject["samplingMesh"].detach().clone()
                atlasLabel = transformer.sampleImage(
                    labels.unsqueeze(0), self.atlasMesh.unsqueeze(0), interpolationType="nearest"
                )[0]

                sampledData = transformer.sampleImage(imgData.unsqueeze(0), self.atlasMesh.unsqueeze(0))
                self.atlasImage = sampledData[0]
                self.atlasOrigin = subject["meshOrigin"].detach().clone()

            if atlasLabel is None:
                atlasLabel = self._craeteLabelImageData(self.atlasImage, self.atlasMesh)
            if self.doNormalisation:
                subject = tio.Subject(
                    {"label": tio.LabelMap(tensor=atlasLabel), "image": tio.ScalarImage(tensor=self.atlasImage)}
                )
                transform = tio.ZNormalization()  # (masking_method="label")
                normalizedAtlasSubject = transform(subject)
                self.atlasImage = normalizedAtlasSubject["image"][tio.DATA]

            self.atlasLabel = atlasLabel.unsqueeze(0)
            self.atlasImage = self.atlasImage.unsqueeze(0)
            self.atlasMesh = self.atlasMesh.unsqueeze(0)
        else:
            self.atlasImage = None
            self.atlasLabel = None
            self.atlasMesh = None
            self.atlasOrigin = None

    def getSampleMesh(self, scalarImage, labelImage):
        sitkScalarImage = scalarImage.as_sitk()

        if labelImage:
            sitkLabelImage = labelImage.as_sitk()
            label_statistic = sitk.LabelIntensityStatisticsImageFilter()
            label_statistic.Execute(sitkLabelImage > 0, sitkLabelImage)
            centerPoint = label_statistic.GetCentroid(1)
        else:
            centerPoint = sitkScalarImage.TransformContinuousIndexToPhysicalPoint(
                np.asarray(sitkScalarImage.GetSize()) / 2.0
            )

        centerPoint = centerPoint - (np.multiply(self.registrationGridsize, self.registrationGridSpacing) / 2.0)

        imgSize = torch.asarray(sitkScalarImage.GetSize())
        dirMatrix = torch.inverse(torch.Tensor(sitkScalarImage.GetDirection()).reshape([3, 3]))
        orig = torch.Tensor(sitkScalarImage.GetOrigin())
        spacing = torch.Tensor(sitkScalarImage.GetSpacing())

        gridVecWorldC = [
            (torch.arange(s) * self.registrationGridSpacing[idx]) + centerPoint[idx]
            for idx, s in enumerate(self.registrationGridsize)
        ]
        gridWorldC = torch.meshgrid(*gridVecWorldC)
        gridShape = gridWorldC[0].shape + (len(gridWorldC),)
        flatGridWorldC = [s.flatten() for s in gridWorldC]
        flatGridWorldC = torch.stack(flatGridWorldC, 1)
        gridOrigin = flatGridWorldC[0, :]

        flatImgC = torch.matmul(dirMatrix, ((flatGridWorldC - orig) / spacing)[:, :, None])
        flatImgC = flatImgC.squeeze()
        flatImgC = (flatImgC / (imgSize - 1.0)) * 2.0 - 1.0

        gridImgC = flatImgC.reshape(gridShape)  # .flip(-1)
        gridImgC = torch.moveaxis(gridImgC, -1, 0)

        # gridImgC = torch.unsqueeze(gridImgC, 0)
        gridImgC = gridImgC.type(torch.FloatTensor)

        return gridImgC, gridOrigin

    # pytorch lightning hook
    def setup(self, stage: Optional[str] = None):
        self._prepare_data()
        self._setAtlasImage()
        transform = self._getAugmentationTransform()
        if stage == "fit" or stage is None:
            train_subjects, val_subjects = self._dataSplit()
            self.train_sampler = None
            self.validation_sampler = None
            self.shuffle = True

            self.train_set = tio.SubjectsDataset(train_subjects, transform=transform)
            if len(val_subjects) > 0:
                self.val_set = tio.SubjectsDataset(val_subjects, transform=transform)
            else:
                self.val_set = []

            print("size of training set: ", len(self.train_set))
            print("size of validation set: ", len(self.val_set))

        if stage == "test" or stage is None:
            self.test_set = tio.SubjectsDataset(self.test_subjects, transform=transform)

    # pytorch lightning hook
    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            self.batchSize,
            num_workers=self.num_workers,
            shuffle=self.shuffle,
            sampler=self.train_sampler,
            persistent_workers=self.persistentWorkers,
            pin_memory=self.pinMemory,
        )

    # pytorch lightning hook
    def val_dataloader(self):
        return DataLoader(
            self.val_set,
            self.batchSize,
            num_workers=self.num_workers,
            sampler=self.validation_sampler,
            persistent_workers=self.persistentWorkers,
            pin_memory=self.pinMemory,
        )

    # pytorch lightning hook
    def test_dataloader(self):
        return DataLoader(
            self.test_set,
            self.batchSize,
            num_workers=self.num_workers,
            persistent_workers=self.persistentWorkers,
            pin_memory=self.pinMemory,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.test_set,
            self.batchSize,
            num_workers=self.num_workers,
            persistent_workers=self.persistentWorkers,
            pin_memory=self.pinMemory,
        )
