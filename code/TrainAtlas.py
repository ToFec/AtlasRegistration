import datetime as dt
import os
import sys
from atlasDataModule import AtlasDataModule
from atlasModule import AtlasModule
import NetworkFactory
from PredictionEvaluationWriter import PredictionEvaluationWriter
import logging

sys.path.append(os.path.realpath(".."))
import argparse

import atlas_utils

from config import Config
from lossCalculator import LossCalculator
from imageTransformation import Transformation

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
import torch
from ImageLogger import ImageLogger
from DeformationFieldAndDeformedImageWriter import DeformationFieldAndDeformedImageWriter

from ray import tune
from ray.tune.integration.pytorch_lightning import TuneReportCallback
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search import ConcurrencyLimiter
from ray.tune.search.optuna import OptunaSearch
from ray.train.constants import DEFAULT_STORAGE_PATH as RayDefaultStoratePath


def getCheckPointString(config):
    seed = config.getParam("seed")

    max_epochs = config.getParam("epochs")
    batch_size = config.getParam("batchSize")
    lr = config.getParam("learningRate")
    atlas_lr = config.getParam("atlasLearningRate")
    loss_name = config.getParam("similarityLoss")
    reg_loss_name = config.getParam("regularizationLoss")
    labelLoss = config.getParam("labelLoss")
    labelSimilarityFactor = config.getParam("labelSimilarityFactor")
    labelSimilarityFactorAtlasSpace = config.getParam("labelSimilarityFactorAtlasSpace")
    inverseConsistencyLoss = config.getParam("defDieldInverseConsistencyLossFactor")

    reg_factor = config.getParam("regularizationFactor")
    sim_factor = config.getParam("similarityFactor")
    pair_sim_factor = config.getParam("imagePairSimFactor")
    imgSpaceLabelSimFactor = config.getParam("imageSpaceLabelSimFactor")
    atlasSpaceLabelSimFactor = config.getParam("atlasSpaceLabelSimFactor")
    atlas_Pair_Sim_Factor = config.getParam("atlasPairSimFactor")
    smooth_factor = config.getParam("smoothingFactor")
    gridSize = config.getParam("registrationGridsize")
    gridSpacing = config.getParam("registrationGridSpacing")

    volumePreservationLossFactor = config.getParam("volumePreservationLossFactor")

    gridSizeStr = "".join(map(str, gridSize))
    gridSpacingStr = "".join(map(str, gridSpacing))

    stringForStoringVariables = (
        "AR_"
        + str(loss_name)
        + "_"
        + str(labelLoss)
        + "_"
        + str(reg_loss_name)
        + "_"
        + gridSizeStr
        + "_"
        + gridSpacingStr
        + "_s_"
        + str(seed)
        + "_r_"
        + "{:.2f}".format(reg_factor)
    )
    if sim_factor > 0.0:
        stringForStoringVariables = stringForStoringVariables + "{:.2f}".format(sim_factor)
    if pair_sim_factor is not None and pair_sim_factor > 0.0:
        stringForStoringVariables = stringForStoringVariables + "_Isim_" + "{:.2f}".format(pair_sim_factor)

    if imgSpaceLabelSimFactor is not None and imgSpaceLabelSimFactor > 0.0:
        stringForStoringVariables = stringForStoringVariables + "_ILSim" + "{:.2f}".format(imgSpaceLabelSimFactor)

    if atlas_Pair_Sim_Factor is not None and atlas_Pair_Sim_Factor > 0.0:
        stringForStoringVariables = stringForStoringVariables + "_Asim" + "{:.2f}".format(atlas_Pair_Sim_Factor)

    if atlasSpaceLabelSimFactor is not None and atlasSpaceLabelSimFactor > 0.0:
        stringForStoringVariables = stringForStoringVariables + "_ALSim" + "{:.2f}".format(atlasSpaceLabelSimFactor)

    if smooth_factor is not None and smooth_factor > 0.0:
        stringForStoringVariables = stringForStoringVariables + "_smooth_" + "{:.2f}".format(smooth_factor)

    if labelSimilarityFactor is not None and labelSimilarityFactor > 0.0:
        stringForStoringVariables = stringForStoringVariables + "_lSim_" + "{:.2f}".format(labelSimilarityFactor)

    if labelSimilarityFactorAtlasSpace is not None and labelSimilarityFactorAtlasSpace > 0.0:
        stringForStoringVariables = (
            stringForStoringVariables + "_lSimA_" + "{:.2f}".format(labelSimilarityFactorAtlasSpace)
        )

    if volumePreservationLossFactor is not None and volumePreservationLossFactor > 0.0:
        stringForStoringVariables = stringForStoringVariables + "_vp_" + "{:.2f}".format(volumePreservationLossFactor)

    if inverseConsistencyLoss is not None and inverseConsistencyLoss > 0.0:
        stringForStoringVariables = stringForStoringVariables + "_inv_" + "{:.2f}".format(inverseConsistencyLoss)

    stringForStoringVariables = (
        stringForStoringVariables
        + "_e_"
        + str(max_epochs)
        + "_b_"
        + str(batch_size)
        + "_nlr_"
        + str(lr)
        + "_alr_"
        + str(atlas_lr)
    )

    return stringForStoringVariables


def getModelAndData(config, stageType):
    seed = config.getParam("seed")

    if seed is not None:
        atlas_utils.setSeeds(seed)

    matMulPrecision = config.getParam("matMulPrecision")
    if matMulPrecision:
        atlas_utils.setMatmulPrecision(matMulPrecision)

    stringForStoringVariables = getCheckPointString(config)

    f = open(os.path.join(config.getParam("checkPointPath"), stringForStoringVariables + ".txt"), "r")
    checkPointPath = f.read().splitlines()[0]

    model = AtlasModule.load_from_checkpoint(checkPointPath)

    network = model.net
    newShape = network.getShapeForModel(config.getParam("registrationGridsize"))
    config.setParam("registrationGridsize", newShape.tolist())

    data = AtlasDataModule(config)
    data.prepare_data()
    data.setup(stage=stageType)

    return model, data


def runTests(config):
    model, data = getModelAndData(config, "test")

    logger = TensorBoardLogger("tb_logs", name="test_" + getCheckPointString(config))
    pred_writer = DeformationFieldAndDeformedImageWriter(config)
    evaluationWriter = PredictionEvaluationWriter(config)
    trainer = pl.Trainer(
        accelerator=config.getParam("accelerator"),
        devices="auto",
        precision=32,
        logger=logger,
        callbacks=[pred_writer, evaluationWriter],
    )
    start = dt.datetime.now()

    print("Training started at", start)
    trainer.test(model=model, datamodule=data)
    print("Training duration:", dt.datetime.now() - start)


def runPrediction(config):
    model, data = getModelAndData(config, "test")
    pred_writer = DeformationFieldAndDeformedImageWriter(config, isStageTypePredict=True)
    evaluationWriter = PredictionEvaluationWriter(config)
    trainer = pl.Trainer(
        accelerator=config.getParam("accelerator"),
        devices="auto",
        precision=32,
        callbacks=[pred_writer, evaluationWriter],
    )

    start = dt.datetime.now()
    print("Training started at", start)
    _ = trainer.predict(model=model, dataloaders=data.predict_dataloader())
    print("Training duration:", dt.datetime.now() - start)


def runTestImgSampling(config, nuOfFilesToWrite):
    network = NetworkFactory.getNetwork(config)
    newShape = network.getShapeForModel(config.getParam("registrationGridsize"))
    config.setParam("registrationGridsize", newShape.tolist())

    transformer = Transformation()

    output_dir = config.getParam("outputPath")
    meshDir = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    meshSpacing = config.getParam("registrationGridSpacing")

    data = AtlasDataModule(config)
    data.setup(stage="fit")

    atlasImage, atlasMesh, atlasOrigin, atlasLabel = data.getInitalAtlas()

    model = AtlasModule(
        network,
        atlasImage,
        atlasLabel,
        atlasMesh,
        atlasOrigin,
        None,
        networkLearning_rate=config.getParam("learningRate"),
        atlasLearning_rate=config.getParam("atlasLearningRate"),
        networkOptimizer_class=None,
        atlasOptimizer_class=None,
        useLrScheduler=config.getParam("lrScheduler"),
    )

    sampledAtlas = transformer.sampleImage(atlasImage, atlasMesh)

    atlasOrigin = atlasOrigin.tolist()
    atlas_utils.saveImageTensor(
        sampledAtlas[0, None, ...],
        os.path.join(output_dir, "AtlasImageSampled.nrrd"),
        atlasOrigin,
        meshSpacing,
        meshDir,
    )

    filesWritten = 0
    for batch in data.train_dataloader():
        images, meshes, _ = model.prepare_batch(batch)
        imageNames = batch["imagePath"]
        meshOrigin = batch["meshOrigin"]
        networkImageToRegInput = transformer.sampleImage(images, meshes)
        for i in range(0, networkImageToRegInput.shape[0]):
            fileBaseName = os.path.splitext(os.path.basename(imageNames[i]))[0]
            atlas_utils.saveImageTensor(
                networkImageToRegInput[i, None, ...],
                os.path.join(output_dir, fileBaseName + str(filesWritten) + "Sampled.nrrd"),
                meshOrigin[i].tolist(),
                meshSpacing,
                meshDir,
            )

            filesWritten = filesWritten + 1
            if filesWritten >= nuOfFilesToWrite:
                return


def runTraining(config, resume: str = None):
    seed = config.getParam("seed")

    if seed is not None:
        atlas_utils.setSeeds(seed)

    matMulPrecision = config.getParam("matMulPrecision")
    if matMulPrecision:
        atlas_utils.setMatmulPrecision(matMulPrecision)

    stringForStoringVariables = getCheckPointString(config)

    data = AtlasDataModule(config)
    data.prepare_data()
    data.setup(stage="fit")
    atlasImage, atlasMesh, atlasOrigin, atlasLabel = data.getInitalAtlas()

    if resume is None:
        network = NetworkFactory.getNetwork(config)
        newShape = network.getShapeForModel(config.getParam("registrationGridsize"))
        config.setParam("registrationGridsize", newShape.tolist())
        loss = LossCalculator(config)
        optimizer = atlas_utils.getOptimizer(config.getParam("optimizer"))

        model = AtlasModule(
            network,
            atlasImage,
            atlasLabel,
            atlasMesh,
            atlasOrigin,
            loss,
            networkLearning_rate=config.getParam("learningRate"),
            atlasLearning_rate=config.getParam("atlasLearningRate"),
            networkOptimizer_class=optimizer,
            atlasOptimizer_class=optimizer,
            useLrScheduler=config.getParam("lrScheduler"),
            logTemporaryDeformationFields=config.getParam("logTemporaryDeformationFields"),
        )
    else:
        f = open(resume, "r")
        checkPointPath = f.read().splitlines()[0]
        logging.warn(f"trying to load model file from {resume}")
        model = AtlasModule.load_from_checkpoint(checkPointPath)
        newShape = model.net.getShapeForModel(config.getParam("registrationGridsize"))
        config.setParam("registrationGridsize", newShape.tolist())
        loss = LossCalculator(config)
        model.criterion = loss
        model.hparams["loss"] = loss
        model.configure_optimizers()
        model.setAtlasInformation(
            atlasImage, atlasLabel, atlasMesh, atlasOrigin, atlasLearning_rate=config.getParam("atlasLearningRate")
        )

    callBackFunctions = []

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=config.getParam("checkPointPath"),
        filename=stringForStoringVariables + "-{epoch:02d}-{val_loss:.2f}",
        every_n_epochs=config.getParam("saveEveryEpoch"),
        monitor="val_loss",
        mode="min",
        save_top_k=3,
    )
    callBackFunctions.append(checkpoint_callback)

    # callBackFunctions.append(pl.callbacks.DeviceStatsMonitor())

    if config.getParam("tuneHyperParams"):
        metrics = {
            "atlasLabelSim": "ptl/label_sim_loss_atlas_space",
            "labelSimAtlasSpace": "ptl/atlas_space_label_loss",
        }
        callBackFunctions.append(TuneReportCallback(metrics, on="validation_epoch_end"))

    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval="step")
    callBackFunctions.append(lr_monitor)

    logger = TensorBoardLogger("tb_logs", name=stringForStoringVariables)
    meshDir = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    meshOrigin = atlasOrigin.tolist()
    meshSpacing = config.getParam("registrationGridSpacing")
    imageLogger = ImageLogger(
        "tb_logs",
        name=stringForStoringVariables,
        imageOrigin=meshOrigin,
        imageSpacing=meshSpacing,
        imageDirections=meshDir,
        version=logger.version,
    )

    profiler = pl.profilers.AdvancedProfiler(dirpath=".", filename="perf_logs")
    # profiler = pl.profilers.PyTorchProfiler(
    #     on_trace_ready=torch.profiler.tensorboard_trace_handler("./tb_logs/profile0")
    # )

    trainer = pl.Trainer(
        accelerator=config.getParam("accelerator"),
        devices="auto",
        # strategy="auto",
        precision=32,
        callbacks=callBackFunctions,
        auto_lr_find=config.getParam("tuneLR"),
        logger=[logger, imageLogger],
        profiler=profiler,  # "pytorch"
        deterministic="warn",
        check_val_every_n_epoch=5,
        max_epochs=config.getParam("epochs"),
    )

    trainer.tune(model, datamodule=data)

    trainer.logger._default_hp_metric = False

    start = dt.datetime.now()

    print("Training started at", start)
    trainer.fit(model=model, datamodule=data)
    print("Training duration:", dt.datetime.now() - start)

    if checkpoint_callback.best_model_path:
        f = open(os.path.join(config.getParam("checkPointPath"), stringForStoringVariables + ".txt"), "w")
        f.write(checkpoint_callback.best_model_path)
        f.close()


parser = argparse.ArgumentParser(description="Atlas Registration")
parser.add_argument("-c", "--configFile", dest="configFile", help="configuration file")
parser.add_argument("-t", "--test", dest="runTests", action="store_true", help="run tests with best model")
parser.add_argument("-p", "--predict", dest="predict", action="store_true")
parser.add_argument("-o", "--optimiseParams", dest="hpyerSearch", action="store_true")
parser.add_argument("-a", "--analyseHyperParamSearch", dest="analyseHyperSearch", action="store_true")
parser.add_argument("-s", "--testSampling", dest="testSampling", default=0, type=int)
parser.add_argument("-r", "--resume", dest="resume")


def valiateConfigFile(configuration: Config):
    useAtlasSpaceAsReferenceForMeshCreation = configuration.getParam("useAtlasSpaceAsReferenceForMeshCreation")
    atlasImage = configuration.getParam("atlasImage")
    initializeAtlasWithAverageImg = configuration.getParam("initializeAtlasWithAverageImg")

    retVal = True

    if useAtlasSpaceAsReferenceForMeshCreation:
        if initializeAtlasWithAverageImg:
            retVal = False
            logging.warn(
                "Configuration mistake: initializeAtlasWithAverageImg and useAtlasSpaceAsReferenceForMeshCreation must not both be true"
            )
        if atlasImage is None or not os.path.exists(atlasImage):
            retVal = False
            logging.warn(
                "Configuration mistake: when useAtlasSpaceAsReferenceForMeshCreation is true, atlasImage has to be set to a valid value"
            )

    return retVal


def runHyperParamSearchTraining(configDict: dict):
    configObj = Config()
    configObj.setParams(configDict)
    runTraining(configObj)


def analyseHyperParamSearch(config: Config):
    analysisPath = os.path.join(RayDefaultStoratePath, "tune_atlas")
    analysis = tune.ExperimentAnalysis(analysisPath, default_metric="atlasLabelSim", default_mode="min")
    best_result = analysis.best_result
    print("##### Best config #########")
    print(best_result)
    print("#### experiment directory #####")
    print(analysis.experiment_path)
    print("## Checkpoint best config ##")
    print(analysis.best_checkpoint)
    resultDataFrame = analysis.results_df

    resultDataFrame.to_csv(os.path.join(config.getParam("outputPath"), "HyperParamSearchResults.csv"))


def runHyperParamSearch(config: Config):
    trainable = tune.with_parameters(runHyperParamSearchTraining)
    max_epochs = config.getParam("epochs")
    scheduler = ASHAScheduler(
        max_t=max_epochs,
        grace_period=1,
        reduction_factor=2,
    )

    algo = OptunaSearch()
    algo = ConcurrencyLimiter(algo, max_concurrent=4)
    numSamples = 1000  # 10

    analysis = tune.run(
        trainable,
        metric="atlasLabelSim",
        mode="min",
        config=config.getParams(),
        search_alg=algo,
        num_samples=numSamples,
        name="tune_atlas",
        scheduler=scheduler,
        resources_per_trial={
            "cpu": 4,
            "gpu": 1,
        },
        resume=True,
    )

    best_result = analysis.best_result
    print("##### Best config #########")
    print(best_result)
    print("#### experiment directory #####")
    print(analysis.experiment_path)
    print("## Checkpoint best config ##")
    print(analysis.best_checkpoint)


if __name__ == "__main__":
    args = parser.parse_args()

    configFile = args.configFile
    if configFile:
        config = Config(configFile)
    else:
        config = Config()

    if valiateConfigFile(config):
        if args.runTests:
            runTests(config)
        elif args.predict:
            runPrediction(config)
        elif args.hpyerSearch:
            if args.analyseHyperSearch:
                analyseHyperParamSearch(config)
            else:
                runHyperParamSearch(config)
        elif args.testSampling > 0:
            runTestImgSampling(config, args.testSampling)
        elif args.resume:
            runTraining(config, resume=args.resume)
        else:
            runTraining(config)
