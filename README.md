# Atlas Registration
A preprocessing-free, lesion-aware deep learning framework for robust atlas registration.

This repository contains a deep-learning atlas registration framework designed for pathological images, with a special focus on cases where lesions have no anatomical counterpart in the atlas. The method operates directly on native medical images—no preprocessing or lesion masks required—and robustly handles missing correspondences using distance-map–based similarity and a volume-preserving loss. It supports one-shot overfitting for patient-specific refinement and achieves high-accuracy, anatomically plausible registrations across multi-centre clinical datasets. The framework enables reproducible cohort-level spatial analyses and has been successfully applied to melanoma brain metastases across multiple institutions.

**Please note:** There is currently no maintained main branch, please check out the *refactoring* branch instead.

This is a highly refactored fork of **Aladdin** that I adapted to my needs for atlas registration. Here you can find the original work:
**Aladdin: Joint Atlas Building and Diffeomorphic Registration Learning with Pairwise Alignment**   
[Zhipeng Ding](https://biag.cs.unc.edu/author/zhipeng-ding/) and [Marc Niethammer](https://biag.cs.unc.edu/author/marc-niethammer/)   
*CVPR 2022* [eprint arxiv](https://arxiv.org/abs/2202.03563)

## Key Features

- **Preprocessing-free**: Works directly on native medical images
- **Lesion-aware**: Handles cases where lesions have no anatomical counterpart in the atlas
- **Robust registration**: Uses distance-map-based similarity and volume-preserving loss
- **Multi-center support**: Achieves high-accuracy registrations across clinical datasets
- **Flexible training**: Supports calssical model training as well as one-shot overfitting for patient-specific refinement

## Installation

For managing dependencies we use [Poetry](https://python-poetry.org/docs/basic-usage/).
After checking out the repository call:
```
poetry install
```
To activate the environment call:
```
poetry shell
```

## Basic Usage

The main entry point for the framework is `TrainAtlas.py`. This script provides multiple modes of operation through command-line arguments:

### Training a Model
To start training a new model, run:
```bash
python ./code/TrainAtlas.py -c ./Path/to/your/config/file.json
```

### Testing a Trained Model
To test a trained model with the best checkpoint, use:
```bash
python ./code/TrainAtlas.py -c ./Path/to/your/config/file.json -t
```
In testing mode the framwork will store the results in the *outputPath* configured in the config-file.

### Making Predictions
To make predictions with a trained model, use:
```bash
python ./code/TrainAtlas.py -c ./Path/to/your/config/file.json -p
```
In prediction mode the framwork will store the results for each input dataset in the folder of the respective dataset.

### Resuming Training
To resume training from a specific checkpoint, use:
```bash
python ./code/TrainAtlas.py -c ./Path/to/your/config/file.json -r path/to/checkpoint.txt
```

### Testing Image Sampling
To test image sampling functionality (e.g., for debugging), use:
```bash
python ./code/TrainAtlas.py -c ./Path/to/your/config/file.json -s 10
```
This will sample 10 images from the dataset.

### Hyperparameter Optimization
To run hyperparameter optimization with [TUNE](https://docs.ray.io/en/latest/tune/index.html):
```bash
python ./code/TrainAtlas.py -c ./Path/to/your/config/file.json -o
```

### Analyzing Hyperparameter Search Results
To analyze results from a hyperparameter search:
```bash
python ./code/TrainAtlas.py -c ./Path/to/your/config/file.json -a
```

## Configuration

The framework uses JSON configuration files to specify all parameters. A sample configuration file can be found in the `resources` directory. The configuration file includes parameters for:

- Data paths and settings
- Network architecture
- Loss functions
- Optimization settings
- Logging and checkpointing
- Registration grid parameters
- Learning rates
- Loss weights

