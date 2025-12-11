# Atlas Registration
A preprocessing-free, lesion-aware deep learning framework for robust atlas registration.

This repository contains a deep-learning atlas registration framework designed for pathological images, with a special focus on cases where lesions have no anatomical counterpart in the atlas. The method operates directly on native medical images—no preprocessing or lesion masks required—and robustly handles missing correspondences using distance-map–based similarity and a volume-preserving loss. It supports one-shot overfitting for patient-specific refinement and achieves high-accuracy, anatomically plausible registrations across multi-centre clinical datasets. The framework enables reproducible cohort-level spatial analyses and has been successfully applied to melanoma brain metastases across multiple institutions.

**Please note:** There is currently no maintained main branch, please check out the *refactoring* branch instead.

This is a highly refactored fork of **Aladdin** that I adapted to my needs for atlas registration. Here you can find the original work:
**Aladdin: Joint Atlas Building and Diffeomorphic Registration Learning with Pairwise Alignment**   
[Zhipeng Ding](https://biag.cs.unc.edu/author/zhipeng-ding/) and [Marc Niethammer](https://biag.cs.unc.edu/author/marc-niethammer/)   
*CVPR 2022* [eprint arxiv](https://arxiv.org/abs/2202.03563)


## Installation & Usage

For managing dependencies we use [Poetry](https://python-poetry.org/docs/basic-usage/).
After checking out the repository call:
```
poetry install
```
To activate the environment call:
```
poetry shell
```
The main function is in TrainAtlas.py. To start training you can run:
```
python ./code/TrainAtlas.py -c ./Path/to/your/config/file.json
```

## License 





