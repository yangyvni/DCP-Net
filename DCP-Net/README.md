# <p align=center> DCP-Net: Learning Detail–Context Perception via
Spatial-Frequency Guidance for Tiny Object
Detection in Remote Sensing Images </p>

<!-- This repository contains python implementation of our paper [DCP-Net](https://ieeexplore.ieee.org/document/10462223). -->

### 1. Required environments:
* Linux
* Python 3.9
* PyTorch 1.12.1 
* CUDA 11.8 
* GCC 5+

### 2. Install and start DCP-Net:

Note that our DCP-Net is based on the [YOLO11](https://github.com/ultralytics/ultralytics). Assume that your environment has satisfied the above requirements, please follow the following steps for installation.

```shell script
# setup_env.sh
conda env create -f environment.yaml
conda activate dcpnet

pip install -r requirements.txt
```
### Prepare Dataset:
Download [AI-TODv2](https://drive.google.com/drive/folders/1Er14atDO1cBraBD4DSFODZV1x7NHO_PY?usp=sharing) dataset; Download [DOTA-v1.0](https://captain-whu.github.io/DOTA/dataset.html) dataset.

### Train and test：
##### Train aitodv2 dataset:
```
python train.py # dataset: dataset/dataset.yaml
```
##### Train dotav1.0 dataset:
```
python train.py # dataset: dataset/dataset_dota.yaml
```
##### Test dataset:
```
python test.py 
```
