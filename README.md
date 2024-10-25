<div align="center">

# SARAD: Spatial Association-Aware Anomaly Detection and Diagnosis for Multivariate Time Series

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>
[![Paper](https://img.shields.io/badge/paper-neurips.2024.94119-B31B1B.svg)](https://neurips.cc/virtual/2024/poster/94119)
[![Conference](https://img.shields.io/badge/NeurIPS-2024-4b44ce.svg)](https://neurips.cc)

</div>

This is the official repo for **SARAD (NeurIPS 2024)**.

## Description
Anomaly detection in time series data is fundamental to the design, deployment, and evaluation of industrial control systems. We propose SARAD, an approach that leverages spatial information beyond data autoencoding errors to improve the detection and diagnosis of anomalies. SARAD trains a Transformer to learn the spatial associations, the pairwise inter-feature relationships which ubiquitously characterize such feedback-controlled systems. As new associations form and old ones dissolve, SARAD applies subseries division to capture their changes over time. Anomalies exhibit association descending patterns, a key phenomenon we exclusively observe and attribute to the disruptive nature of anomalies detaching anomalous features from others. To exploit the phenomenon and yet dismiss non-anomalous descent, SARAD performs anomaly detection via autoencoding in the association space.

## Installation

```bash

# [OPTIONAL] create conda environment
conda create -n sar76 python=3.9
conda activate sar76

# install pytorch according to instructions
# https://pytorch.org/get-started/

# install requirements
pip install -r requirements.txt
```


## How to run

Train model with default configuration

```bash
# train on CPU
python src/train.py data=smd  trainer=cpu

# train on GPU
python src/train.py data=smd  trainer=gpu
```


You can override any parameter from command line like this

```bash
python src/train.py data=smd  data.batch_size=64
```

## Acknowledgments
Part of our implementation is adapted from [Volume Under the Surface](https://github.com/TheDatumOrg/VUS) and [PyTorch Tutorials](https://pytorch.org/tutorials/).
We are grateful for the author of both.


## Citation

If you find our work useful, please consider citing our work.

- [SARAD: Spatial Association-Aware Anomaly Detection and Diagnosis for Multivariate Time Series.](https://openreview.net/forum?id=gmf5Aj01Hz) Zhihao Dai, Ligang He, Shuang-Hua Yang, Matthew Leeke, In the Thirty-Eighth Annual Conference on Neural Information Processing Systems (NeurIPS), 2024.

```BibTeX
@inproceedings{
dai2024sarad,
title={{SARAD}: Spatial Association-Aware Anomaly Detection and Diagnosis for Multivariate Time Series},
author={Dai, Zhihao and He, Ligang and Yang, Shuanghua and Leeke, Matthew},
booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems},
year={2024},
url={https://openreview.net/forum?id=gmf5Aj01Hz}
}
```
