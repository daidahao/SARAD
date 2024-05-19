<div align="center">

# SAR76: Spatial Associations for Time Series Anomaly Detection and Diagnosis

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>
[![Paper](http://img.shields.io/badge/paper-arxiv.1001.2234-B31B1B.svg)](https://www.nature.com/articles/nature14539)
[![Conference](http://img.shields.io/badge/AnyConference-year-4b44ce.svg)](https://papers.nips.cc/paper/2020)

</div>

## Description
Anomaly detection in time series data is fundamental to the design, deployment, and evaluation of industrial control systems. Temporal modeling has been the natural focus of anomaly detection approaches for time series data. However, the focus on temporal modeling can obscure or dilute the spatial information that can be used to capture complex interactions in multivariate time series. In this paper, we propose SAR76, an approach that leverages spatial information beyond data reconstruction errors to improve the detection and diagnosis of anomalies in such systems. SAR76 trains a Transformer to learn the spatial associations and to capture their changes over time via series division. Anomalies exhibit association descending patterns and SAR76 exploits that via reconstruction in the association reduction space. We present experimental results to demonstrate that SAR76 achieves state-of-the-art performances, providing robust anomaly detection and a nuanced understanding of anomalous events.

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
