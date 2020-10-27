# MADDPG-Based Mobile Crowdsensing

Contents:

- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Citation](#citation)

## Introduction

This is a repository for "[Multi-Agent Actor-Critic Network-Based Incentive Mechanism for Mobile Crowdsensing in Industrial System](https://ieeexplore.ieee.org/document/9201550)".

## Installation

- Python 3.7.3
- Numpy
- cvxpy
- matplotlib
- pyyaml
- tensorboardX
- pytorch

You can install all the dependencies with `pip`:

```
cd environment
pip install -r requirments.txt
```

## Usage

There is a simple script with which you will begin the training process. A sample command to launch training is
```
python train.py
```

Moreover, the settings for training are decribed in `platform-config.yaml` at `\code\environment` directory. And if needed, you can modify the script to satisfy other purposes.

## Citation

If this repository do some help, please consider to cite the paper "Multi-Agent Actor-Critic Network-Based Incentive Mechanism for Mobile Crowdsensing in Industrial System".

### Format

> B. Gu, X. Yang, Z. Lin, W. Hu, M. Alazab and R. Kharel, "Multi-Agent Actor-Critic Network-based Incentive Mechanism for Mobile Crowdsensing in Industrial Systems," in IEEE Transactions on Industrial Informatics, doi: 10.1109/TII.2020.3024611.