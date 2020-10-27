# MADDPG-Based-MCS

Contents:

- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Citation](#citation)

## Introduction

This is a repository for "This is a repository for "Multi-Agent Actor-Critic Network-Based Incentive Mechanism for Mobile Crowdsensing in Industrial System".

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
