# Session 7 - Advanced Convolutions

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1PD07tY3pCloDfQVl_F7pcFWtr6TcPysj)

The model reaches a test accuracy of **84.83%** in **CIFAR-10** dataset. The model uses the following types of convolutions:

- 3x3 Convolution
- Pointwise Convolution
- Atrous Convolution
- Depthwise Separable Convolution
- Max Pooling

The model has **94,218 parameters**.

## Model Architecture

![architecture](images/architecture.png)

### Parameters and Hyperparameters

- Loss Function: Cross Entropy Loss
- Optimizer: SGD
- Learning Rate: 0.01
- Dropout Rate: 0.1
- Batch Size: 64
- Epochs: 50

## Change in Validation Loss and Accuracy

<img src="images/loss_change.png" width="450px">
<img src="images/accuracy_change.png" width="450px">

## Project Setup

### On Local System

Install the required packages  
 `$ pip install -r requirements.txt`

### On Google Colab

Upload the files in the root folder and select Python 3 as the runtime type and GPU as the harware accelerator.

## Group Members

- Rakhee (Canvas ID: 25180625)
- Shantanu Acharya (Canvas ID: 25180630)
