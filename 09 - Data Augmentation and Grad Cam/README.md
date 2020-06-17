# Session 9 - Data Augmentation and Grad Cam

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1ckckI9yP0TJtXe7m59H4Tak5OmlPXy2c)

The model reaches a maximum accuracy of **91.53%** in **41 epochs** on CIFAR-10 dataset using **ResNet-18** model.

**Gradient-weighted Class Activation Map (GradCAM)** was implemented for each convolution block to generate model prediction heatmaps (Examples shown below).

### Parameters and Hyperparameters

- Loss Function: Cross Entropy Loss (combination of `nn.LogSoftmax` and `nn.NLLLoss`)
- Optimizer: SGD
- Learning Rate: 0.01
- LR Step Size: 25
- LR Gamma: 0.1
- Batch Size: 64
- Epochs: 50

### Data Augmentation

The following data augmentation techniques were applied to the dataset during training:

- Horizontal Flip
- Rotation
- CutOut

The `albumentations` package was used to apply augmentation.

## GradCAM

Some of the examples where the network was focusing while predicting the output is shown below:

### Image 1

![grad_cam1](images/grad_cam_1.png)

### Image 2

![grad_cam2](images/grad_cam_2.png)

### Image 3

![grad_cam3](images/grad_cam_3.png)

### Image 4

![grad_cam4](images/grad_cam_4.png)

## Change in Validation Loss and Accuracy

<img src="images/loss_change.png" width="450px">
<img src="images/accuracy_change.png" width="450px">

## Correctly Classified Images

![correct_predictions](images/correct_predictions.png)

## Misclassified Images

![incorrect_predictions](images/incorrect_predictions.png)

## Project Setup

### On Local System

Install the required packages  
 `$ pip install -r requirements.txt`

### On Google Colab

Upload the files in the root folder and select Python 3 as the runtime type and GPU as the harware accelerator.

## Group Members

- Shantanu Acharya (Canvas ID: 25180630)
- Rakhee (Canvas ID: 25180625)
