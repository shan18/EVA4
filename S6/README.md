# Session 6 - Regularization

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1uB_O0TfklhPlV0Ycl9VcwratJVX_X8Vc)

The goal of this assignment is to apply L1 and L2 regularization on the final model from the [previous](../S5/) session and plot the changes in validation loss and accuracy obtained during model training in the following scenarios:

1. Without L1 and L2 regularization
2. With L1 regularization
3. With L2 regularization
4. With L1 and L2 regularization

After model training, display 25 misclassified images for L1 and L2 models.

## Model Architecture

![architecture](images/architecture.png)

### Parameters and Hyperparameters

- Kernel Size: 3x3
- Loss Function: Negative Log Likelihood
- Optimizer: SGD
- Dropout Rate: 0.01
- Batch Size: 64
- Learning Rate: 0.01
- **L1 Factor:** 0.001
- **L2 Factor:** 0.0001

## Results

### Change in Validation Loss and Accuracy

<img src="images/loss_change.png" width="600px">
<img src="images/accuracy_change.png" width="600px">

## Misclassified Images

### With L1 Regularization

![plain](images/l1_incorrect_predictions.png)

### With L2 Regularization

![plain](images/l2_incorrect_predictions.png)

## Project Setup

### On Local System

Install the required packages  
 `$ pip install -r requirements.txt`

### On Google Colab

Select Python 3 as the runtime type and GPU as the harware accelerator.

## Group Members

- Shantanu Acharya (Canvas ID: 25180630)
- Rakhee (Canvas ID: 25180625)
