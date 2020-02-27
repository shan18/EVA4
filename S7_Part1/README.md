# Session 7 (Part 1) - Regularization

The goal of this assignment is to convert the code from the [previous](../S6/) session into a library where each functionality of the code must be present in a separate file. The main file should only contain the configuration parameters and call the required functions to execute the model.

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

- Install the required packages  
  `$ pip install -r requirements.txt`

## Group Members

- Shantanu Acharya (Canvas ID: 25180630)
- Rakhee (Canvas ID: 25180625)
