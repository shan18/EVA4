import torch.optim as optim

from torch.optim.lr_scheduler import StepLR
from torchsummary import summary

from model.network import Net


def sgd_optimizer(model, learning_rate, l2_factor, momentum):
    """Create optimizer.

    Args:
        model: Model instance.
        learning_rate: Learning rate for the optimizer.
        l2_factor: Factor for L2 regularization.
        momentum: Momentum of optimizer.
    
    Returns:
        SGD optimizer.
    """
    return optim.SGD(
        model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=l2_factor
    )


def lr_scheduler(optimizer, step_size, gamma):
    """Create LR scheduler.

    Args:
        optimizer: Model optimizer.
        step_size: Frequency for changing learning rate.
        gamma: Factor for changing learning rate.
    
    Returns:
        StepLR: Learning rate scheduler.
    """

    return StepLR(optimizer, step_size=step_size, gamma=gamma)


def model_summary(model):
    """Print model summary.

    Args:
        model: Model instance.
    """

    print(summary(model, input_size=(1, 28, 28)))
