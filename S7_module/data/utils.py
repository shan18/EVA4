import os
import numpy as np

from torchvision import datasets


def download_mnist(train=True, transform=None):
    """Download MNIST dataset

    Args:
        train: If True, download training data else test data.
            Defaults to True.
        transform: Data transformations to be applied on the data.
            Defaults to None.
    
    Returns:
        Downloaded dataset.
    """

    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mnist')
    return datasets.MNIST(
        data_path, train=train, download=True, transform=transform
    )


def calculate_mean_and_std():
    """Calculate the mean and standard deviation of the MNIST dataset.

    Returns:
        Mean and standard deviation value of the MNIST dataset.
    """

    # Download data
    data = download_mnist().data

    # Setting the values in the data to be within the range [0, 1]
    data = data.numpy() / 255

    # Calculate and return mean and std
    return np.mean(data), np.std(data)
