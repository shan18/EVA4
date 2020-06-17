import os

from torchvision import datasets


def download_cifar10(train=True, transform=None):
    """Download CIFAR10 dataset

    Args:
        train: If True, download training data else test data.
            Defaults to True.
        transform: Data transformations to be applied on the data.
            Defaults to None.
    
    Returns:
        Downloaded dataset.
    """

    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cifar10')
    return datasets.CIFAR10(
        data_path, train=train, download=True, transform=transform
    )
