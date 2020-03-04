from data.downloader import download_cifar10
from data.processing import transformations, data_loader


def cifar10_classes():
    return (
        'plane', 'car', 'bird', 'cat', 'deer',
        'dog', 'frog', 'horse', 'ship', 'truck'
    )


def cifar10_dataset(batch_size, cuda, num_workers, train=True, augmentation=False, rotation=0.0):
    """Download and create dataset.

    Args:
        batch_size: Number of images to considered in each batch.
        cuda: True is GPU is available.
        num_workers: How many subprocesses to use for data loading.
        train: If True, download training data else test data.
            Defaults to True.
        augmentation: Whether to apply data augmentation.
            Defaults to False.
        rotation: Angle of rotation of images for image augmentation.
            Defaults to 0. It won't be needed if augmentation is False.
    
    Returns:
        Dataloader instance.
    """

    # Define data transformations
    if train:
        transforms = transformations(
            augmentation, rotation
        )
    else:
        transforms = transformations()

    # Download training and validation dataset
    data = download_cifar10(train=train, transform=transforms)

    # create and return dataloader
    return data_loader(data, batch_size, num_workers, cuda)
