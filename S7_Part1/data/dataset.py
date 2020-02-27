from data.utils import download_mnist, calculate_mean_and_std
from data.processing import transformations, data_loader


def mnist_dataset(train, batch_size, cuda, num_workers, apply_augmentation, rotation_degree):
    """Download and create dataset.

    Args:
        train: If True, download training data else test data.
            Defaults to True.
        batch_size: Number of images to considered in each batch.
        cuda: True is GPU is available.
        num_workers: How many subprocesses to use for data loading.
        apply_augmentation: Whether to apply data augmentation.
            Defaults to False.
        rotation_degree: Angle of rotation of images for image augmentation.
            Defaults to 0. It won't be needed if apply_augmentation is False.
    
    Returns:
        Dataloader instance.
    """

    # Define data transformations
    mean, std = calculate_mean_and_std()
    if train:
        transforms = transformations(
            mean, std, apply_augmentation, rotation_degree
        )
    else:
        transforms = transformations(mean, std)

    # Download training and validation dataset
    data = download_mnist(train=train, transform=transforms)

    # create and return dataloader
    return data_loader(data, batch_size, num_workers, cuda)
