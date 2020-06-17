import torch
from torchvision import transforms


def transformations(augmentation=False, rotation=0.0):
    """Create data transformations
    
    Args:
        augmentation: Whether to apply data augmentation.
            Defaults to False.
        rotation: Angle of rotation for image augmentation.
            Defaults to 0. It won't be needed if augmentation is False.
    
    Returns:
        Transform object containing defined data transformations.
    """

    transforms_list = [
        # convert the data to torch.FloatTensor
        # with values within the range [0.0 ,1.0]
        transforms.ToTensor(),

        # normalize the data with mean and standard deviation to keep values in range [-1, 1]
        # since there are 3 channels for each image,
        # we have to specify mean and std for each channel
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]

    if augmentation:
        transforms_list = [
            # Rotate image by 6 degrees
            transforms.RandomRotation((-rotation, rotation), fill=(1,))
        ] + transforms_list
    
    return transforms.Compose(transforms_list)


def data_loader(data, batch_size, num_workers, cuda):
    """Create data loader

    Args:
        data: Downloaded dataset.
        batch_size: Number of images to considered in each batch.
        num_workers: How many subprocesses to use for data loading.
        cuda: True is GPU is available.
    
    Returns:
        DataLoader instance.
    """

    loader_args = {
        'shuffle': True,
        'batch_size': batch_size
    }

    # If GPU exists
    if cuda:
        loader_args['num_workers'] = num_workers
        loader_args['pin_memory'] = True
    
    return torch.utils.data.DataLoader(data, **loader_args)
