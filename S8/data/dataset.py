from data.downloader import download_cifar10
from data.processing import transformations, data_loader


class CIFAR10:
    """ Load CIFAR-10 Dataset. """

    def __init__(self, batch_size=1, cuda=False, num_workers=1, path=None, augmentation=False, rotation=0.0):
        """Initializes the dataset for loading.

        Args:
            batch_size: Number of images to consider in each batch.
            cuda: True is GPU is available.
            num_workers: How many subprocesses to use for data loading.
            path: Path where dataset will be downloaded. Defaults to None.
                If no path provided, data will be downloaded in a pre-defined
                directory.
            augmentation: Whether to apply data augmentation.
                Defaults to False.
            rotation: Angle of rotation of images for image augmentation.
                Defaults to 0. It won't be needed if augmentation is False.
        """
        
        self.cuda = cuda
        self.num_workers = num_workers
        self.path = path
        self.batch_size = batch_size

        # Define classes present in the dataset
        self.classes = (
            'plane', 'car', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck'
        )

        # Set data augmentation parameters
        self.augmentation = augmentation
        self.rotation = rotation

        # Set transforms
        self.train_transform = self._transform()
        self.val_transform = self._transform(train=False)

        # Download dataset
        self.train_data = self._download()
        self.val_data = self._download(train=False)
    
    def _transform(self, train=True):
        """Define data transformations
        
        Args:
            train: If True, download training data else test data.
                Defaults to True.
            augmentation: Whether to apply data augmentation.
                Defaults to False.
            rotation: Angle of rotation of images for image augmentation.
                Defaults to 0. It won't be needed if augmentation is False.
        
        Returns:
            Returns data transforms based on the training mode.
        """
        return transformations(self.augmentation, self.rotation) if train else transformations()
    
    def _download(self, train=True):
        """Download dataset.

        Args:
            train: True for training data.
        
        Returns:
            Downloaded dataset.
        """
        return download_cifar10(self.path, train=train, transform=self.train_transform)
    
    def get_classes(self):
        """ Return list of classes in the dataset. """
        return self.classes
    
    def get_data(self, train=True):
        """ Return data based on train mode.

        Args:
            train: True for training data.
        
        Returns:
            Training or validation data.
        """
        return self.train_data if train else self.val_data
    
    def loader(self, train=True):
        """Create data loader.

        Args:
            train: True for training data.
        
        Returns:
            Dataloader instance.
        """

        loader_args = {
            'batch_size': self.batch_size,
            'num_workers': self.num_workers,
            'cuda': self.cuda
        }

        return data_loader(
            self.train_data, **loader_args
        ) if train else data_loader(self.val_data, **loader_args)
