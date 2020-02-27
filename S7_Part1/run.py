import argparse

from utils.cuda import initialize_cuda
from data.dataset import mnist_dataset


def main(args):
    # Initialize CUDA and set random seed
    cuda = initialize_cuda(args.random_seed)

    # Create train data loader
    train_loader = mnist_dataset(
        args.train_batch_size,
        cuda, args.num_workers,
        train=True,
        augmentation=args.augmentation,
        rotation=args.rotation
    )

    # Create val data loader
    val_loader = mnist_dataset(
        args.val_batch_size,
        cuda,
        args.num_workers,
        train=False
    )

    print(train_loader)
    print(val_loader)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Implementing L1 and L2 regularization on a model being trained on MNIST dataset.'
    )

    # Data Loading
    # ============

    parser.add_argument('--train_batch_size', type=int, default=64, help='Number of images per batch in training set')
    parser.add_argument('--val_batch_size', type=int, default=64, help='Number of images per batch in validation set')
    parser.add_argument('--num_workers', type=int, default=4, help='How many subprocesses to use for data loading')
    parser.add_argument('--augmentation', type=bool, default=True, help='If True, apply data augmentation')
    parser.add_argument('--rotation', type=float, default=6.0, help='Angle of rotation of images for image augmentation')

    # Model
    # =====

    parser.add_argument('--dropout', type=float, default=0.01, help='Dropout rate')

    # Training
    # ========
    parser.add_argument('--random_seed', type=int, default=1, help='Random seed value for result reproducibility')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate for the optimizer')
    parser.add_argument('--l1', type=float, default=0.001, help='Factor for L1 regularization')
    parser.add_argument('--l2', type=float, default=0.0001, help='Factor for L2 regularization')

    args = parser.parse_args()

    main(args)
