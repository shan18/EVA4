import argparse


def main(args):
    pass


if __name__ == '__main___':
    parser = argparse.ArgumentParser(
        description='Implementing L1 and L2 regularization on a model being trained on MNIST dataset.'
    )

    # Data Loading
    # ============

    parser.add_argument('--batch_size', type=int, default=64, help='Number of images per batch')
    parser.add_argument('--num_workers', type=int, default=4, help='How many subprocesses to use for data loading')
    parser.add_argument('--apply_augmentation', type=bool, default=True, help='If True, apply data augmentation')
    parser.add_argument('--rotation_degree', type=float, default=6.0, help='Angle of rotation of images for image augmentation')

    # Model
    # =====

    parser.add_argument('--dropout', type=float, default=0.01, help='Dropout rate')

    # Training
    # ========
    parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate for the optimizer')
    parser.add_argument('--l1', type='float', default=0.001, help='Factor for L1 regularization')
    parser.add_argument('--l2', type='float', default=0.0001, help='Factor for L2 regularization')

    args = parser.parse_args()

    main(args)
