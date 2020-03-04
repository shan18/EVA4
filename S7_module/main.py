import argparse

from cuda import initialize_cuda
from run import create_data_loaders, run_model
from result import plot_metric, save_and_show_result


def execute(
    device, train_loader, val_loader, epochs, learning_rate, momentum, dropout, step_size, gamma, l1, l2
):
    """ Execute the four model types. """

    results = {x: {} for x in ['plain', 'l1', 'l2', 'l1_l2']}

    # Without L1 and L2 regularization
    print('\nTraining model without L1 and L2 regularization...')
    results['plain']['loss'], results['plain']['accuracy'], results['plain']['incorrect'] = run_model(
        device, train_loader, val_loader, epochs, learning_rate,
        momentum, dropout, step_size, gamma
    )

    # With L1 regularization
    print('\nTraining model with L1 regularization...')
    results['l1']['loss'], results['l1']['accuracy'], results['l1']['incorrect'] = run_model(
        device, train_loader, val_loader, epochs, learning_rate,
        momentum, dropout, step_size, gamma, l1=l1
    )

    # With L2 regularization
    print('\nTraining model with L2 regularization...')
    results['l2']['loss'], results['l2']['accuracy'], results['l2']['incorrect'] = run_model(
        device, train_loader, val_loader, epochs, learning_rate,
        momentum, dropout, step_size, gamma, l2=l2
    )

    # With L1 and L2 regularization
    print('\nTraining model with L1 and L2 regularization...')
    results['l1_l2']['loss'], results['l1_l2']['accuracy'], results['l1_l2']['incorrect'] = run_model(
        device, train_loader, val_loader, epochs, learning_rate,
        momentum, dropout, step_size, gamma, l1=l1, l2=l2
    )

    return results


def plot_loss_accuracy(results):
    """ Plot changes in loss and accuracy """

    # Plot loss of all models
    print('\nPlotting changes in validation loss.')
    plot_metric(
        results['plain']['loss'], results['l1']['loss'],
        results['l1']['loss'], results['l1_l2']['loss'], 'Loss'
    )

    # Plot accuracy of all models
    print('\nPlotting changes in validation accuracy.')
    plot_metric(
        results['plain']['accuracy'], results['l1']['accuracy'],
        results['l1']['accuracy'], results['l1_l2']['accuracy'], 'Accuracy'
    )


def save_incorrect_predictions(results):
    """ Save 25 misclassified images predicted from L1 and L2 models """

    # Save misclassified images from L1 model
    print('\nSaving misclassified images from L1 model')
    save_and_show_result(results['l1']['incorrect'], 'l1')
    print('Done.')

    # Save misclassified images from L2 model
    print('\nSaving misclassified images from L2 model')
    save_and_show_result(results['l2']['incorrect'], 'l2')
    print('Done.')


def main(args):
    # Initialize CUDA and set random seed
    cuda, device = initialize_cuda(args.random_seed)

    # Create data loaders
    train_loader, val_loader = create_data_loaders(
        args.train_batch_size, args.val_batch_size, cuda,
        args.num_workers, args.augmentation, args.rotation
    )

    # Execute models
    results = execute(
        device, train_loader, val_loader, args.epochs, args.learning_rate,
        args.momentum, args.dropout, args.step_size, args.gamma, args.l1, args.l2
    )

    # Plot loss and accuracy
    plot_loss_accuracy(results)

    # Save misclassified images
    save_incorrect_predictions(results)


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

    # Regularization
    # ==============

    parser.add_argument('--dropout', type=float, default=0.01, help='Dropout rate')
    parser.add_argument('--l1', type=float, default=0.001, help='Factor for L1 regularization')
    parser.add_argument('--l2', type=float, default=0.0001, help='Factor for L2 regularization')

    # Training
    # ========
    parser.add_argument('--random_seed', type=int, default=1, help='Random seed value for result reproducibility')
    parser.add_argument('--epochs', type=int, default=40, help='Number of epochs to train the model')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate for the optimizer')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum for the optimizer')
    parser.add_argument('--step_size', type=int, default=5, help='Frequency for changing learning rate')
    parser.add_argument('--gamma', type=float, default=0.15, help='Factor for changing learning rate')

    args = parser.parse_args()

    main(args)
