from model.train import train
from model.evaluate import val
from model.network import Net
from model.utils import sgd_optimizer, lr_scheduler, model_summary
from data.dataset import mnist_dataset


def create_data_loaders(train_batch_size, val_batch_size, cuda, num_workers, augmentation, rotation):
    """ Create training and validation dataset loaders. """

    # Create train data loader
    train_loader = mnist_dataset(
        train_batch_size,
        cuda,
        num_workers,
        train=True,
        augmentation=augmentation,
        rotation=rotation
    )

    # Create val data loader
    val_loader = mnist_dataset(
        val_batch_size,
        cuda,
        num_workers,
        train=False
    )

    return train_loader, val_loader


def run_model(
    device, train_loader, val_loader, epochs, learning_rate, momentum, dropout, step_size, gamma, l1=0.0, l2=0.0
):
    losses = []
    accuracies = []
    incorrect_samples = []
    
    print('\nCreating model')
    model = Net(dropout).to(device)  # Create model
    model_summary(model)  # Display model summary

    optimizer = sgd_optimizer(model, learning_rate, l2, momentum)  # Create optimizer
    scheduler = lr_scheduler(optimizer, step_size, gamma)  # Set LR scheduler

    for epoch in range(1, epochs + 1):
        print(f'Epoch {epoch}:')
        train(model, device, train_loader, optimizer, epoch, l1)
        scheduler.step()
        val(model, device, val_loader, losses, accuracies, incorrect_samples)
    
    return losses, accuracies, incorrect_samples
