import os
import matplotlib.pyplot as plt


def plot_metric(plain, l1, l2, l1_l2, metric):
    """Plot changes in validation loss and accuracy obtained during model training.

    Args:
        plain: Change in values of specified metric in the model trained
            without L1 and L2 regularization.
        l1: Change in values of specified metric in the model trained
            with L1 regularization.
        l2: Change in values of specified metric in the model trained
            with L2 regularization.
        l1_l2: Change in values of specified metric in the model trained
            with both L1 and L2 regularization.
        metric: Name of the metric whose change in values will be plotted.
    """

    # Initialize a figure
    fig = plt.figure(figsize=(13, 11))

    # Plot values
    plain_plt, = plt.plot(plain)
    l1_plt, = plt.plot(l1)
    l2_plt, = plt.plot(l2)
    l1_l2_plt, = plt.plot(l1_l2)

    # Set plot title
    plt.title(f'Validation {metric}')

    # Label axes
    plt.xlabel('Epoch')
    plt.ylabel(metric)

    # Set legend
    location = 'upper' if metric == 'Loss' else 'lower'
    plt.legend(
        (plain_plt, l1_plt, l2_plt, l1_l2_plt),
        ('Plain', 'L1', 'L2', 'L1 + L2'),
        loc=f'{location} right',
        shadow=True,
        prop={'size': 20}
    )

    # Save plot
    fig.savefig(f'{metric.lower()}_change.png')


def save_and_show_result(data, metric):
    """Display 25 misclassified images.

    Args:
        data: Contains model predictions and labels.
        metric: Name of the model.
    """

    # Create directories for saving data
    metric_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'predictions', metric
    )
    labelled_path = os.path.join(metric_path, 'labelled')  # To store individual misclassified images
    if not os.path.exists(labelled_path):
        os.makedirs(labelled_path)

    # Initialize plot
    row_count = -1
    fig, axs = plt.subplots(5, 5, figsize=(10, 10))
    fig.tight_layout()

    for idx, result in enumerate(data):

        # If 25 samples have been stored, break out of loop
        if idx > 24:
            break
        
        label = result['label'].item()
        prediction = result['prediction'].item()

        # Plot image
        if idx % 5 == 0:
            row_count += 1
        axs[row_count][idx % 5].axis('off')
        axs[row_count][idx % 5].set_title(f'Label: {label}\nPrediction: {prediction}')
        axs[row_count][idx % 5].imshow(result['image'][0], cmap='gray_r')

        # Save each image individually in labelled format
        extent = axs[row_count][idx % 5].get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        fig.savefig(f'{labelled_path}/{metric}_{idx + 1}.png', bbox_inches=extent.expanded(1.1, 1.5))
    
    # Save image
    fig.savefig(f'{metric_path}/{metric}_incorrect_predictions.png', bbox_inches='tight')
