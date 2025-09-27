import matplotlib.pyplot as plt
import numpy as np

def plot_multilabel_confusion_matrix(cm, class_names):
    num_classes = cm.shape[0]
    ncols = 3  # Set the number of columns for the plot
    nrows = (num_classes + ncols - 1) // ncols  # Calculate the number of rows needed

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 4, nrows * 4))
    axes = axes.flatten()  # Flatten the 2D array of axes for easy iteration

    for i in range(num_classes):
        ax = axes[i]
        ax.matshow(cm[i], cmap=plt.cm.Blues, alpha=0.5)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title(class_names[i])

        # Set x and y axis ticks to show "Positive" first and "Negative" second
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Positive', 'Negative'])  # Positive first
        ax.set_yticks([0, 1])
        ax.set_yticklabels(['Positive', 'Negative'])  # Positive first

        # Show the counts
        for j in range(cm[i].shape[0]):
            for k in range(cm[i].shape[1]):
                ax.text(k, j, cm[i][j, k], ha='center', va='center')

    # Hide any unused subplots
    for i in range(num_classes, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()
