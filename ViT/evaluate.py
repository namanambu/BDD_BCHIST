import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import Normalize


def plot_confusion_matrix(cm, title, save_path=None):
    '''
    Function: plot_confusion_matrix
    Purpose: Plots a clean 2x2 confusion matrix using the Blues colormap
             with no gridlines or colorbars.
    Inputs:
      cm        - np.ndarray [2, 2]; confusion matrix with layout
                  [[TN, FP], [FN, TP]].
      title     - str; title displayed above the matrix.
      save_path - str | None; if provided, saves the figure to this path
                  at 300 dpi. Defaults to None (display only).
    Returns:
      N/A
    Behaviour:
      Draws each cell as a Rectangle patch colored by the Blues colormap
      scaled to the matrix maximum. Overlays the cell count as bold text,
      white for dark cells and dark for light cells. Removes all spines,
      gridlines, and tick marks for a clean publication-ready appearance.
      Saves to save_path if provided.
    '''
    cmap   = plt.get_cmap('Blues')
    norm   = Normalize(vmin=0, vmax=cm.max())
    labels = ['Benign', 'Malignant']

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.set_facecolor('white')
    ax.grid(False)

    for i in range(2):
        for j in range(2):
            # draw each cell as a rectangle with no border
            rect = patches.Rectangle(
                [j, 1-i], 1, 1,
                linewidth=0, edgecolor='none',
                facecolor=cmap(norm(cm[i, j]))
            )
            ax.add_patch(rect)

            # white text on dark cells, dark text on light cells
            tc = 'white' if cm[i, j] > cm.max() * 0.55 else '#1a1a1a'
            ax.text(j+0.5, 1-i+0.5, str(cm[i, j]),
                    ha='center', va='center',
                    fontsize=16, fontweight='bold', color=tc)

    ax.set_xlim(0, 2); ax.set_ylim(0, 2)
    ax.set_xticks([0.5, 1.5]); ax.set_xticklabels(labels, fontsize=10)
    ax.set_yticks([0.5, 1.5]); ax.set_yticklabels(labels[::-1], fontsize=10,
                                                    rotation=90, va='center')
    ax.set_xlabel('Predicted label', fontsize=10)
    ax.set_ylabel('True label', fontsize=10)
    ax.set_title(title, fontsize=11, fontweight='bold', pad=10)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()


def plot_training_curves(history, save_path=None):
    '''
    Function: plot_training_curves
    Purpose: Plots binary classification AUC and multi-task loss curves
             over training epochs.
    Inputs:
      history   - dict; training history with keys train_binary_auc,
                  val_binary_auc, train_loss, val_loss. Each value is
                  a list of per-epoch values.
      save_path - str | None; if provided, saves the figure to this path
                  at 300 dpi. Defaults to None (display only).
    Returns:
      N/A
    Behaviour:
      Plots AUC curves on the left and loss curves on the right in a
      1x2 figure. Adds a horizontal dashed line marking the best
      validation AUC. Removes top and right spines and adds light
      gridlines for readability.
    '''
    epochs = np.arange(1, len(history['train_binary_auc']) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    for ax in [ax1, ax2]:
        ax.set_facecolor('white')
        ax.grid(True, color='#E8E6E0', linewidth=0.7, zorder=0)
        ax.spines[['top', 'right']].set_visible(False)

    # AUC plot
    ax1.plot(epochs, history['train_binary_auc'], 'o-',
             color='#1D9E75', lw=2, ms=5, label='Train AUC', zorder=3)
    ax1.plot(epochs, history['val_binary_auc'], 's--',
             color='#D85A30', lw=2, ms=5, label='Val AUC', zorder=3)
    ax1.axhline(max(history['val_binary_auc']), color='#D85A30',
                lw=0.8, ls=':', alpha=0.6,
                label=f'Best val AUC = {max(history["val_binary_auc"]):.3f}',
                zorder=2)
    ax1.set_ylim(0.89, 0.98)
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('AUC')
    ax1.set_title('Binary classification AUC', fontweight='bold')
    ax1.legend(fontsize=9, frameon=False)

    # loss plot
    ax2.plot(epochs, history['train_loss'], 'o-',
             color='#1D9E75', lw=2, ms=5, label='Train loss', zorder=3)
    ax2.plot(epochs, history['val_loss'], 's--',
             color='#D85A30', lw=2, ms=5, label='Val loss', zorder=3)
    ax2.set_xlabel('Epoch'); ax2.set_ylabel('Loss')
    ax2.set_title('Multi-task loss divergence', fontweight='bold')
    ax2.legend(fontsize=9, frameon=False)

    plt.tight_layout(pad=2)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()


def plot_subtype_accuracy(history, save_path=None):
    '''
    Function: plot_subtype_accuracy
    Purpose: Plots subtype classification accuracy per epoch as a grouped
             bar chart for training and validation sets.
    Inputs:
      history   - dict; training history with keys train_subtype_acc and
                  val_subtype_acc. Each value is a list of per-epoch values.
      save_path - str | None; if provided, saves the figure to this path
                  at 300 dpi. Defaults to None (display only).
    Returns:
      N/A
    Behaviour:
      Draws grouped bars for train and val subtype accuracy at each epoch.
      Adds a dashed reference line at 12.5% marking random chance for an
      8-class problem. Removes top and right spines. Moves the legend
      outside the plot to avoid overlap with the bars.
    '''
    epochs = np.arange(len(history['train_subtype_acc']))

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.set_facecolor('white')
    ax.grid(True, color='#E8E6E0', linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)

    w = 0.38
    ax.bar(epochs - w/2, history['train_subtype_acc'], w,
           color='#1D9E75', alpha=0.85, label='Train', zorder=3)
    ax.bar(epochs + w/2, history['val_subtype_acc'], w,
           color='#D85A30', alpha=0.85, label='Val', zorder=3)
    ax.axhline(0.125, color='#888780', lw=1, ls='--', alpha=0.6,
               label='Random (8-class = 12.5%)', zorder=2)

    ax.set_xticks(epochs)
    ax.set_xticklabels(np.arange(1, len(epochs) + 1))
    ax.set_ylim(0, 0.85)
    ax.set_xlabel('Epoch'); ax.set_ylabel('Subtype accuracy')
    ax.set_title('Subtype classification: train vs. val generalization gap',
                 fontweight='bold')
    ax.legend(fontsize=9, frameon=False, loc='upper left',
              bbox_to_anchor=(1, 1))
    ax.spines[['top', 'right']].set_visible(False)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()


def plot_figures(history, cm_img, cm_pat,
                 save_cm='vit_cm.png',
                 save_curves='vit_curves.png',
                 save_subtype='vit_subtype.png'):
    '''
    Function: plot_figures
    Purpose: Generates and saves all three result figures in one call.
    Inputs:
      history     - dict; training history with keys train_binary_auc,
                    val_binary_auc, train_loss, val_loss,
                    train_subtype_acc, val_subtype_acc.
      cm_img      - np.ndarray [2, 2]; image-level confusion matrix.
      cm_pat      - np.ndarray [2, 2]; patient-level confusion matrix.
      save_cm     - str; save path for confusion matrix figure.
                    Defaults to vit_cm.png.
      save_curves - str; save path for training curves figure.
                    Defaults to vit_curves.png.
      save_subtype- str; save path for subtype accuracy figure.
                    Defaults to vit_subtype.png.
    Returns:
      N/A
    Behaviour:
      Calls plot_confusion_matrix twice (image and patient level),
      plot_training_curves, and plot_subtype_accuracy with the
      provided save paths.
    '''
    # side by side confusion matrices
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))

    from matplotlib.colors import Normalize
    import matplotlib.patches as patches

    cmap = plt.get_cmap('Blues')
    labels = ['Benign', 'Malignant']

    for ax, cm, title in zip(
        axes,
        [cm_img, cm_pat],
        ['Image-level confusion matrix\n(epoch 5, val AUC = 0.932)',
         'Patient-level confusion matrix\n(majority vote, 17 subjects)']
    ):
        norm = Normalize(vmin=0, vmax=cm.max())
        ax.set_facecolor('white')
        ax.grid(False)
        for i in range(2):
            for j in range(2):
                rect = patches.Rectangle(
                    [j, 1-i], 1, 1,
                    linewidth=0, edgecolor='none',
                    facecolor=cmap(norm(cm[i, j]))
                )
                ax.add_patch(rect)
                tc = 'white' if cm[i, j] > cm.max() * 0.55 else '#1a1a1a'
                ax.text(j+0.5, 1-i+0.5, str(cm[i, j]),
                        ha='center', va='center',
                        fontsize=16, fontweight='bold', color=tc)
        ax.set_xlim(0, 2); ax.set_ylim(0, 2)
        ax.set_xticks([0.5, 1.5]); ax.set_xticklabels(labels, fontsize=10)
        ax.set_yticks([0.5, 1.5]); ax.set_yticklabels(labels[::-1],
                                                        fontsize=10,
                                                        rotation=90,
                                                        va='center')
        ax.set_xlabel('Predicted label', fontsize=10)
        ax.set_ylabel('True label', fontsize=10)
        ax.set_title(title, fontsize=11, fontweight='bold', pad=10)
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)

    plt.tight_layout(pad=2)
    plt.savefig(save_cm, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()

    plot_training_curves(history, save_path=save_curves)
    plot_subtype_accuracy(history, save_path=save_subtype)
