import numpy as np
import matplotlib.pyplot as plt
from math import ceil
import textwrap

def plot_multilabel_confusion_matrix(
    cm: np.ndarray,
    class_names=None,
    ncols: int = 3,
    normalize: bool = False,
    cmap = plt.cm.Blues,
    suptitle: str | None = None,
    wrap_width: int = 14,
):
    assert cm.ndim == 3 and cm.shape[1:] == (2, 2), "cm must be (C, 2, 2)"
    C = cm.shape[0]
    if class_names is None:
        class_names = [f"class_{i}" for i in range(C)]

    data = cm.astype(float).copy()
    if normalize:
        rowsums = data.sum(axis=2, keepdims=False)
        rowsums[rowsums == 0] = 1.0
        for i in range(C):
            data[i, 0, :] /= rowsums[i, 0]
            data[i, 1, :] /= rowsums[i, 1]

    nrows = ceil(C / ncols)

    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols,
        figsize=(ncols * 4.6, nrows * 4.6),
        constrained_layout=True
    )
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    axes = axes.flatten()

    vmin = 0.0
    vmax = 1.0 if normalize else float(cm.max() if cm.size else 1.0)

    for i in range(C):
        ax = axes[i]
        ax.imshow(data[i], cmap=cmap, vmin=vmin, vmax=vmax)

        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Negative", "Positive"], fontsize=9)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["Negative", "Positive"], fontsize=9)

        title_txt = textwrap.fill(class_names[i].replace("_", " "), width=wrap_width)
        ax.set_title(
            title_txt,
            pad=12,
            fontsize=11,
            bbox=dict(facecolor="white", alpha=0.9, edgecolor="none", boxstyle="round,pad=0.2")
        )

        annot = cm[i] if not normalize else data[i]
        thresh = (vmax - vmin) / 2.0
        for r in range(2):
            for c in range(2):
                val = annot[r, c]
                txt = f"{int(val)}" if not normalize else f"{val:.2f}"
                ax.text(
                    c, r, txt,
                    ha="center", va="center",
                    color="white" if data[i, r, c] > thresh else "black",
                    fontsize=11
                )

        ax.set_xticks(np.arange(-.5, 2, 1), minor=True)
        ax.set_yticks(np.arange(-.5, 2, 1), minor=True)
        ax.grid(which="minor", linestyle="-", linewidth=0.5, alpha=0.4)
        ax.tick_params(axis='both', which='both', length=0)

    for j in range(C, len(axes)):
        axes[j].axis("off")

    fig.supxlabel("Predicted", fontsize=12)
    fig.supylabel("True", fontsize=12)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes[:C], fraction=0.02, pad=0.02)
    cbar.set_label("Proportion" if normalize else "Count")

    if suptitle:
        fig.suptitle(suptitle, fontsize=14)

    plt.show()
    return fig, axes
