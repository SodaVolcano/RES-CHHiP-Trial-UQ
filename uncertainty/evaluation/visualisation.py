import torch
import numpy as np
import matplotlib.pyplot as plt


def display_slices_grid(
    x: torch.Tensor,
    y: torch.Tensor,
    y_pred: torch.Tensor,
    fig_name: str,
    n_slices: int = 10,
):
    """
    Save a grid of slices from the CT scan, the ground truth and the prediction.

    Parameters
    ----------
    x: torch.Tensor
        The CT scan of shape (channel, H, W, D)
    y: torch.Tensor
        The ground truth of shape (channel, H, W, D)
    y_pred: torch.Tensor
        The prediction of shape (channel, H, W, D)
    """
    assert (
        x.shape[1:] == y.shape[1:]
    ), "x and y must have the same shape (channel, H, W, D)"
    assert (
        y.shape[1:] == y_pred.shape[1:]
    ), "y and y_pred must have the same H, W dimensions"

    _, _, _, D = x.shape
    slice_indices = [int(i * D / 10) for i in range(n_slices)]

    fig, axs = plt.subplots(n_slices, 6, figsize=(30, 5 * n_slices))

    x = torch.moveaxis(x, 0, -1)
    y = torch.moveaxis(y, 0, -1)
    y_pred = torch.moveaxis(y_pred, 0, -1)

    for i, idx in enumerate(slice_indices):
        # Original images
        axs[i, 0].imshow(x[:, :, idx], cmap="gray")
        axs[i, 0].set_title(f"CT scan, slice {idx}")
        axs[i, 0].axis("off")

        axs[i, 1].imshow(y[:, :, idx], cmap="gray")
        axs[i, 1].set_title(f"Ground truth, slice {idx}")
        axs[i, 1].axis("off")

        axs[i, 2].imshow(y_pred[:, :, idx], cmap="gray")
        axs[i, 2].set_title(f"Prediction, slice {idx}")
        axs[i, 2].axis("off")

        # Overlays
        axs[i, 3].imshow(x[:, :, idx], cmap="gray")
        axs[i, 3].imshow(y[:, :, idx], cmap="hot", alpha=0.5)
        axs[i, 3].set_title(f"Overlaying GT on scan, slice {idx}")
        axs[i, 3].axis("off")

        axs[i, 4].imshow(x[:, :, idx], cmap="gray")
        axs[i, 4].imshow(y_pred[:, :, idx], cmap="hot", alpha=0.5)
        axs[i, 4].set_title(f"Overlay prediction on scan, slice {idx}")
        axs[i, 4].axis("off")

        axs[i, 5].imshow(x[:, :, idx], cmap="gray")
        axs[i, 5].imshow(y[:, :, idx], cmap="hot", alpha=0.3)
        axs[i, 5].imshow(y_pred[:, :, idx], cmap="cool", alpha=0.3)
        axs[i, 5].set_title(f"Overlay GT and prediction on scan, slice {idx}")
        axs[i, 5].axis("off")

    fig.tight_layout()
    fig.savefig(fig_name)
