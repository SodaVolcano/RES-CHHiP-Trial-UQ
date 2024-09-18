from matplotlib.figure import Figure
import torch
import matplotlib.pyplot as plt
from .uncertainties import probability_map, entropy_map, variance_map


def display_slices_grid(
    x: torch.Tensor,
    y: torch.Tensor,
    y_pred: torch.Tensor,
    fig_name: str | None = None,
    n_slices: int = 10,
) -> Figure:
    """
    Return a grid of slices from the CT scan, the ground truth and the prediction.

    Parameters
    ----------
    x : torch.Tensor
        The CT scan of shape (channel, H, W, D)
    y : torch.Tensor
        The ground truth of shape (channel, H, W, D)
    y_pred : torch.Tensor
        The prediction of shape (channel, H, W, D)
    fig_name : str | None
        If provided, the figure will be saved to this path
    n_slices : int
        Number of slices to display, slices will be evenly sampled from
        the list of all slices
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
        axs[i, 0].set_title(f"CT scan, slice {idx}")
        axs[i, 0].imshow(x[:, :, idx], cmap="gray")
        axs[i, 0].axis("off")

        axs[i, 1].set_title(f"Ground truth, slice {idx}")
        axs[i, 1].imshow(y[:, :, idx], cmap="gray")
        axs[i, 1].axis("off")

        axs[i, 2].set_title(f"Prediction, slice {idx}")
        axs[i, 2].imshow(y_pred[:, :, idx], cmap="gray")
        axs[i, 2].axis("off")

        # Overlays
        axs[i, 3].set_title(f"Overlaying GT on scan, slice {idx}")
        axs[i, 3].imshow(x[:, :, idx], cmap="gray")
        axs[i, 3].imshow(y[:, :, idx], cmap="hot", alpha=0.5)
        axs[i, 3].axis("off")

        axs[i, 4].set_title(f"Overlay prediction on scan, slice {idx}")
        axs[i, 4].imshow(x[:, :, idx], cmap="gray")
        axs[i, 4].imshow(y_pred[:, :, idx], cmap="hot", alpha=0.5)
        axs[i, 4].axis("off")

        axs[i, 5].set_title(f"Overlay GT and prediction on scan, slice {idx}")
        axs[i, 5].imshow(x[:, :, idx], cmap="gray")
        axs[i, 5].imshow(y[:, :, idx], cmap="hot", alpha=0.3)
        axs[i, 5].imshow(y_pred[:, :, idx], cmap="cool", alpha=0.3)
        axs[i, 5].axis("off")

    fig.tight_layout()
    if fig_name is not None:
        fig.savefig(fig_name)
    return fig


def display_uncertainty_maps(
    x: torch.Tensor,
    y: torch.Tensor,
    y_preds: torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor],
    fig_name: str | None = None,
    n_slices: int = 10,
):
    """
    Display a grid with `x`, `y`, and probability, entropy, and variance maps of `y_preds`.

    If the input have multiple channels, the maps will be computed for each channel and summed
    into a single channel.

    Parameters
    ----------
    x : torch.Tensor
        The CT scan of shape (channel, H, W, D)
    y : torch.Tensor
        The ground truth of shape (channel, H, W, D)
    y_preds : torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor]
        The predictions of shape (N, channel, H, W, D) for N prediction samples or
        a list/tuple of N predictions
    fig_name : str | None
        If provided, the figure will be saved to this path
    n_slices : int
        Number of slices to display, slices will be evenly sampled from
        the list of all slices
    """
    _, _, _, D = x.shape
    slice_indices = [int(i * D / 10) for i in range(n_slices)]

    # Generate the uncertainty maps
    probability = probability_map(y_preds).sum(dim=0)  # Shape: (channel, H, W, D)
    entropy = entropy_map(y_preds).sum(dim=0)  # Shape: (channel, H, W, D)
    variance = variance_map(y_preds).sum(dim=0)  # Shape: (channel, H, W, D)

    # Create the figure with 5 columns: CT scan, ground truth, probability, entropy, variance
    fig, axs = plt.subplots(n_slices, 5, figsize=(30, 5 * n_slices))

    x = torch.moveaxis(x, 0, -1)  # Move channels to the end for plotting
    y = torch.moveaxis(y, 0, -1)
    var_min, var_max = variance.min(), variance.max()
    ent_min, ent_max = entropy.min(), entropy.max()

    for i, idx in enumerate(slice_indices):
        # Column 0: CT scan
        axs[i, 0].set_title(f"CT scan, slice {idx}")
        axs[i, 0].imshow(x[:, :, idx], cmap="gray")
        axs[i, 0].axis("off")

        # Column 1: Ground truth
        axs[i, 1].set_title(f"Ground truth, slice {idx}")
        axs[i, 1].imshow(y[:, :, idx], cmap="gray")
        axs[i, 1].axis("off")

        # Column 2: Probability map
        axs[i, 2].set_title(f"Probability, slice {idx}")
        fig.colorbar(axs[i, 2].imshow(probability[:, :, idx], cmap="jet"), ax=axs[i, 2])
        axs[i, 2].imshow(probability[:, :, idx], cmap="jet", vmin=0, vmax=1)
        axs[i, 2].axis("off")

        # Column 3: Entropy map
        axs[i, 3].set_title(f"Entropy, slice {idx}")
        axs[i, 3].imshow(entropy[:, :, idx], cmap="inferno", vmin=ent_min, vmax=ent_max)
        fig.colorbar(
            axs[i, 3].imshow(
                entropy[:, :, idx], cmap="inferno", vmin=ent_min, vmax=ent_max
            ),
            ax=axs[i, 3],
        )
        axs[i, 3].axis("off")

        # Column 4: Variance map
        axs[i, 4].set_title(f"Variance, slice {idx}")
        axs[i, 4].imshow(
            variance[:, :, idx], cmap="inferno", vmin=var_min, vmax=var_max
        )
        fig.colorbar(
            axs[i, 4].imshow(
                variance[:, :, idx], cmap="inferno", vmin=var_min, vmax=var_max
            ),
            ax=axs[i, 4],
        )
        axs[i, 4].axis("off")
        axs[i, 4].axis("off")

    fig.tight_layout()

    if fig_name is not None:
        fig.savefig(fig_name)

    return fig
