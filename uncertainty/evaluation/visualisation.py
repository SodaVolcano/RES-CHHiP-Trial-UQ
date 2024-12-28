"""
Functions to display slices of CT scans, ground truth, etc...
"""

from typing import Callable

import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns
import toolz as tz
import torch
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from toolz import curried

from ..metrics import entropy_map, probability_map, variance_map
from ..utils import star, transform_nth


def assert_dfs(dfs: list[pl.LazyFrame | pl.DataFrame], df_names: list[str]) -> None:
    assert all(
        set(dfs[0].columns) == set(df.columns) for df in dfs
    ), "ALl Dataframes must have the same columns"
    assert (n_dfs := len(dfs)) == (
        n_names := len(df_names)
    ), f"Specified {n_names} dataframe names but got {n_dfs} dataframes"


def box_plot(
    dfs: list[pl.LazyFrame | pl.DataFrame],
    df_names: list[str],
    x_label: str,
    y_label: str,
    col_group_title: str,
    col_name_transform: Callable[[str], str],
    col_filter: str | list[str],
    figsize: tuple[int, int] = (10, 10),
    gap: float = 0.2,
) -> tuple[Figure, Axes]:
    """
    Show a box plot of the data in `dfs` grouped by `col_group_title`.

    Each dataframe in `dfs` will be displayed as separate clusters of boxes
    in the plot. `col_filter` is used to select specific columns from the
    dataframes in `dfs`, and each of the remaining columns will be displayed
    as a single box as a part of the cluster for a dataframe.

    Parameters
    ----------
    dfs : list[pl.LazyFrame | pl.DataFrame]
        List of dataframes with the same columns to plot. Each dataframe is
        displayed as a separate cluster of boxes in the plot with cluster
        names given by `df_names`. A cluster consists of boxes for each
        column in a single dataframe (which is selected using `col_filter`).
    x_label : str
        Label for the x-axis of the plot
    y_label : str
        Label for the y-axis of the plot
    df_names : list[str]
        List of names corresponding to each dataframe in `dfs`; ticks to display on the x-axis.
    col_group_title : str
        Overall title for the column groups in the plot.
    col_name_transform : Callable[[str], str]
        Function to transform the column names of the dataframes.
    col_filter : str | list[str]
        Regex pattern or a list of column names to filter the columns of the
        dataframes in `dfs`. The remaining columns will be displayed as
        separate boxes in the plot for each cluster. If Regex, it must
        begin with "^" and end with "$".
    figsize : tuple[int, int]
        Size of the figure in inches. Change this to adjust the size of the
        boxes.
    gap : float
        The gap between the boxes in the plot.

    Examples (NOTE: only abstract representation of the output graph)
    --------
    >>> import polars as pl
    >>> dfs = [
    ...     pl.DataFrame({'a1': [1, 2], 'a2': [2, 1], 'b1': [2, 1], 'b2': [1, 1]}),
    ...     pl.DataFrame({'a1': [1, 2], 'a2': [1, 2], 'b1': [1, 2], 'b2': [1, 2]}),
    ...     pl.DataFrame({'a1': [2, 2], 'a2': [2, 1], 'b1': [1, 2], 'b2': [2, 1]}),
    ... ]
    >>> col_transform = lambda name: f"{name[0]}_{name[-1]}"
    ... f, ax = box_plot(
    ...     dfs, 'X', 'Y', ['C1', 'C2'], "Groups", col_transform, r"^a.*$", (5, 5)
    ... )
    >>> f.show()
          ^
      2.0 |   ┬     ┬              ┬     ┬
          |   |     |              |     |
      1.8 | ┌─┴─┐ ┌─┴─┐          ┌─┴─┐ ┌─┴─┐
          | │   │ |   │          │   │ │   │
      1.6 | |   │ |   │          │   │ │   │
    Y     | ├───┤ ├───┤          ├───┤ ├───┤
      1.4 | │   | │   │          │   | │   │
          | │   │ │   │          │   │ │   │
      1.2 | └───┘ └───┘  Groups  └───┘ └───┘
          |   |     |    -  a_1    |     |
      1.0 |   ┴     ┴    -  a_2    ┴     ┴
          +------------------------------------>
                C1                 C2
                          X
    """
    assert_dfs(dfs, df_names)

    combined = tz.pipe(
        zip(df_names, dfs),
        curried.map(transform_nth(1, lambda df: df.lazy())),
        curried.map(transform_nth(1, lambda df: df.select(pl.col(col_filter)))),
        curried.map(
            star(
                lambda name, lf: lf.unpivot(
                    variable_name=col_group_title, value_name=y_label
                ).with_columns(pl.lit(name).alias(x_label))
            )
        ),
        pl.concat,
        lambda lf: lf.with_columns(
            pl.col(col_group_title).map_elements(
                col_name_transform, return_dtype=pl.String
            )
        ),
        lambda lf: lf.collect().to_pandas(),
    )

    f = plt.figure(figsize=figsize)
    ax = f.add_subplot(111)

    sns.stripplot(
        x=x_label,
        y=y_label,
        hue=col_group_title,
        data=combined,  # type: ignore
        palette="pastel",
        dodge=True,
        marker="o",
        alpha=0.3,
        size=4,
        ax=ax,
    )
    sns.boxplot(
        x=x_label,
        y=y_label,
        hue=col_group_title,
        data=combined,  # type: ignore
        palette="dark",
        fliersize=0,
        fill=False,
        linewidth=1,
        width=0.8,
        dodge=True,
        ax=ax,
        gap=gap,
    )

    n_handles = len(dfs[0].select(pl.col(col_filter)).columns)
    handles, labels = ax.get_legend_handles_labels()
    for handle in handles[:n_handles]:
        handle.set_alpha(1)
    ax.legend(handles[:n_handles], labels[:n_handles], title=col_group_title)
    f.tight_layout()
    return f, ax


def plot_surface_dices(
    dfs: list[pl.LazyFrame | pl.DataFrame],
    df_names: list[str],
    y_label: str = "Surface Dice",
    x_label: str = "Tolerance (mm)",
    figsize: tuple[int, int] = (10, 10),
    max_tolerance: float | None = None,
    re_pattern: str = r"^(\w+)_surface_dice_([\d\.]+)$",
) -> tuple[Figure, Axes]:
    """
    Plot the surface dices from the dataframes in `dfs` across increasing thresholds.

    Parameters
    ----------
    dfs : list[pl.LazyFrame | pl.DataFrame]
        List of dataframes with columns of surface dice scores. Each organ and dataframe
        in `dfs` is displayed as a separate colour and line style in the plot respectively.
    df_names : list[str]
        List of names corresponding to each dataframe in `dfs`.
    y_label : str
        Label for the y-axis of the plot.
    x_label : str
        Label for the x-axis of the plot.
    figsize : tuple[int, int]
        Size of the figure in inches.
    max_tolerance : float | None
        Maximum tolerance to consider for the plot (inclusive). If None, all tolerances are considered.
    re_pattern : str
        Regex pattern to identify columns in the dataframe as surface dices. The pattern must
        have two groups: the first group must capture the organ name and the second group must
        capture the tolerance. The default pattern assumes the column names are of the form
        "<organ>_surface_dice_<tolerance>".
    """
    assert_dfs(dfs, df_names)

    surface_dices = tz.pipe(
        zip(df_names, dfs),
        curried.map(transform_nth(1, lambda df: df.lazy())),
        curried.map(transform_nth(1, lambda lf: lf.select(pl.col(re_pattern)))),
        curried.map(
            star(lambda name, lf: lf.with_columns(pl.lit(name).alias("Method")))
        ),
        pl.concat,
        lambda stacked_lf: stacked_lf.group_by("Method").mean(),
        lambda mean_lf: mean_lf.unpivot(
            on=pl.col(re_pattern),
            variable_name="organ_tolerance",
            value_name="surface_dice",
            index="Method",
        ),
        lambda lf: lf.with_columns(
            [
                pl.col("organ_tolerance").str.extract(re_pattern, 1).alias("Organ"),
                pl.col("organ_tolerance")
                .str.extract(re_pattern, 2)
                .cast(pl.Float64)
                .alias("Tolerance"),
            ]
        ),
        lambda lf: lf.sort(["Method", "Organ", "Tolerance"]),
        (
            (lambda lf: lf.filter(pl.col("Tolerance") <= max_tolerance))
            if max_tolerance
            else tz.identity
        ),
        lambda lf: lf.collect().to_pandas(),
    )

    f = plt.figure(figsize=figsize)
    ax = f.add_subplot(111)
    sns.lineplot(
        data=surface_dices,
        x="Tolerance",
        y="surface_dice",
        hue="Organ",
        style="Method",
        markers=True,
    )
    f.tight_layout()
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.legend()
    ax.grid()
    return f, ax


def plot_aurc(
    df: pl.DataFrame | pl.LazyFrame,
):
    """
    df: hhave eval + confidence
    """


# def display_slices_grid(
#     x: torch.Tensor,
#     y: torch.Tensor,
#     y_pred: torch.Tensor,
#     fig_name: str | None = None,
#     n_slices: int = 10,
# ) -> Figure:
#     """
#     Return a grid of slices from the CT scan, the ground truth and the prediction.

#     Parameters
#     ----------
#     x : torch.Tensor
#         The CT scan of shape (channel, H, W, D)
#     y : torch.Tensor
#         The ground truth of shape (channel, H, W, D)
#     y_pred : torch.Tensor
#         The prediction of shape (channel, H, W, D)
#     fig_name : str | None
#         If provided, the figure will be saved to this path
#     n_slices : int
#         Number of slices to display, slices will be evenly sampled from
#         the list of all slices
#     """
#     assert (
#         x.shape[1:] == y.shape[1:]
#     ), "x and y must have the same shape (channel, H, W, D)"
#     assert (
#         y.shape[1:] == y_pred.shape[1:]
#     ), "y and y_pred must have the same H, W dimensions"

#     _, _, _, D = x.shape
#     slice_indices = [int(i * D / 10) for i in range(n_slices)]

#     fig, axs = plt.subplots(n_slices, 6, figsize=(30, 5 * n_slices))

#     x = torch.moveaxis(x, 0, -1)
#     y = torch.moveaxis(y, 0, -1)
#     y_pred = torch.moveaxis(y_pred, 0, -1)

#     for i, idx in enumerate(slice_indices):
#         axs[i, 0].set_title(f"CT scan, slice {idx}")
#         axs[i, 0].imshow(x[:, :, idx], cmap="gray")
#         axs[i, 0].axis("off")

#         axs[i, 1].set_title(f"Ground truth, slice {idx}")
#         axs[i, 1].imshow(y[:, :, idx], cmap="gray")
#         axs[i, 1].axis("off")

#         axs[i, 2].set_title(f"Prediction, slice {idx}")
#         axs[i, 2].imshow(y_pred[:, :, idx], cmap="gray")
#         axs[i, 2].axis("off")

#         # Overlays
#         axs[i, 3].set_title(f"Overlaying GT on scan, slice {idx}")
#         axs[i, 3].imshow(x[:, :, idx], cmap="gray")
#         axs[i, 3].imshow(y[:, :, idx], cmap="hot", alpha=0.5)
#         axs[i, 3].axis("off")

#         axs[i, 4].set_title(f"Overlay prediction on scan, slice {idx}")
#         axs[i, 4].imshow(x[:, :, idx], cmap="gray")
#         axs[i, 4].imshow(y_pred[:, :, idx], cmap="hot", alpha=0.5)
#         axs[i, 4].axis("off")

#         axs[i, 5].set_title(f"Overlay GT and prediction on scan, slice {idx}")
#         axs[i, 5].imshow(x[:, :, idx], cmap="gray")
#         axs[i, 5].imshow(y[:, :, idx], cmap="hot", alpha=0.3)
#         axs[i, 5].imshow(y_pred[:, :, idx], cmap="cool", alpha=0.3)
#         axs[i, 5].axis("off")

#     fig.tight_layout()
#     if fig_name is not None:
#         fig.savefig(fig_name)
#     return fig


# def display_uncertainty_maps(
#     x: torch.Tensor,
#     y: torch.Tensor,
#     y_preds: torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor],
#     fig_name: str | None = None,
#     n_slices: int = 10,
# ):
#     """
#     Display a grid with `x`, `y`, and probability, entropy, and variance maps of `y_preds`.

#     If the input have multiple channels, the maps will be computed for each channel and summed
#     into a single channel.

#     Parameters
#     ----------
#     x : torch.Tensor
#         The CT scan of shape (channel, H, W, D)
#     y : torch.Tensor
#         The ground truth of shape (channel, H, W, D)
#     y_preds : torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor]
#         The predictions of shape (N, channel, H, W, D) for N prediction samples or
#         a list/tuple of N predictions
#     fig_name : str | None
#         If provided, the figure will be saved to this path
#     n_slices : int
#         Number of slices to display, slices will be evenly sampled from
#         the list of all slices
#     """
#     _, _, _, D = x.shape
#     slice_indices = [int(i * D / 10) for i in range(n_slices)]

#     # Generate the uncertainty maps
#     probability = probability_map(y_preds).sum(dim=0)  # Shape: (channel, H, W, D)
#     entropy = entropy_map(y_preds).sum(dim=0)  # Shape: (channel, H, W, D)
#     variance = variance_map(y_preds).sum(dim=0)  # Shape: (channel, H, W, D)

#     # Create the figure with 5 columns: CT scan, ground truth, probability, entropy, variance
#     fig, axs = plt.subplots(n_slices, 5, figsize=(30, 5 * n_slices))

#     x = torch.moveaxis(x, 0, -1)  # Move channels to the end for plotting
#     y = torch.moveaxis(y, 0, -1)
#     var_min, var_max = variance.min(), variance.max()
#     ent_min, ent_max = entropy.min(), entropy.max()

#     for i, idx in enumerate(slice_indices):
#         # Column 0: CT scan
#         axs[i, 0].set_title(f"CT scan, slice {idx}")
#         axs[i, 0].imshow(x[:, :, idx], cmap="gray")
#         axs[i, 0].axis("off")

#         # Column 1: Ground truth
#         axs[i, 1].set_title(f"Ground truth, slice {idx}")
#         axs[i, 1].imshow(y[:, :, idx], cmap="gray")
#         axs[i, 1].axis("off")

#         # Column 2: Probability map
#         axs[i, 2].set_title(f"Probability, slice {idx}")
#         fig.colorbar(axs[i, 2].imshow(probability[:, :, idx], cmap="jet"), ax=axs[i, 2])
#         axs[i, 2].imshow(probability[:, :, idx], cmap="jet", vmin=0, vmax=1)
#         axs[i, 2].axis("off")

#         # Column 3: Entropy map
#         axs[i, 3].set_title(f"Entropy, slice {idx}")
#         axs[i, 3].imshow(entropy[:, :, idx], cmap="inferno", vmin=ent_min, vmax=ent_max)
#         fig.colorbar(
#             axs[i, 3].imshow(
#                 entropy[:, :, idx], cmap="inferno", vmin=ent_min, vmax=ent_max
#             ),
#             ax=axs[i, 3],
#         )
#         axs[i, 3].axis("off")

#         # Column 4: Variance map
#         axs[i, 4].set_title(f"Variance, slice {idx}")
#         axs[i, 4].imshow(
#             variance[:, :, idx], cmap="inferno", vmin=var_min, vmax=var_max
#         )
#         fig.colorbar(
#             axs[i, 4].imshow(
#                 variance[:, :, idx], cmap="inferno", vmin=var_min, vmax=var_max
#             ),
#             ax=axs[i, 4],
#         )
#         axs[i, 4].axis("off")
#         axs[i, 4].axis("off")

#     fig.tight_layout()

#     if fig_name is not None:
#         fig.savefig(fig_name)

#     return fig
