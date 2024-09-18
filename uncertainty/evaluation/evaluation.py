from typing import Literal
import torch
import toolz as tz
from toolz import curried

from ..utils.wrappers import curry
from .metrics import _get_metric_func
from tqdm import tqdm


@curry
def evaluate_prediction(
    prediction: torch.Tensor,
    label: torch.Tensor,
    metric_names: list[
        Literal[
            "hd",
            "hd95",
            "asd",
            "assd",
            "dice",
            "recall",
            "sen",
            "precision",
            "ppv",
        ]
    ],
    average: Literal["micro", "macro", "none"] = "macro",
) -> torch.Tensor:
    """
    Evaluate a `prediction` against a `label` tensor using a list of metrics.

    Parameters
    ----------
    prediction : torch.Tensor
        The predicted tensor.
    label : torch.Tensor
        The ground truth tensor.
    metric_names : list[Literal]
        List of metric names to evaluate.
    average : Literal["micro", "macro", "none"]
        Averaging method for the metrics.
        - "micro": Calculate metrics globally across all classes.
        - "macro": Calculate metrics for each class and average them.
        - "none": Return the metrics for each class separately.

    Returns
    -------
    torch.Tensor
        Concatenated metrics for the prediction.
    """
    return tz.pipe(
        metric_names,
        curried.map(_get_metric_func),
        curried.map(lambda metric: metric(prediction, label, average=average)),
        list,
        torch.tensor if average != "none" else torch.cat,
    )  # type: ignore


def evaluate_predictions(
    predictions: torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor],
    label: torch.Tensor,
    metric_names: list[
        Literal[
            "hd",
            "hd95",
            "asd",
            "assd",
            "dice",
            "recall",
            "sen",
            "precision",
            "ppv",
        ]
    ],
    average: Literal["micro", "macro", "none"] = "macro",
    summarise: bool = True,
    prog_bar: bool = False,
) -> torch.Tensor:
    """
    Evaluate a list of `predictions` against a `label` tensor using a list of metrics

    Parameters
    ----------
    predictions : torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor]
        List of predictions of shape (N, C, ...)
    label : torch.Tensor
        The ground truth tensor of shape (C, ...)
    metric_names : list[Literal]
        List of metric names to evaluate
    average : Literal["micro", "macro", "none"]
        Averaging mode for the channel-wise metrics per prediction
        - "micro": Calculate metrics globally across all channels
        - "macro": Calculate metrics for each channel, and calculate their mean
        - "none": Return the metrics for each channel
    summarise : bool
        If True, the metrics are averaged across all predictions
    prog_bar : bool
        If True, display a progress bar

    Returns
    -------
    torch.Tensor
        If `summarise` is True, a tensor of shape `(num_metrics,)` with the
        average metrics across all predictions, or a tensor of shape
        `(N, num_metrics)` if `summarise` is False
    """

    return tz.pipe(
        predictions,
        curried.map(
            evaluate_prediction(label=label, metric_names=metric_names, average=average)
        ),
        tqdm if prog_bar else tz.identity,
        list,
        torch.stack,
        lambda x: torch.mean(x, dim=0) if summarise else x,
    )  # type: ignore
