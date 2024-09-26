from typing import Literal
import torch
import toolz as tz
from toolz import curried

from ..utils.wrappers import curry
from .inference import (
    sliding_inference,
    tta_inference,
    ensemble_inference,
    mc_dropout_inference,
)
from .metrics import get_metric_func
from tqdm import tqdm
from .uncertainties import probability_map


@torch.no_grad()
@curry
def evaluate_prediction(
    prediction: torch.Tensor,
    label: torch.Tensor,
    metric_names: list[str],
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
        List of metric names to evaluate. Can be any of
        - "hd": Hausdorff distance
        - "hd95": Hausdorff distance at 95th percentile
        - "asd": Average surface distance
        - "assd": Average symmetric surface distance
        - "dice": Dice similarity coefficient
        - "surface_dice_<float>": Surface Dice similarity coefficient at tolerance <float>
        - "recall": Recall
        - "sen": Sensitivity
        - "precision": Precision
        - "ppv": Positive predictive value

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
        curried.map(get_metric_func),
        curried.map(lambda metric: metric(prediction, label, average=average)),
        list,
        torch.tensor if average != "none" else torch.cat,
    )  # type: ignore


@torch.no_grad()
@curry
def evaluate_predictions(
    predictions: torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor],
    label: torch.Tensor,
    metric_names: list[str],
    average: Literal["micro", "macro", "none"] = "macro",
    aggregate: bool = True,
) -> torch.Tensor:
    """
    Evaluate a list of `predictions` against a `label` tensor using a list of metrics

    Mean of each prediction is produced as the final prediction metric.

    Parameters
    ----------
    predictions : torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor]
        List of predictions of shape (N, C, ...)
    label : torch.Tensor
        The ground truth tensor of shape (C, ...)
    metric_names : list[Literal]
        List of metric names to evaluate. Can be any of
        - "hd": Hausdorff distance
        - "hd95": Hausdorff distance at 95th percentile
        - "asd": Average surface distance
        - "assd": Average symmetric surface distance
        - "dice": Dice similarity coefficient
        - "surface_dice_<float>": Surface Dice similarity coefficient at tolerance <float>
        - "recall": Recall
        - "sen": Sensitivity
        - "precision": Precision
        - "ppv": Positive predictive value

    average : Literal["micro", "macro", "none"]
        Averaging mode for the channel-wise metrics per prediction
        - "micro": Calculate metrics globally across all channels
        - "macro": Calculate metrics for each channel, and calculate their mean
        - "none": Return the metrics for each channel
    aggregate: bool
        If True, all predictions are aggregated into a single map before evaluation

    Returns
    -------
    torch.Tensor
        If `summarise` is True, a tensor of shape `(num_metrics,)` with the
        average metrics across all predictions, or a tensor of shape
        `(N, num_metrics)` if `summarise` is False
    """
    if aggregate:
        return tz.pipe(
            predictions,
            probability_map,
            evaluate_prediction(
                label=label, metric_names=metric_names, average=average
            ),
        )  # type: ignore

    return tz.pipe(
        predictions,
        curried.map(
            evaluate_prediction(label=label, metric_names=metric_names, average=average)
        ),
        list,
        torch.stack,
    )  # type: ignore


def get_inference_mode(mode: str):
    return {
        "single": sliding_inference,
        "tta": tta_inference,
        "ensemble": ensemble_inference,
        "mcdo": mc_dropout_inference,
    }[mode]
