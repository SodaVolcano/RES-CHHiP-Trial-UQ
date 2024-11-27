"""
Evaluate prediction(s) against ground truth label(s) using a list of metrics.
"""

from typing import Literal

import toolz as tz
import torch
from toolz import curried

from ..utils import curry
from .metrics import get_metric_func
from .uncertainties import get_uncertainty_metric, probability_map


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
        - "sen": Sensitivity (AKA recall)
        - "precision": Precision
        - "ppv": Positive predictive value (AKA precision)

    average : Literal["micro", "macro", "none"]
        Averaging method for the metrics.
        - "micro": Calculate metrics globally across all classes.
        - "macro": Calculate metrics for each class and average them.
        - "none": Return the metrics for each class separately.

    Returns
    -------
    torch.Tensor
        1D concatenated metrics for the prediction in the order of the metric names.
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
    predictions: torch.Tensor,
    label: torch.Tensor,
    metric_names: list[str],
    average: Literal["micro", "macro", "none"] = "macro",
    aggregate_before_eval: bool = True,
) -> torch.Tensor:
    """
    Evaluate a list of `predictions` against a `label` tensor using a list of metrics

    Mean of each prediction is produced as the final prediction metric.

    Parameters
    ----------
    predictions : torch.Tensor
        N predictions of shape (N, C, ...)
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
    aggregate_before_eval : bool
        Whether to aggregate N predictions into a single, averaged prediction
        before evaluating the metrics or to evaluate each prediction separately
        and then average the metrics afterwards.

    Returns
    -------
    torch.Tensor
        A tensor of shape `(num_metrics,)` with the average metrics across all predictions
    """
    if aggregate_before_eval:
        return tz.pipe(
            predictions,
            probability_map,
            evaluate_prediction(
                label=label, metric_names=metric_names, average=average
            ),
        )
    return tz.pipe(
        predictions,
        curried.map(
            evaluate_prediction(label=label, metric_names=metric_names, average=average)
        ),
    )


def compute_uncertainty():
    """
    - "mean_variance": Mean of the variance between all N predictions
    - "mean_entropy": Mean of the entropy between all N predictions
    - "pairwise_dice": Mean pairwise Dice between all N predictions
    - "pairwise_surface_dice_<float>": Mean pairwise surface Dice at tolerance <float>
      between all N predictions

    """
    pass
    # uncertainty_names = ["mean_variance", "mean_entropy", "pairwise_dice"]
    # spatial_metrics = list(
    #     filter(
    #         lambda x: x not in uncertainty_names
    #         and not x.startswith("pairwise_surface_dice"),
    #         metric_names,
    #     )
    # )
    # uncertainty_metrics = list(
    #     filter(
    #         lambda x: x in uncertainty_names or x.startswith("pairwise_surface_dice"),
    #         metric_names,
    #     )
    # )

    # uncertainty_scores = tz.pipe(
    #     uncertainty_metrics,
    #     curried.map(
    #         lambda name: get_uncertainty_metric(name)(predictions, average=average)
    #     ),
    #     # some tensors have no shape, add a shape to it
    #     curried.map(lambda score: score.unsqueeze(0) if score.shape == () else score),
    #     list,
    # )
