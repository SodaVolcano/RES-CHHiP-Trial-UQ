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

    Note that for uncertainty metrics, the `label` is not used. Uncertainty
    such as variance and entropy are calculated pixelwise by interpreting the
    prediction as a probability map.

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
        - "mean_variance": Mean pixelwise variance of the prediction (label not used)
        - "mean_entropy": Mean pixelwise entropy of the prediction (label not used)

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

    def _get_metric(name: str):
        return get_metric_func(name) or (
            # change uncertainty function signature to take in pred and label (not used)
            lambda pred, _, average: get_uncertainty_metric(f"{name}_pixelwise")(pred, average=average)  # type: ignore
        )

    return tz.pipe(
        metric_names,
        curried.map(_get_metric),
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
    aggregate_before_eval: bool = True,
) -> torch.Tensor:
    """
    Evaluate a list of `predictions` using a list of metrics against a single `label`

    If `aggregate_before_eval` is True, the predictions are aggregated into a single
    prediction before evaluating the metrics. If False, the metrics are evaluated
    for each prediction separately and then averaged.

    Note that for uncertainty metrics, the `label` is not used.

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
        - "mean_variance": Mean variance between all N predictions (label not used)
        - "mean_entropy": Mean entropy between all N predictions (label not used)
        - "pairwise_dice": Mean pairwise Dice between all N predictions (label not used)
        - "pairwise_surface_dice_<float>": Mean pairwise surface Dice at tolerance <float>
          between all N predictions (label not used)
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
    if not isinstance(predictions, torch.Tensor):
        predictions = torch.stack(predictions)

    if aggregate_before_eval:
        return tz.pipe(
            predictions,
            probability_map,
            evaluate_prediction(
                label=label, metric_names=metric_names, average=average
            ),
        )
    # Evaluate each prediction separately and then average the metrics
    return tz.pipe(
        predictions,
        curried.map(
            evaluate_prediction(label=label, metric_names=metric_names, average=average)
        ),
        list,
        torch.vstack,
        curry(torch.mean)(dim=0),
    )


def compute_uncertainty():
    """ """
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
