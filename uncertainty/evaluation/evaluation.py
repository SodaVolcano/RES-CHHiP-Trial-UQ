"""
Evaluate prediction(s) against ground truth label(s) using a list of metrics.
"""

from typing import Callable, Literal

import toolz as tz
import torch
from toolz import curried

from ..metrics import get_classification_metric
from ..metrics.uncertainties import get_uncertainty_metric, probability_map
from ..utils import curry


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
        return get_classification_metric(name) or (
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

    Note that for uncertainty metrics, the `label` is not used. The `aggregate_before_eval`
    parameter does not affect uncertainty evaluation, i.e. the predictions are NOT aggregated
    before evaluating uncertainty metrics but rather uncertainty is evaluated using all
    predictions.

    Parameters
    ----------
    predictions : torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor]
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

    def _get_metric(
        name: str,
    ) -> Callable[
        [torch.Tensor, torch.Tensor, Literal["micro", "macro", "none"]], torch.Tensor
    ]:
        """Return (wrapped) spatial metric or uncertainty metric function"""
        spatial_metric = get_classification_metric(name)
        if spatial_metric is None:
            # change uncertainty function signature to take in pred and label (not used)
            return lambda pred, _, avg: get_uncertainty_metric(f"{name}")(pred, average=avg)  # type: ignore

        if aggregate_before_eval:
            return lambda preds, label, avg: tz.pipe(
                preds, probability_map, spatial_metric(label=label, average=avg)
            )

        return lambda preds, label, avg: tz.pipe(
            preds,
            curried.map(spatial_metric(label=label, average=avg)),
            list,
            torch.vstack,
            curry(torch.mean)(dim=0),
        )

    if not isinstance(predictions, torch.Tensor):
        predictions = torch.stack(predictions)

    return tz.pipe(
        metric_names,
        curried.map(lambda name: _get_metric(name)(predictions, label, average)),
        list,
        torch.tensor if average != "none" else torch.cat,
    )  # type: ignore
