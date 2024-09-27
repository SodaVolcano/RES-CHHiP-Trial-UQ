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
from .uncertainties import probability_map, mean_entropy, mean_variance, pairwise_dice


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
) -> torch.Tensor:
    """
    Evaluate a list of `predictions` against a `label` tensor using a list of metrics

    Mean of each prediction is produced as the final prediction metric.

    **WARNING**: "mean_variance", "mean_entropy" and "pairwise_dice", if specified,
    will always appear at the end of the output tensor!!

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
        - "mean_variance": Mean of the variance between all N predictions
        - "mean_entropy": Mean of the entropy between all N predictions
        - "pairwise_dice": Mean pairwise Dice between all N predictions

    average : Literal["micro", "macro", "none"]
        Averaging mode for the channel-wise metrics per prediction
        - "micro": Calculate metrics globally across all channels
        - "macro": Calculate metrics for each channel, and calculate their mean
        - "none": Return the metrics for each channel

    Returns
    -------
    torch.Tensor
        A tensor of shape `(num_metrics,)` with the average metrics across all predictions
    """
    if not isinstance(predictions, torch.Tensor):
        predictions = torch.stack(predictions)
    uncertainties = {
            "mean_variance": lambda preds, average: mean_variance(preds, average),
            "mean_entropy": lambda preds, average: mean_entropy(preds, average),
            "pairwise_dice": lambda preds, average: pairwise_dice(preds > 0.5, average=average)
    }
    spatial_metrics = list(filter(lambda x: x not in uncertainties.keys(), metric_names))
    uncertainty_metrics = list(filter(lambda x: x in uncertainties.keys(), metric_names))

    uncertainty_scores = tz.pipe(
        uncertainty_metrics,
        curried.map(lambda name: uncertainties[name](predictions, average=average)),
        # some tensors have no shape, add a shape to it
        curried.map(lambda score: score.unsqueeze(0) if score.shape == () else score),
        list
    )
 
    return tz.pipe(
        predictions,
        probability_map,
        evaluate_prediction(
            label=label, metric_names=spatial_metrics, average=average
        ),
        lambda results: torch.cat([results] + uncertainty_scores)
    )  # type: ignore


def get_inference_mode(mode: str):
    return {
        "single": sliding_inference,
        "tta": tta_inference,
        "ensemble": ensemble_inference,
        "mcdo": mc_dropout_inference,
    }[mode]
