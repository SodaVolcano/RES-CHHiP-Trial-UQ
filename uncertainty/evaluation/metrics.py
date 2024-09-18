from typing import Callable, Literal
import torch
from torchmetrics.classification import (
    MultilabelF1Score,
    MultilabelRecall,
    MultilabelPrecision,
)
from medpy.metric import hd, hd95, asd, assd
from torch.nn.functional import sigmoid
import toolz as tz
from toolz import curried
import numpy as np

from ..utils.wrappers import curry


@curry
def _distance_with_default(
    distance: Callable[[np.ndarray, np.ndarray], float], pred: np.ndarray, y: np.ndarray
) -> float:
    """
    Return default value if either `pred` or `y` are empty
    """
    return distance(pred, y) if pred.sum() > 0 and y.sum() > 0 else 0


def _average_methods(average: Literal["micro", "macro", "none"]) -> Callable[
    [np.ndarray, np.ndarray, Callable[[np.ndarray, np.ndarray], float]],
    torch.Tensor,
]:
    return {
        "micro": lambda pred, label, metric: torch.tensor(metric(pred, label)),
        "macro": lambda preds, label, metric: torch.mean(
            torch.tensor([metric(pred, y) for pred, y in zip(preds, label)]), dim=0
        ),
        "none": lambda preds, label, metric: torch.tensor(
            [metric(pred, y) for pred, y in zip(preds, label)]
        ),
    }[average]


def _prepare_tensors(
    prediction: torch.Tensor, label: torch.Tensor
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply sigmoid if `prediction` not in [0, 1], threshold, and convert to numpy arrays
    """
    return tz.pipe(
        prediction,
        lambda x: sigmoid(x) if x.min() < 0 or x.max() > 1 else x,
        lambda pred: (pred > 0.5, label),
        curried.map(lambda tensor: tensor.detach().numpy()),
        tuple,
    )  # type: ignore


def dice(
    prediction: torch.Tensor,
    label: torch.Tensor,
    average: Literal["micro", "macro", "weighted", "none"] = "macro",
) -> torch.Tensor:
    """
    Compute the Dice coefficient between two tensors of shape (C, ...).

    Parameters
    ----------
    prediction : torch.Tensor
        The predicted tensor of shape (C, ...).
    label : torch.Tensor
        The ground truth tensor of shape (C, ...).
    average : Literal["micro", "macro", "weighted", "none"]
        Averaging method for the metric.
        - "micro": Calculate metrics globally across all classes.
        - "macro": Calculate metrics for each class and average them.
        - "weighted": Weighted averaging of metrics.
        - "none": Return the metrics for each class separately.
    """

    return MultilabelF1Score(
        num_labels=label.shape[0], average=average, zero_division=1
    )(prediction.unsqueeze(0), label.unsqueeze(0))


def surface_dice(prediction: torch.Tensor, label: torch.Tensor):
    pass


def hausdorff_distance(
    prediction: torch.Tensor,
    label: torch.Tensor,
    average: Literal["micro", "macro", "none"] = "macro",
) -> torch.Tensor:
    """
    Compute the Hausdorff distance between two tensors of shape (C, ...).

    If `prediction` is not in the range [0, 1], it is assumed to be logits and
    a sigmoid activation is applied.

    Parameters
    ----------
    prediction : torch.Tensor
        The predicted tensor of shape (C, ...).
    label : torch.Tensor
        The ground truth tensor of shape (C, ...).
    average : Literal["micro", "macro", "none"]
        Averaging method for the metric.
        - "micro": Calculate metrics globally across all classes.
        - "macro": Calculate metrics for each class and average them.
        - "none": Return the metrics for each class separately.
    """
    return tz.pipe(
        _prepare_tensors(prediction, label),
        lambda np_arrays: _average_methods(average)(
            np_arrays[0], np_arrays[1], _distance_with_default(hd)
        ),
    )  # type: ignore


def hausdorff_distance_95(
    prediction: torch.Tensor,
    label: torch.Tensor,
    average: Literal["micro", "macro", "none"] = "macro",
) -> torch.Tensor:
    """
    Compute the 95th percentile Hausdorff distance between two tensors of shape (C, ...).

    If `prediction` is not in the range [0, 1], it is assumed to be logits and
    a sigmoid activation is applied.

    Parameters
    ----------
    prediction : torch.Tensor
        The predicted tensor of shape (C, ...).
    label : torch.Tensor
        The ground truth tensor of shape (C, ...).
    average : Literal["micro", "macro", "none"]
        Averaging method for the metric.
        - "micro": Calculate metrics globally across all classes.
        - "macro": Calculate metrics for each class and average them.
        - "none": Return the metrics for each class separately.
    """
    return tz.pipe(
        _prepare_tensors(prediction, label),
        lambda np_arrays: _average_methods(average)(np_arrays[0], np_arrays[1], _distance_with_default(hd95)),  # type: ignore
    )  # type: ignore


def recall(
    prediction: torch.Tensor,
    label: torch.Tensor,
    average: Literal["micro", "macro", "weighted", "none"] = "macro",
) -> torch.Tensor:
    """
    Compute the recal between two tensors of shape (C, ...).

    AKA sensitivity, or true positive rate (TPR).

    Parameters
    ----------
    prediction : torch.Tensor
        The predicted tensor of shape (C, ...).
    label : torch.Tensor
        The ground truth tensor of shape (C, ...).
    average : Literal["micro", "macro", "weighted", "none"]
        Averaging method for the metric.
        - "micro": Calculate metrics globally across all classes.
        - "macro": Calculate metrics for each class and average them.
        - "weighted": Weighted averaging of metrics.
        - "none": Return the metrics for each class separately.
    """
    return MultilabelRecall(num_labels=label.shape[0], average=average)(
        prediction.unsqueeze(0), label.unsqueeze(0)
    )


def precision(
    prediction: torch.Tensor,
    label: torch.Tensor,
    average: Literal["micro", "macro", "weighted", "none"] = "macro",
) -> torch.Tensor:
    """
    Compute the precision between two tensors of shape (C, ...).

    AKA positive predictive value (PPV).

    Parameters
    ----------
    prediction : torch.Tensor
        The predicted tensor of shape (C, ...).
    label : torch.Tensor
        The ground truth tensor of shape (C, ...).
    average : Literal["micro", "macro", "weighted", "none"]
        Averaging method for the metric.
        - "micro": Calculate metrics globally across all classes.
        - "macro": Calculate metrics for each class and average them.
        - "weighted": Weighted averaging of metrics.
        - "none": Return the metrics for each class separately.
    """
    return MultilabelPrecision(num_labels=label.shape[0], average=average)(
        prediction.unsqueeze(0), label.unsqueeze(0)
    )


def average_surface_distance(
    prediction: torch.Tensor,
    label: torch.Tensor,
    average: Literal["micro", "macro", "none"] = "macro",
) -> torch.Tensor:
    """
    Compute the Average Surface Distance between two tensors of shape (C, ...).

    Parameters
    ----------
    prediction : torch.Tensor
        The predicted tensor of shape (C, ...).
    label : torch.Tensor
        The ground truth tensor of shape (C, ...).
    average : Literal["micro", "macro", "none"]
        Averaging method for the metric.
        - "micro": Calculate metrics globally across all classes.
        - "macro": Calculate metrics for each class and average them.
        - "none": Return the metrics for each class separately.
    """
    return tz.pipe(
        _prepare_tensors(prediction, label),
        lambda np_arrays: _average_methods(average)(
            np_arrays[0], np_arrays[1], _distance_with_default(asd)
        ),
    )  # type: ignore


def average_symmetric_surface_distance(
    prediction: torch.Tensor,
    label: torch.Tensor,
    average: Literal["micro", "macro", "none"] = "macro",
) -> torch.Tensor:
    """
    Compute the Average Symmetric Surface Distance between two tensors of shape (C, ...).

    Parameters
    ----------
    prediction : torch.Tensor
        The predicted tensor of shape (C, ...).
    label : torch.Tensor
        The ground truth tensor of shape (C, ...).
    average : Literal["micro", "macro", "none"]
        Averaging method for the metric.
        - "micro": Calculate metrics globally across all classes.
        - "macro": Calculate metrics for each class and average them.
        - "none": Return the metrics for each class separately.
    """
    return tz.pipe(
        _prepare_tensors(prediction, label),
        lambda np_arrays: _average_methods(average)(
            np_arrays[0], np_arrays[1], _distance_with_default(assd)
        ),
    )  # type: ignore


def generalised_energy_distance(
    prediction: torch.Tensor,
    label: torch.Tensor,
    average: Literal["micro", "macro", "none"] = "macro",
) -> torch.Tensor:
    pass


def _get_metric_func(
    name: Literal[
        "hd",
        "hd95",
        "asd",
        "assd",
        "dice",
        "recall",
        "precision",
        "ppv",
    ]
):
    return {
        "hd": hausdorff_distance,
        "hd95": hausdorff_distance_95,
        "asd": average_surface_distance,
        "assd": average_symmetric_surface_distance,
        "dice": dice,
        "recall": recall,
        "sen": recall,
        "precision": precision,
        "ppv": precision,
    }[name]


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
        torch.tensor,
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
        list,
        torch.stack,
        lambda x: torch.mean(x, dim=0) if summarise else x,
    )  # type: ignore
