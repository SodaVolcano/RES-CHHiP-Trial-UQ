"""
Classification metrics for evaluating model performance.
"""

from itertools import combinations_with_replacement, product
from typing import Callable, Literal, Optional

import numpy as np
import toolz as tz
import torch
from loguru import logger
from medpy.metric import asd, assd, hd, hd95
from toolz import curried
from torch.nn.functional import sigmoid
from torchmetrics.classification import (
    BinaryF1Score,
    BinaryPrecision,
    BinaryRecall,
    MultilabelF1Score,
    MultilabelPrecision,
    MultilabelRecall,
)

from ..utils import curry, star, starmap
from .surface_dice import compute_surface_dice_at_tolerance, compute_surface_distances


@curry
def _distance_with_default(
    distance: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    pred: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
    """
    Return default distance 0 if either `pred` or `y` are empty
    """
    return distance(pred, y) if pred.sum() > 0 and y.sum() > 0 else torch.tensor(0)


def _average_methods(average: Literal["micro", "macro", "none"]) -> Callable[
    [torch.Tensor, torch.Tensor, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]],
    torch.Tensor,
]:
    """
    Return a function that take in `preds`, `label`, and `metric` and compute
    the metric between them and averages them according to `average`
    """
    return {
        "micro": lambda pred, label, metric: metric(pred, label),
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
    Apply sigmoid if `prediction` not in [0, 1], threshold, and convert to numpy array
    """
    return tz.pipe(
        prediction,
        lambda x: sigmoid(x) if x.min() < 0 or x.max() > 1 else x,
        lambda pred: (pred > 0.5, label),
        curried.map(lambda arr: arr.detach().numpy()),
        tuple,
    )  # type: ignore


def _medpy_wrapper(
    metric: Callable[..., torch.Tensor]
) -> Callable[
    [torch.Tensor, torch.Tensor, Literal["micro", "macro", "none"]], torch.Tensor
]:
    """
    Wrapper around medpy hd, hd95 etc: prepare tensor, apply metric, and return tensor
    """
    return lambda prediction, label, average: tz.pipe(
        _prepare_tensors(prediction, label),
        star(
            lambda pred, label: _average_methods(average)(
                pred, label, _distance_with_default(metric)  # type: ignore
            )
        ),
        lambda arr: torch.tensor(arr) if not isinstance(arr, torch.Tensor) else arr,
    )  # type: ignore


def _torchmetric_wrapper(
    binary_metric: Callable[..., torch.Tensor],
    multilabel_metric: Callable[..., Callable[..., torch.Tensor]],
    prediction: torch.Tensor,
    label: torch.Tensor,
    average: Literal["micro", "macro", "weighted", "none"],
) -> torch.Tensor:
    """
    Wrapper around torchmetrics: select binary or multilabel metric, apply metric, and return tensor
    """
    metric = (
        binary_metric
        if label.shape[0] == 1
        else multilabel_metric(num_labels=label.shape[0], average=average)
    )
    return metric(prediction.unsqueeze(0), label.unsqueeze(0))


def _torchmetric_wrapper_batched(
    binary_metric: Callable[..., torch.Tensor],
    multilabel_metric: Callable[..., Callable[..., torch.Tensor]],
    prediction: torch.Tensor,
    label: torch.Tensor,
    average: Literal["micro", "macro", "weighted", "none"],
) -> torch.Tensor:
    """
    Wrapper around torchmetricsfor batched inputs: select binary or multilabel metric,
    """
    metric = (
        binary_metric
        if label.shape[1] == 1
        else multilabel_metric(num_labels=label.shape[1], average=average)
    )
    return metric(prediction, label)


def _batched_eval(
    prediction: torch.Tensor, label: torch.Tensor, metric: Callable[..., torch.Tensor]
) -> torch.Tensor:
    """
    Apply metric to batched inputs
    """
    return tz.pipe(
        zip(prediction, label),
        starmap(metric),
        list,
        torch.stack,
        curry(torch.mean, fallback=True)(dim=0),
    )  # type: ignore


@curry
def dice(
    prediction: torch.Tensor,
    label: torch.Tensor,
    average: Literal["micro", "macro", "weighted", "none"] = "macro",
) -> torch.Tensor:
    """
    Compute the Dice coefficient between two tensors of shape (B, C, ...).

    Parameters
    ----------
    prediction : torch.Tensor
        The predicted tensor of shape (C, ...).
    label : torch.Tensor
        The ground truth tensor of shape (C, ...).
    average : Literal["micro", "macro", "weighted", "none"]
        Averaging method for the class channels.
        - "micro": Calculate metrics globally across all classes.
        - "macro": Calculate metrics for each class and average them.
        - "weighted": Weighted averaging of metrics.
        - "none": Return the metrics for each class separately.
    """
    return _torchmetric_wrapper(
        BinaryF1Score(zero_division=1),
        curry(MultilabelF1Score)(zero_division=1),  # type: ignore
        prediction,
        label,
        average,
    )


@curry
def dice_batched(
    prediction: torch.Tensor,
    label: torch.Tensor,
    average: Literal["micro", "macro", "weighted", "none"] = "macro",
) -> torch.Tensor:
    """
    Compute the Dice coefficient between two tensors of shape (B, C, ...).

    Parameters
    ----------
    prediction : torch.Tensor
        The predicted tensor of shape (B, C, ...).
    label : torch.Tensor
        The ground truth tensor of shape (B, C, ...).
    average : Literal["micro", "macro", "weighted", "none"]
        Averaging method for the class channels.
        - "micro": Calculate metrics globally across all classes.
        - "macro": Calculate metrics for each class and average
        - "weighted": Weighted averaging of metrics.
        - "none": Return the metrics for each class separately.
    """
    return _torchmetric_wrapper_batched(
        BinaryF1Score(zero_division=1),
        curry(MultilabelF1Score)(zero_division=1),  # type: ignore
        prediction,
        label,
        average,
    )


@curry
def surface_dice(
    prediction: torch.Tensor,
    label: torch.Tensor,
    average: Literal["macro", "none"] = "macro",
    tolerance: float = 1.0,
) -> torch.Tensor:
    """
    Compute the Surface Dice between two tensors of shape (C, ...) at specified tolerance.

    The surface DICE measures the overlap of two surfaces instead of two volumes.
    A surface element is counted as overlapping (or touching), when the closest
    distance to the other surface is less or equal to the specified tolerance.
    The DICE coefficient is in the range between 0.0 (no overlap) to 1.0
    (perfect overlap).

    Taken from https://github.com/google-deepmind/surface-distance

    Parameters
    ----------
    prediction : torch.Tensor
        The predicted tensor of shape (C, ...).
    label : torch.Tensor
        The ground truth tensor of shape (C, ...).
    tolerance : float
        Tolerance in mm.
    average : Literal["macro", "none"]
        Averaging method for the class channels.
        - "macro": Calculate metrics for each class and average them.
        - "none": Return the metrics for each class separately.
    """
    if average == "micro":
        logger.error(
            "Micro averaging is not supported for surface dice, nan tensor returned"
        )
        return torch.tensor(torch.nan)

    def compute_surface_dice(pred: torch.Tensor, y: torch.Tensor) -> float:
        """
        Expects pred and y to be 3D binary masks with uniform 1 mm spacing
        """
        distances = compute_surface_distances(pred, y, (1, 1, 1))
        return compute_surface_dice_at_tolerance(distances, tolerance)

    return tz.pipe(
        _prepare_tensors(prediction, label),
        star(lambda pred, label: (pred, label.astype(bool))),
        star(
            lambda pred, label: _average_methods(average)(
                pred,
                label,
                _distance_with_default(compute_surface_dice),  # type: ignore
            )
        ),
    )  # type: ignore


@curry
def surface_dice_batched(
    prediction: torch.Tensor,
    label: torch.Tensor,
    average: Literal["macro", "none"] = "macro",
    tolerance: float = 1.0,
) -> torch.Tensor:
    """
    Compute the Surface Dice between two tensors of shape (B, C, ...) at specified tolerance.

    The surface DICE measures the overlap of two surfaces instead of two volumes.
    A surface element is counted as overlapping (or touching), when the closest
    distance to the other surface is less or equal to the specified tolerance.

    Parameters
    ----------
    prediction : torch.Tensor
        The predicted tensor of shape (B, C, ...).
    label : torch.Tensor
        The ground truth tensor of shape (B, C, ...).
    tolerance : float
        Tolerance in mm.
    average : Literal["macro", "none"]
        Averaging method for the class channels.
        - "macro": Calculate metrics for each class and average
        - "none": Return the metrics for each class separately.
    """
    if average == "micro":
        logger.error(
            "Micro averaging is not supported for surface dice, nan tensor returned"
        )
        return torch.tensor(torch.nan)

    return _batched_eval(
        prediction, label, surface_dice(average=average, tolerance=tolerance)  # type: ignore
    )


@curry
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
        Averaging method for the class channels.
        - "micro": Calculate metrics globally across all classes.
        - "macro": Calculate metrics for each class and average them.
        - "none": Return the metrics for each class separately.
    """
    return _medpy_wrapper(hd)(prediction, label, average)


@curry
def hausdorff_distance_batched(
    prediction: torch.Tensor,
    label: torch.Tensor,
    average: Literal["micro", "macro", "none"] = "macro",
) -> torch.Tensor:
    """
    Compute the Hausdorff distance between two tensors of shape (B, C, ...).

    If `prediction` is not in the range [0, 1], it is assumed to be logits and
    a sigmoid activation is applied.

    Parameters
    ----------
    prediction : torch.Tensor
        The predicted tensor of shape (B, C, ...).
    label : torch.Tensor
        The ground truth tensor of shape (B, C, ...).
    average : Literal["micro", "macro", "none"]
        Averaging method for the class channels.
        - "micro": Calculate metrics globally across all classes.
        - "macro": Calculate metrics for each class and average them.
        - "none": Return the metrics for each class separately.
    """
    return _batched_eval(prediction, label, hausdorff_distance(average=average))  # type: ignore


@curry
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
        Averaging method for the class channels.
        - "micro": Calculate metrics globally across all classes.
        - "macro": Calculate metrics for each class and average them.
        - "none": Return the metrics for each class separately.
    """
    return _medpy_wrapper(hd95)(prediction, label, average)  # type: ignore


def hausdorff_distance_95_batched(
    prediction: torch.Tensor,
    label: torch.Tensor,
    average: Literal["micro", "macro", "none"] = "macro",
) -> torch.Tensor:
    """
    Compute the 95th percentile Hausdorff distance between two tensors of shape (B, C, ...).

    If `prediction` is not in the range [0, 1], it is assumed to be logits and
    a sigmoid activation is applied.

    Parameters
    ----------
    prediction : torch.Tensor
        The predicted tensor of shape (B, C, ...).
    label : torch.Tensor
        The ground truth tensor of shape (B, C, ...).
    average : Literal["micro", "macro", "none"]
        Averaging method for the class channels.
        - "micro": Calculate metrics globally across all classes.
        - "macro": Calculate metrics for each class and average them.
        - "none": Return the metrics for each class separately.
    """
    return _batched_eval(prediction, label, hausdorff_distance_95(average=average))  # type: ignore


@curry
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
        Averaging method for the class channels.
        - "micro": Calculate metrics globally across all classes.
        - "macro": Calculate metrics for each class and average them.
        - "weighted": Weighted averaging of metrics.
        - "none": Return the metrics for each class separately.
    """
    return _torchmetric_wrapper(
        BinaryRecall(), MultilabelRecall, prediction, label, average
    )


@curry
def recall_batched(
    prediction: torch.Tensor,
    label: torch.Tensor,
    average: Literal["micro", "macro", "weighted", "none"] = "macro",
) -> torch.Tensor:
    """
    Compute the recal between two tensors of shape (B, C, ...).

    AKA sensitivity, or true positive rate (TPR).

    Parameters
    ----------
    prediction : torch.Tensor
        The predicted tensor of shape (B, C, ...).
    label : torch.Tensor
        The ground truth tensor of shape (B, C, ...).
    average : Literal["micro", "macro", "weighted", "none"]
        Averaging method for the class channels.
        - "micro": Calculate metrics globally across all classes.
        - "macro": Calculate metrics for each class and average them.
        - "weighted": Weighted averaging of metrics.
        - "none": Return the metrics for each class separately.
    """
    return _torchmetric_wrapper_batched(
        BinaryRecall(), MultilabelRecall, prediction, label, average
    )


@curry
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
        Averaging method for the class channels.
        - "micro": Calculate metrics globally across all classes.
        - "macro": Calculate metrics for each class and average them.
        - "weighted": Weighted averaging of metrics.
        - "none": Return the metrics for each class separately.
    """
    return _torchmetric_wrapper(
        BinaryPrecision(), MultilabelPrecision, prediction, label, average
    )


@curry
def precision_batched(
    prediction: torch.Tensor,
    label: torch.Tensor,
    average: Literal["micro", "macro", "weighted", "none"] = "macro",
) -> torch.Tensor:
    """
    Compute the precision between two tensors of shape (B, C, ...).

    AKA positive predictive value (PPV).

    Parameters
    ----------
    prediction : torch.Tensor
        The predicted tensor of shape (B, C, ...).
    label : torch.Tensor
        The ground truth tensor of shape (B, C, ...).
    average : Literal["micro", "macro", "weighted", "none"]
        Averaging method for the class channels.
        - "micro": Calculate metrics globally across all classes.
        - "macro": Calculate metrics for each class and average
        - "weighted": Weighted averaging of metrics.
        - "none": Return the metrics for each class separately.
    """
    return _torchmetric_wrapper_batched(
        BinaryPrecision(), MultilabelPrecision, prediction, label, average
    )


@curry
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
        Averaging method for the class channels.
        - "micro": Calculate metrics globally across all classes.
        - "macro": Calculate metrics for each class and average them.
        - "none": Return the metrics for each class separately.
    """
    return _medpy_wrapper(asd)(prediction, label, average)


@curry
def average_surface_distance_batched(
    prediction: torch.Tensor,
    label: torch.Tensor,
    average: Literal["micro", "macro", "none"] = "macro",
) -> torch.Tensor:
    """
    Compute the Average Surface Distance between two tensors of shape (B, C, ...).

    Parameters
    ----------
    prediction : torch.Tensor
        The predicted tensor of shape (B, C, ...).
    label : torch.Tensor
        The ground truth tensor of shape (B, C, ...).
    average : Literal["micro", "macro", "none"]
        Averaging method for the class channels.
        - "micro": Calculate metrics globally across all classes.
        - "macro": Calculate metrics for each class and average them.
        - "none": Return the metrics for each class separately.
    """
    return _batched_eval(prediction, label, average_surface_distance(average=average))  # type: ignore


@curry
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
        Averaging method for the class channels.
        - "micro": Calculate metrics globally across all classes.
        - "macro": Calculate metrics for each class and average them.
        - "none": Return the metrics for each class separately.
    """
    return _medpy_wrapper(assd)(prediction, label, average)


@curry
def average_symmetric_surface_distance_batched(
    prediction: torch.Tensor,
    label: torch.Tensor,
    average: Literal["micro", "macro", "none"] = "macro",
) -> torch.Tensor:
    """
    Compute the Average Symmetric Surface Distance between two tensors of shape (B, C, ...).

    Parameters
    ----------
    prediction : torch.Tensor
        The predicted tensor of shape (B, C, ...).
    label : torch.Tensor
        The ground truth tensor of shape (B, C, ...).
    average : Literal["micro", "macro", "none"]
        Averaging method for the class channels.
        - "micro": Calculate metrics globally across all classes.
        - "macro": Calculate metrics for each class and average them.
        - "none": Return the metrics for each class separately.
    """
    return _batched_eval(
        prediction, label, average_symmetric_surface_distance(average=average)  # type: ignore
    )


@curry
def generalised_energy_distance(
    a: torch.Tensor,
    b: torch.Tensor,
    distance: (
        Literal["hd", "hd95", "asd", "assd", "dice"]
        | Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    ) = "dice",
    average: Literal["micro", "macro", "none"] = "macro",
) -> torch.Tensor:
    """
    Calculate the Generalised Energy Distance (GED) between `a` and `b` using `distance`.

    Parameters
    ----------
    a : torch.Tensor
        Tensor of shape `(N, C, ...)` containing N samples of data sampled
        from the first distribution.
    b : torch.Tensor
        Tensor of shape `(M, C, ...)` containing M samples of data sampled
        from the second distribution.
    distance : str | callable
        A function that computes the distance between two samples (from tensors `a` and `b`)
        that returns smaller value for more similar samples.
        Default is 1 - dice(x > 0.5, y > 0.5).
    average : Literal["micro", "macro", "none"]
        Averaging method for the class channels. ignored if supplying custom `distance` function.
        - "micro": Calculate metrics globally across all classes.
        - "macro": Calculate metrics for each class and average them.
        - "none": Return the metrics for each class

    Returns
    -------
    torch.Tensor
        The computed General Energy Distance (GED) between the two tensors.
        If the squared distance `ged_squared` is negative due to numerical precision
        issues, the function returns 0.0.

    Notes
    -----
    The General Energy Distance is defined as:

    .. math::
       D^2_{\text{GED}}(a, b) = 2E[ d(a_i, b_i) ] - E[ d(a_i, a_j) ] - E[ d(b_i, b_j) ]

    Where:
        - `d(a_i, b_i)` is the distance between samples `a_i` from `a` and `b_i` from `b`.
        - `E[.]` denotes the expected value (i.e., mean over all sample pairs).
    """
    assert a.shape == b.shape, "Input arrays must have the same shape"

    dist_func_ = (
        (lambda x, y: (get_classification_metric(distance)(x > 0.5, y > 0.5, average=average)))  # type: ignore
        if isinstance(distance, str)
        else distance
    )
    dist_func = (
        (lambda x1, y1: 1 - dist_func_(x1, y1)) if distance == "dice" else dist_func_
    )

    get_distances = tz.compose_left(
        lambda pairs: [dist_func(x1, x2) for x1, x2 in pairs],
        torch.tensor if average != "none" else torch.stack,
    )

    E_d_ab = torch.mean(get_distances(product(a, b)), dim=0)  # type: ignore
    E_d_aa = torch.mean(get_distances(combinations_with_replacement(a, 2)), dim=0)  # type: ignore
    E_d_bb = torch.mean(get_distances(combinations_with_replacement(b, 2)), dim=0)  # type: ignore

    ged_squared = 2 * E_d_ab - E_d_aa - E_d_bb
    return torch.clip(torch.sqrt(ged_squared), 0, None)


def get_classification_metric(name: str) -> Optional[Callable]:
    """
    Return the metric function given the name, or None if not found.
    """
    if "surface_dice" in name:
        return (
            surface_dice_batched(tolerance=float(name.split("_")[-1]))
            if "batched" in name
            else surface_dice(tolerance=float(name.split("_")[-1]))
        )  # type: ignore

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
        "hd_batched": hausdorff_distance_batched,
        "hd95_batched": hausdorff_distance_95_batched,
        "asd_batched": average_surface_distance_batched,
        "assd_batched": average_symmetric_surface_distance_batched,
        "dice_batched": dice_batched,
        "recall_batched": recall_batched,
        "sen_batched": recall_batched,
        "precision_batched": precision_batched,
        "ppv_batched": precision_batched,
    }.get(name, None)
