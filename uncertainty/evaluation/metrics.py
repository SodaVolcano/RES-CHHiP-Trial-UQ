from itertools import combinations_with_replacement, product
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
from .surface_dice import compute_surface_dice_at_tolerance, compute_surface_distances


@curry
def _distance_with_default(
    distance: Callable[[np.ndarray, np.ndarray], float], pred: np.ndarray, y: np.ndarray
) -> float:
    """
    Return default distance 0 if either `pred` or `y` are empty
    """
    return distance(pred, y) if pred.sum() > 0 and y.sum() > 0 else 0


def _average_methods(average: Literal["micro", "macro", "none"]) -> Callable[
    [np.ndarray, np.ndarray, Callable[[np.ndarray, np.ndarray], float]],
    torch.Tensor,
]:
    """
    Return a function that take in `preds`, `label`, and `metric` and compute
    the metric between them and averages them according to `average`
    """
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


@curry
def surface_dice(
    prediction: torch.Tensor,
    label: torch.Tensor,
    tolerance: float,
    average: Literal["micro", "macro", "none"] = "macro",
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
    average : Literal["micro", "macro", "weighted", "none"]
        Averaging method for the metric.
        - "micro": Calculate metrics globally across all classes.
        - "macro": Calculate metrics for each class and average them.
        - "weighted": Weighted averaging of metrics.
        - "none": Return the metrics for each class separately.
    """

    def compute_surface_dice(pred: np.ndarray, y: np.ndarray) -> float:
        """
        Expects pred and y to be 3D binary masks with uniform 1 mm spacing
        """
        distances = compute_surface_distances(pred, y, (1, 1, 1))
        return compute_surface_dice_at_tolerance(distances, tolerance)

    return tz.pipe(
        _prepare_tensors(prediction, label),
        lambda pred_label: (pred_label[0] > 0.5, pred_label[1].astype(bool)),
        lambda pred_label: _average_methods(average)(
            pred_label[0],
            pred_label[1],
            _distance_with_default(compute_surface_dice),
        ),
    )  # type: ignore


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


def general_energy_distance(
    a: torch.Tensor,
    b: torch.Tensor,
    distance: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = (
        lambda x, y: 1 - dice(x > 0.5, y > 0.5)
    ),
) -> torch.Tensor:
    """
    Calculate the General Energy Distance (GED) between `a` and `b` using `distance`.

    Parameters
    ----------
    a : torch.Tensor
        Tensor of shape `(N, C, ...)` containing N samples of data sampled
        from the first distribution.

    b : torch.Tensor
        Tensor of shape `(M, C, ...)` containing M samples of data sampled
        from the second distribution.

    distance : callable
        A function that computes the distance between two samples (from tensors `a` and `b`)
        that returns smaller value for more similar samples.
        Default is 1 - dice(x > 0.5, y > 0.5).

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
    # lambda x, y: 1 - dice(x > 0.5, y > 0.5)
    E_d_ab = np.mean([distance(x1, x2) for x1, x2 in product(a, b)])
    E_d_aa = np.mean(
        [distance(x1, x2) for x1, x2 in combinations_with_replacement(a, 2)]
    )
    E_d_bb = np.mean(
        [distance(x1, x2) for x1, x2 in combinations_with_replacement(b, 2)]
    )

    ged_squared = 2 * E_d_ab - E_d_aa - E_d_bb
    return torch.tensor(np.sqrt(ged_squared) if ged_squared >= 0 else 0.0)


def _get_metric_func(name: str):

    tolerance = float(name.split("_")[-1]) if "surface_dice" in name else 0

    return {
        "hd": hausdorff_distance,
        "hd95": hausdorff_distance_95,
        "asd": average_surface_distance,
        "assd": average_symmetric_surface_distance,
        "dice": dice,
        f"surface_dice_{tolerance}": surface_dice(tolerance=tolerance),
        "recall": recall,
        "sen": recall,
        "precision": precision,
        "ppv": precision,
    }[name]
