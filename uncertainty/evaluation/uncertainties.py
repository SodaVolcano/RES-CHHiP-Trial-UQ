from itertools import combinations_with_replacement
from typing import Callable, Literal
from numpy import var
import torch

from ..utils.wrappers import curry

import toolz as tz
from toolz import curried
from .metrics import dice, surface_dice


def _stack_if_sequence(
    preds: torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor],
) -> torch.Tensor:
    """
    Stack predictions if they are a sequence
    """
    preds = list(preds) if not isinstance(preds, list | tuple | torch.Tensor) else preds
    if isinstance(preds, list | tuple):
        return torch.stack(preds)
    return preds


@curry
def probability_map(
    preds: torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor],
) -> torch.Tensor:
    """
    Produce a probability map from an iterable of tensors

    Parameters
    ----------
    preds: torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor]
        Tensor of shape (N, ...) where N is the number of tensors
        or a list/tuple of N tensors

    Returns
    -------
    torch.Tensor
        Probability map from averaging the input tensors
    """
    return _stack_if_sequence(preds).mean(dim=0)


def entropy_map(
    preds: torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor],
    smooth: float = 1e-6,
) -> torch.Tensor:
    """
    Compute entropy map from a list of predictions

    Parameters
    ----------
    preds: torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor]
        Tensor of shape (N, ...) where N is the number of tensors
        or a list/tuple of N tensors

    Returns
    -------
    torch.Tensor
        Entropy map computed from the input tensors
    """
    preds = _stack_if_sequence(preds)
    return -torch.sum(preds * torch.log(preds + smooth), dim=0)


def entropy_map_pixel_wise(
    prob_map: torch.Tensor, smooth: float = 1e-6
) -> torch.Tensor:
    """
    Compute pixel-wise entropy for a softmax map

    Entropy is defined as -(p * log(p) + (1 - p) * log(1 - p))
    for binary classification.

    Parameters
    ----------
    softmax_map : torch.Tensor
        Softmax map of shape (C, H, W, D) with C classes.
    """
    log_prob_map = torch.log(prob_map + smooth)
    prob_map_bg = 1 - prob_map
    log_prob_map_bg = torch.log(prob_map_bg + smooth)
    return -(prob_map * log_prob_map + prob_map_bg * log_prob_map_bg)


def variance_map(
    preds: torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor],
) -> torch.Tensor:
    """
    Compute variance map from a list of predictions

    Parameters
    ----------
    preds: torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor]
        Tensor of shape (N, ...) where N is the number of tensors
        or a list/tuple of N tensors

    Returns
    -------
    torch.Tensor
        Variance map computed from the input tensors
    """
    return _stack_if_sequence(preds).var(dim=0)


def variance_pixel_wise(prob_map: torch.Tensor) -> torch.Tensor:
    """
    Compute pixel-wise variance for a softmax map

    Variance of a Bernoulli distribution is p * (1-p).

    Parameters
    ----------
    softmax_map : torch.Tensor
        Softmax map of shape (C, H, W, D) with C classes.
    """
    return prob_map * (1 - prob_map)


def mean_variance(
    preds: torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor],
    average: Literal["micro", "macro", "none"] = "macro",
) -> torch.Tensor:
    """
    Compute scalar uncertainty as average variance from list of predictions

    Parameters
    ----------
    preds: torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor],
        Prediction, either a tensor of shape (N, C, ...) or a sequence of
        (C, ...) tensors.
    average : Literal["micro", "macro", "weighted", "none"]
        Averaging method for the class channels.
        - "micro": Calculate metrics globally across all classes.
        - "macro": Calculate metrics for each class and average them.
        - "none": Return the metrics for each class separately.

    Returns
    -------
    torch.Tensor
        Tensor of mean variance of shape (,) if average is not "none", else
        a tensor of shape (C,)
    """
    var_map = variance_map(preds)
    return (
        var_map.mean()
        if average != "none"
        else torch.cat(
            [var_map[i].mean().unsqueeze(0) for i in range(var_map.shape[0])]
        )
    )


def mean_variance_pixel_wise(
    prob_map: torch.Tensor,
    average: Literal["micro", "macro", "none"] = "macro",
) -> torch.Tensor:
    """
    Compute pixel-wise variance from a softmax map

    Parameters
    ----------
    prob_map : torch.Tensor
        Softmax map of shape (C, H, W, D) with C classes.
    average : Literal["micro", "macro", "weighted", "none"]
        Averaging method for the class channels.
        - "micro": Calculate metrics globally across all classes.
        - "macro": Calculate metrics for each class and average
        - "none": Return the metrics for each class separately.

    Returns
    -------
    torch.Tensor
        Tensor of mean variance of shape (,) if average is not "none", else
        a tensor of shape (C,)
    """
    var_map = variance_pixel_wise(prob_map)
    return (
        var_map.mean()
        if average != "none"
        else torch.cat(
            [var_map[i].mean().unsqueeze(0) for i in range(var_map.shape[0])]
        )
    )


def mean_entropy(
    preds: torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor],
    average: Literal["micro", "macro", "none"] = "macro",
    smooth: float = 1e-6,
) -> torch.Tensor:
    """
    Compute scalar uncertainty as average entropy from list of predictions

    Parameters
    ----------
    preds: torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor],
        Prediction, either a tensor of shape (N, C, ...) or a sequence of
        (C, ...) tensors.
    average : Literal["micro", "macro", "weighted", "none"]
        Averaging method for the class channels.
        - "micro": Calculate metrics globally across all classes.
        - "macro": Calculate metrics for each class and average them.
        - "none": Return the metrics for each class separately.

    Returns
    -------
    torch.Tensor
        Tensor of mean variance of shape (,) if average is not "none", else
        a tensor of shape (C,)
    """
    ent_map = entropy_map(preds, smooth)
    return (
        ent_map.mean()
        if average != "none"
        else torch.cat(
            [ent_map[i].mean().unsqueeze(0) for i in range(ent_map.shape[0])]
        )
    )


def mean_entropy_pixel_wise(
    prob_map: torch.Tensor,
    average: Literal["micro", "macro", "none"] = "macro",
    smooth: float = 1e-7,
) -> torch.Tensor:
    """
    Compute pixel-wise entropy from a softmax map

    Parameters
    ----------
    prob_map : torch.Tensor
        Softmax map of shape (C, H, W, D) with C classes.
    average : Literal["micro", "macro", "weighted", "none"]
        Averaging method for the class channels.
        - "micro": Calculate metrics globally across all classes.
        - "macro": Calculate metrics for each class and average
        - "none": Return the metrics for each class separately.

    Returns
    -------
    torch.Tensor
        Tensor of mean variance of shape (,) if average is not "none", else
        a tensor of shape (C,)
    """
    ent_map = entropy_map_pixel_wise(prob_map, smooth)
    return (
        ent_map.mean()
        if average != "none"
        else torch.cat(
            [ent_map[i].mean().unsqueeze(0) for i in range(ent_map.shape[0])]
        )
    )


def _pairwise_metric(
    metric: Callable[
        [torch.Tensor, torch.Tensor, Literal["micro", "macro", "weighted", "none"]],
        torch.Tensor,
    ],
    predictions: torch.Tensor,
    average: Literal["micro", "macro", "weighted", "none"] = "macro",
    aggregate: bool = True,
):
    """
    Compute the average pairwise metric between all pairs of predictions.

    Parameters
    ----------
    metric : callable
        The metric function to use for comparison.
    predictions : torch.Tensor
        The predicted tensor of shape `(N, C, ...)`.
    average : Literal["micro", "macro", "weighted", "none"]
        Averaging method for the class channels.
        - "micro": Calculate dice globally across all classes.
        - "macro": Calculate dice for each class and average
        - "weighted": Weighted averaging of dice scores.
        - "none": Return the dice for each class separately.
    aggregate : bool
        Whether to average the dice scores into a single tensor or
        to return the dice scores for each class separately. Have
        no effect if `average != "none"`.

    Returns
    -------
    torch.Tensor
        The average pairwise dice similarity score of shape (1,) of
        mean dice score across all pairs of predictions across all
        classes if `aggregate=True` or (C,) if `aggregate=False` for
        each of the C classes.
    """
    return tz.pipe(
        predictions,
        lambda preds: combinations_with_replacement(preds, 2),
        curried.filter(lambda x: not x[0].equal(x[1])),
        curried.map(lambda x: metric(x[0], x[1], average)),
        list,
        torch.stack,
        lambda preds: torch.mean(preds, dim=0 if aggregate else None),
    )


def pairwise_dice(
    predictions: torch.Tensor,
    average: Literal["micro", "macro", "weighted", "none"] = "macro",
    aggregate: bool = True,
):
    """
    Compute the average pairwise dice similarity between all pairs of predictions.

    Parameters
    ----------
    predictions : torch.Tensor
        The predicted tensor of shape `(N, C, ...)`.
    average : Literal["micro", "macro", "weighted", "none"]
        Averaging method for the class channels.
        - "micro": Calculate dice globally across all classes.
        - "macro": Calculate dice for each class and average them.
        - "weighted": Weighted averaging of dice scores.
        - "none": Return the dice for each class separately.
    aggregate : bool
        Whether to average the dice scores into a single tensor or
        to return the dice scores for each class separately. Have
        no effect if `average != "none"`.

    Returns
    -------
    torch.Tensor
        The average pairwise dice similarity score of shape (1,) of
        mean dice score across all pairs of predictions across all
        classes if `aggregate=True` or (C,) if `aggregate=False` for
        each of the C classes.
    """
    return _pairwise_metric(dice, predictions, average, aggregate)


def pairwise_surface_dice(
    predictions: torch.Tensor,
    average: Literal["micro", "macro", "none"] = "macro",
    aggregate: bool = True,
    tolerance: float = 1.0,
):
    """
    Compute the average pairwise surface dice similarity between all pairs of predictions.

    Parameters
    ----------
    predictions : torch.Tensor
        The predicted tensor of shape `(N, C, ...)`.
    label : torch.Tensor
        The ground truth tensor of shape `(C, ...)`.
    tolerance : float
        Tolerance in mm for the surface dice computation.
    average : Literal["micro", "macro", "weighted", "none"]
        Averaging method for the class channels.
        - "micro": Calculate dice globally across all classes.
        - "macro": Calculate dice for each class and average
        - "none": Return the dice for each class separately.
    aggregate : bool
        Whether to average the dice scores into a single tensor or
        to return the dice scores for each class separately. Have
        no effect if `average != "none`.

    Returns
    -------
    torch.Tensor
        The average pairwise surface dice similarity score of shape (1,) of
        mean dice score across all pairs of predictions across all
        classes if `aggregate=True` or (C,) if `aggregate=False` for
        each of the C classes.
    """
    return _pairwise_metric(
        surface_dice(tolerance=tolerance), predictions, average, aggregate
    )
