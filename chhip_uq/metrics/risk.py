import numpy as np
import torch

from ..utils import curry


@curry
def rc_curve_stats(
    risks: torch.Tensor, confids: torch.Tensor
) -> tuple[list[float], list[torch.Tensor], list[float]]:
    """
    Return coverage, selective risk, weights for the Risk-Confidence curve.

    RC-curve is produced by sliding a confidence threshold over `confids`.
    In actuality, sliding the threshold just means we are iteratively removing
    samples with the lowest confidence from calculation.

    The selective risk is the average risk of all samples currently considered.
    If we have N risks for N samples `sorted_risks` sorted in ascending confidence
    using `confids`, the selective risk at the i-th point is:
        R_s(i) = sum([r for r in sorted_risks[i:]]) / len(sorted_risks[i:])

    The coverage is the fraction of samples above some threshold. If we have N samples,
    the coverage at the i-th point is:
        C(i) = (N - i) / N

    Parameters
    ----------
    risks : torch.Tensor
        Array of shape `(N,)` containing the risk scores for N test samples.
    confids : torch.Tensor
        Array of shape `(N,)` containing the confidence scores for N test samples.

    Returns
    -------
    tuple[list[float], list[float], list[float]]
        The coverages, selective risks, and weights for the Risk-Confidence curve.

    References
    ----------
    Taken from:
        https://arxiv.org/abs/2401.08501
        https://github.com/IML-DKFZ/values
    """
    assert (
        len(risks.shape) == 1 and len(confids.shape) == 1 and len(risks) == len(confids)
    )

    coverages = []
    selective_risks = []

    n_samples = len(risks)
    idx_sorted = np.argsort(confids)

    coverage = n_samples
    error_sum = sum(risks[idx_sorted])

    coverages.append(coverage / n_samples)
    selective_risks.append(error_sum / n_samples)

    weights = []

    tmp_weight = 0
    for i in range(0, len(idx_sorted) - 1):
        coverage = coverage - 1
        error_sum -= risks[idx_sorted[i]]
        tmp_weight += 1
        if i == 0 or confids[idx_sorted[i]] != confids[idx_sorted[i - 1]]:
            coverages.append(coverage / n_samples)
            selective_risks.append(error_sum / (n_samples - 1 - i))
            weights.append(tmp_weight / n_samples)
            tmp_weight = 0

    # add a well-defined final point to the RC-curve.
    if tmp_weight > 0:
        coverages.append(0)
        selective_risks.append(selective_risks[-1])
        weights.append(tmp_weight / n_samples)

    return coverages, selective_risks, weights


@curry
def aurc(
    risks: torch.Tensor, confids: torch.Tensor
) -> tuple[torch.Tensor, list[float], list[torch.Tensor]]:
    """
    Compute the Area Under the Risk-Confidence curve (AURC).

    Average selective risk across the confidence thresholds.

    Parameters
    ----------
    risks : torch.Tensor
        Tensor of shape `(N,)` containing the risk scores for N test
        samples.
    confidences : torch.Tensor
        Tensor of shape `(N,)` containing the confidence scores for N
        test samples.

    Returns
    -------
    tuple[torch.Tensor, list[float], list[torch.Tensor]]
        The computed Area Under the Risk-Confidence curve (AURC),
        the coverages, and selective risks for the Risk-Confidence curve.

    References
    ----------
    Taken from:
        https://arxiv.org/abs/2401.08501
        https://github.com/IML-DKFZ/values
    """

    coverage, selective_risks, weights = rc_curve_stats(risks, confids)
    return (
        torch.sum(
            torch.tensor(
                [
                    (selective_risks[i] + selective_risks[i + 1]) * 0.5 * weights[i]
                    for i in range(len(weights))
                ]
            )
        ),
        coverage,
        selective_risks,
    )


@curry
def eaurc(
    risks: torch.Tensor, confids: torch.Tensor
) -> tuple[torch.Tensor, list[float], list[torch.Tensor]]:
    """Compute normalized AURC, i.e. subtract AURC of optimal CSF (given fixed risks)."""
    n = len(risks)
    # optimal confidence sorts risk. Asencding here because we start from coverage 1/n
    selective_risks = np.sort(risks).cumsum() / np.arange(1, n + 1)
    aurc_opt = selective_risks.sum() / n
    aurc_score, coverage, selective_risks = aurc(risks, confids)
    return aurc_score - aurc_opt, coverage, selective_risks
