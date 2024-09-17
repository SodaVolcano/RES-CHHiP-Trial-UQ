import torch

from ..utils.wrappers import curry


def _stack_if_sequence(
    preds: torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor],
) -> torch.Tensor:
    """
    Stack predictions if they are a sequence
    """
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
    smooth: float = 1e-10,
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
