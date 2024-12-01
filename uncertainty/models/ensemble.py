"""
Class representing an ensemble of arbitrary models.
"""

from typing import Callable

import torch
from torch import nn, vmap
from torch.func import functional_call, stack_module_state  # type: ignore


class DeepEnsemble(nn.Module):
    """
    An ensemble of models, produces multiple outputs from each member

    See also uncertainty.evaluation.ensemble_inference for a fucntional
    version for performing ensemble inference using a list of models.

    Parameters
    ----------
    model : Callable[..., nn.Module] | list[nn.Module]
        A callable PyTorch model class, or a list of PyTorch models
    n_members: int
        The number of models in the ensemble
    *args, **kwargs : list, dict
        Additional arguments to pass to the model_fn
    """

    def __init__(
        self,
        model: Callable[..., nn.Module] | list[nn.Module],
        n_members: int,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.models = (
            nn.ModuleList([model(*args, **kwargs) for _ in range(n_members)])
            if isinstance(model, Callable)
            else model
        )
        self.ensemble_params, self.ensemble_buffers = stack_module_state(self.models)  # type: ignore

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        # Vectorised form of the forward pass
        def call_model(params: dict, buffers: dict, x: torch.Tensor) -> torch.Tensor:
            return functional_call(self.models[0], (params, buffers), (x,))

        # each model should have different randomness (dropout)
        return vmap(call_model, in_dims=(0, 0, None), randomness="different")(
            self.ensemble_params, self.ensemble_buffers, x
        )
