"""
An ensemble of models
"""

from typing import Callable
from torch import nn, vmap
import torch
from torch.func import stack_module_state, functional_call  # type: ignore


class DeepEnsemble(nn.Module):
    """
    An ensemble of models, produces multiple outputs from each member
    """

    def __init__(
        self,
        model: Callable[..., nn.Module] | list[nn.Module],
        ensemble_size: int,
        *args,
        **kwargs
    ):
        """
        An ensemble of models

        Parameters
        ----------
        ensemble_size : int
            The number of models in the ensemble
        model : Callable[..., nn.Module] | list[nn.Module]
            A callable PyTorch model class, or a list of PyTorch models
        *args, **kwargs : list, dict
            Additional arguments to pass to the model_fn
        """
        super().__init__()
        self.models = (
            [model(*args, **kwargs) for _ in range(ensemble_size)]
            if isinstance(model, Callable)
            else model
        )
        self.ensemble_params, self.ensemble_buffers = stack_module_state(self.models)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        def call_model(params: dict, buffers: dict, x: torch.Tensor) -> torch.Tensor:
            return functional_call(self.models[0], (params, buffers), (x,))

        # each model should have different randomness (dropout)
        return vmap(call_model, in_dims=(0, 0, None), randomness="different")(
            self.ensemble_params, self.ensemble_buffers, x
        )
