from torch import nn, vmap
import torch
from torch.func import stack_module_state, functional_call  # type: ignore


class DeepEnsemble(nn.Module):
    def __init__(self, ensemble_size: int, model_fn: nn.Module, *args, **kwargs):
        """
        An ensemble of models

        Parameters
        ----------
        ensemble_size : int
            The number of models in the ensemble
        model_fn : nn.Module
            A callable PyTorch model class
        *args, **kwargs : list, dict
            Additional arguments to pass to the model_fn
        """
        super().__init__()
        self.models = [model_fn(*args, **kwargs) for _ in range(ensemble_size)]
        self.ensemble_params, self.ensemble_buffers = stack_module_state(self.models)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        def call_model(params: dict, buffers: dict, x: torch.Tensor) -> torch.Tensor:
            return functional_call(self.models[0], (params, buffers), (x,))

        # each model should have different randomness (dropout)
        return vmap(call_model, in_dims=(0, 0, None), randomness="different")(
            self.ensemble_params, self.ensemble_buffers, x
        )
