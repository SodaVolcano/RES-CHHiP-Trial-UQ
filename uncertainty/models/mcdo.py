"""
Class to enable dropout in any arbitrary model.
"""

from typing import override

import lightning as lit
import torch
from torch import nn


class MCDropout(nn.Module):
    """
    Enable dropout during inference for any model with dropout layer(s).

    See also `uncertainty.evaluation.mc_dropout_inference` which uses
    this class to perform Monte Carlo dropout inference.

    Parameters
    ----------
    model : nn.Module | lit.LightningModule
        Any model with dropout layers.
    """

    @override
    def __init__(self, model: nn.Module | lit.LightningModule):
        super().__init__()
        self.model = model

    @override
    def forward(self, x: torch.Tensor, logits: bool = False) -> torch.Tensor:
        return self.model(x, logits=logits)

    @override
    def eval(self):
        def activate_dropout(module):
            if isinstance(
                module, nn.Dropout | nn.Dropout2d | nn.Dropout3d | nn.Dropout1d
            ):
                module.train(True)

        # Apply dropout during evaluation
        self.model.eval()
        self.model.apply(activate_dropout)
        return self
