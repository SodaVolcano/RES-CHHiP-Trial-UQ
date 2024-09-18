from torch import nn
import lightning as lit
from typing import override
import torch


class MCDropoutUNet(nn.Module):
    """
    Wrapper around U-Net to apply dropout during evaluation

    Note: this class only sets `nn.Dropout` layers to training mode during
    evaluation. It only passes the input through the model once so to
    get multiple predictions, use `uncertainty.training.inference.mc_dropout_inference`.

    Parameters
    ----------
    model : nn.Module | lit.LightningModule
        U-Net model
    """

    @override
    def __init__(self, model: nn.Module | lit.LightningModule):
        """
        Wrap an existing U-Net or create a new one using the provided configuration
        """
        super().__init__()
        self.model = model

    @override
    def forward(self, x: torch.Tensor, logits: bool = False) -> torch.Tensor:
        """
        Single forward pass for an input of shape (B, C, D, H, W)
        """
        return self.model(x, logits=logits)

    @override
    def eval(self):
        def activate_dropout(module):
            if isinstance(module, nn.Dropout):
                module.train(True)

        # Apply dropout during evaluation
        self.model.eval()
        self.model.apply(activate_dropout)
        return self
