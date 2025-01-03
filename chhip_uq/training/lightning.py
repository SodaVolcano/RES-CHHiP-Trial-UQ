"""
Lightning modules for training models, defining training and evaluation steps.
"""

import os
from pathlib import Path

import lightning as lit
import torch
from torch import nn
from torchmetrics.aggregation import RunningMean

from .. import constants as c
from ..config import auto_match_config
from ..metrics.classification import dice_batched


def _dump_tensors(
    path: str,
    x: torch.Tensor,
    y: torch.Tensor,
    y_pred: torch.Tensor,
    dice: torch.Tensor,
    loss: torch.Tensor,
    epoch: int,
):
    name = Path(path) / f"epoch-{epoch}.pt"
    if not name.exists():
        torch.save({"x": x, "y": y, "y_pred": y_pred, "dice": dice, "loss": loss}, name)


class LitModel(lit.LightningModule):
    """
    Wrapper class for PyTorch models defining training and evaluation steps.

    The PyTorch model must have the following attributes:
    - `loss`: a loss function that takes `y_pred` and `y` as arguments.
    - `optimiser`: a PyTorch optimiser.
    - `lr_scheduler`: a PyTorch learning rate scheduler.
    - `deep_supervision`: a boolean indicating whether the model uses deep supervision.

    Additionally, the model must have a `forward` method that takes an input tensor `x`
    and an optional argument `logits` that determines whether the output should be
    logits or probabilities.

    Parameters
    ----------
    model : nn.Module
        PyTorch model to be trained. If training an ensemble, the ensemble
        members are created using the same model class and the same constructor
        arguments.
    class_names : list[str]
        List of names for each class in the dataset used to log classwise dice scores.
    running_loss_window : int
        Size of the window for the running mean of the loss.
    save_hyperparams : bool
        Whether to save the hyperparameters as `hparams` attribute.
    dump_tensors_every_n_epoch : int
        If greater than 0, dump x, y, and predictions to disk every n epochs.
    tensor_dump_dir: str
        Directory to save the dumped tensors.
    """

    @auto_match_config(prefixes=["training"])
    def __init__(
        self,
        model: nn.Module,
        class_names: list[str] = list(c.ORGAN_MATCHES.keys()),
        running_loss_window: int = 10,
        save_hyperparams: bool = True,
        dump_tensors_every_n_epoch: int = 0,
        tensor_dump_dir: str = "tensor-dump",
    ):
        super().__init__()
        if save_hyperparams:
            self.save_hyperparameters(ignore=["model"])
        if dump_tensors_every_n_epoch > 0:
            os.makedirs(tensor_dump_dir, exist_ok=True)

        self.model = model
        self.class_names = class_names
        self.dump_tensors_every_n_epoch = dump_tensors_every_n_epoch
        self.tensor_dump_dir = f"{tensor_dump_dir}/{self.__class__.__name__}"
        if self.dump_tensors_every_n_epoch > 0:
            os.makedirs(self.tensor_dump_dir, exist_ok=True)

        self.dice = dice_batched
        self.dice_classwise = dice_batched(average="none")
        self.running_loss = RunningMean(window=running_loss_window)
        self.running_dice = RunningMean(window=running_loss_window)

    def forward(self, x, logits: bool = False):
        return self.model(x, logits)

    def training_step(self, batch: torch.Tensor):
        x, y = batch
        y_pred = self.model(x, logits=True)
        loss = self.model.loss(y_pred, y, logits=True)

        if self.model.deep_supervision:
            y_pred = y_pred[-1]

        dice = self.dice(y_pred, y)

        self.running_dice(dice.detach())
        self.running_loss(loss.detach())
        self.log(
            "train_dice", self.running_dice.compute(), sync_dist=True, prog_bar=True
        )
        self.log(
            "train_loss", self.running_loss.compute(), sync_dist=True, prog_bar=True
        )

        if (
            self.dump_tensors_every_n_epoch > 0
            and self.current_epoch % self.dump_tensors_every_n_epoch == 0
            and self.current_epoch > 0
        ):
            _dump_tensors(
                self.tensor_dump_dir, x, y, y_pred, dice, loss, self.current_epoch
            )

        return loss

    @torch.no_grad()
    def validation_step(self, batch: torch.Tensor):
        x, y = batch

        y_pred = self.model(x, logits=True)
        loss = self.model.loss(y_pred, y, logits=True)

        dice = self.dice(y_pred, y)
        dice_classwise = self.dice_classwise(y_pred, y)

        for name, class_dice in zip(self.class_names, dice_classwise):
            self.log(f"val_dice_{name}", class_dice, sync_dist=True, prog_bar=False)

        self.log("val_loss", loss, sync_dist=True, prog_bar=True)
        self.log("val_dice", dice, sync_dist=True, prog_bar=True)

        return loss

    @torch.no_grad()
    def test_step(self, batch: torch.Tensor):
        x, y = batch
        y_pred = self.model(x, logits=False)
        loss = self.model.loss(y_pred, y, logits=False)
        if self.deep_supervision:
            y_pred = y_pred[-1]

        dice = self.dice(y_pred, y)

        self.log("test_loss", loss, sync_dist=True, prog_bar=True)
        self.log("test_dice", dice, sync_dist=True, prog_bar=True)

        return loss

    def configure_optimizers(self):  # type: ignore
        return {
            "optimizer": self.model.optimiser,
            "lr_scheduler": self.model.lr_scheduler,
        }
