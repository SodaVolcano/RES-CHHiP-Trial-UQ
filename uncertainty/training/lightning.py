import inspect
import os
from typing import Optional


import dill
from ..constants import ORGAN_MATCHES
import torch.utils
from pytorch_lightning import seed_everything
import time
import lightning as lit
import torch
from torch import nn, vmap
from torch.utils.data import DataLoader, random_split
from torchmetrics.aggregation import RunningMean
from torchmetrics.classification import MultilabelF1Score

from uncertainty.training.datasets import (
    H5Dataset,
    RandomPatchDataset,
    SlidingPatchDataset,
)

from .loss import DeepSupervisionLoss, DiceBCELoss
from ..config import Configuration
from .augmentations import augmentations


def _seed_with_time(id: int):
    seed_everything(int(time.time() + (id * 3)), verbose=False)


def __dump_tensors(
    x: torch.Tensor,
    y: torch.Tensor,
    y_pred: torch.Tensor,
    dice: torch.Tensor,
    loss: torch.Tensor,
    val_counter: int,
    ensemble_id: int = 0,
):
    name = os.path.join("./batches", f"batch_{val_counter}_member_{ensemble_id}.pt")
    torch.save({"x": x, "y": y, "y_pred": y_pred, "dice": dice, "loss": loss}, name)


class SegmentationData(lit.LightningDataModule):
    """
    Wrapper class for PyTorch dataset to be used with PyTorch Lightning
    """

    def __init__(
        self,
        config: Configuration,
        checkpoint_path: str,
        train_val_indices: tuple[list[int], list[int]] | None = None,
    ):
        super().__init__()
        self.config = config
        self.train_fname = os.path.join(config["staging_dir"], config["train_fname"])
        self.test_fname = os.path.join(config["staging_dir"], config["test_fname"])
        self.augmentations = augmentations(p=1)

        indices = list(range(len(H5Dataset(self.train_fname))))
        if train_val_indices:
            self.train_indices, self.val_indices = train_val_indices
        else:
            self.train_indices, self.val_indices = random_split(
                indices, [1 - config["val_split"], config["val_split"]]  # type: ignore
            )
            torch.save(
                {
                    "train_indices": list(self.train_indices),  # type: ignore
                    "val_indices": list(self.val_indices),  # type: ignore
                },
                os.path.join(checkpoint_path, "indices.pt"),
            )

    def train_dataloader(self):
        return DataLoader(
            RandomPatchDataset(
                self.train_fname,
                self.config,
                self.train_indices,
                self.config["batch_size"],
                patch_size=self.config["patch_size"],
                fg_oversample_ratio=self.config["foreground_oversample_ratio"],
                transform=self.augmentations,
            ),
            num_workers=8,
            batch_size=self.config["batch_size"],
            prefetch_factor=1,
            # persistent_workers=True,
            pin_memory=True,
            worker_init_fn=_seed_with_time,
        )

    def val_dataloader(self):
        return DataLoader(
            RandomPatchDataset(
                self.train_fname,
                self.config,
                self.val_indices,
                self.config["batch_size"],
                self.config["patch_size"],
                fg_oversample_ratio=self.config["foreground_oversample_ratio"],
            ),
            num_workers=8,
            batch_size=self.config["batch_size_eval"],
            prefetch_factor=1,
            pin_memory=True,
            # persistent_workers=True,
            worker_init_fn=_seed_with_time,
        )

    def test_dataloader(self):
        return DataLoader(
            SlidingPatchDataset(
                self.test_fname,
                self.config,
                self.config["patch_size"],
                self.config["patch_step"],
            ),
            num_workers=4,
            batch_size=self.config["batch_size_eval"],
            prefetch_factor=3,
            persistent_workers=True,
            pin_memory=True,
        )


class LitSegmentation(lit.LightningModule):
    """
    Wrapper class for PyTorch model to be used with PyTorch Lightning

    If deep supervision is enabled, then for a U-Net with n levels, a loss is calculated
    for each level and summed as
        L = w1 * L1 + w2 * L2 + ... + wn * Ln
    Where the weights halve for each level and are normalised to sum to 1.
    Output from the two levels in the lowest resolution are not used.
    SEE https://arxiv.org/abs/1809.10486

    Parameters
    ----------
    model : nn.Module
        PyTorch model to be trained. If training an ensemble, the ensemble
        members are created using the same model class and the same constructor
        arguments.
    config : Configuration
        Configuration object
    class_weights : Optional[list[float]]
        Weights for each class in the loss function. Default is None.
        Weights are typically calculated using the number of pixels as
            n_background / n_foreground
    """

    def __init__(
        self,
        model: nn.Module,
        config: Configuration,
        class_weights: Optional[torch.Tensor] = None,
        save_hyperparams: bool = True,
    ):
        super().__init__()
        if save_hyperparams:
            self.save_hyperparameters(ignore=["model"])
            with open(
                os.path.join(config["model_checkpoint_path"], "config.pkl"), "wb"
            ) as f:
                dill.dump(config, f)

        self.config = config
        self.class_weights = class_weights
        # Original dice, used for evaluation
        self.dice_eval = MultilabelF1Score(num_labels=config["output_channel"])
        self.running_loss = RunningMean(window=10)
        self.running_dice = RunningMean(window=10)
        self.val_counter = 0

        self.model = model
        self.deep_supervision = (
            hasattr(model, "deep_supervision") and model.deep_supervision
        )
        if self.deep_supervision:
            self.loss = DeepSupervisionLoss(DiceBCELoss(class_weights, logits=True))
        else:
            self.loss = DiceBCELoss(class_weights, logits=True)

    def __dump_tensors(self, x, y, y_pred, dice, loss, val_counter):
        name = os.path.join("./batches", f"batch_{val_counter}.pt")
        torch.save({"x": x, "y": y, "y_pred": y_pred, "dice": dice, "loss": loss}, name)

    def forward(self, x, logits: bool = False):
        return self.model(x, logits)

    def training_step(self, batch: torch.Tensor):
        x, y = batch
        y_pred = self.model(x, logits=True)
        loss = self.loss(y_pred, y)
        self.__dump_tensors(x, y, y_pred, 0, loss, self.val_counter)
        self.val_counter += 1

        if self.deep_supervision:
            y_pred = y_pred[-1]
        dice = self.dice_eval(y_pred, y)

        self.running_dice(dice.detach())
        self.running_loss(loss.detach())
        self.log(
            "train_dice",
            self.running_dice.compute(),
            sync_dist=True,
            prog_bar=True,
        )
        self.log(
            "train_loss",
            self.running_loss.compute(),
            sync_dist=True,
            prog_bar=True,
        )
        return loss

    @torch.no_grad()
    def validation_step(self, batch: torch.Tensor):
        x, y = batch

        y_pred = self.model(x, logits=True)
        loss = self.loss(y_pred, y)
        if self.deep_supervision:
            y_pred = y_pred[-1]
        dice = self.dice_eval(y_pred, y)

        # Get dice of each organ
        for channel, organ_name in zip(range(y_pred.shape[1]), ORGAN_MATCHES.keys()):
            with torch.no_grad():
                organ_dice = self.dice_eval(
                    y_pred[:, channel : channel + 1, ...],
                    y[:, channel : channel + 1, ...],
                )
                self.log(
                    f"val_dice_{organ_name}",
                    organ_dice,
                    sync_dist=True,
                    prog_bar=False,
                )

        if self.val_counter % 10:
            __dump_tensors(x, y, y_pred, dice, loss, self.val_counter)
        self.val_counter += 1

        self.log(
            "val_loss",
            loss,
            sync_dist=True,
            prog_bar=True,
        )
        self.log(
            "val_dice",
            dice,
            sync_dist=True,
            prog_bar=True,
        )
        return loss

    @torch.no_grad()
    def test_step(self, batch: torch.Tensor):
        x, y = batch
        y_pred = self.model(x, logits=False)
        loss = self.loss(y_pred, y)
        if self.deep_supervision:
            y_pred = y_pred[-1]
        dice = self.dice_eval(y_pred, y)

        self.log("test_loss", loss, sync_dist=True, prog_bar=True)
        self.log("test_dice", dice, sync_dist=True, prog_bar=True)
        return loss

    def configure_optimizers(self):  # type: ignore
        optimiser = self.config["optimiser"](
            self.model.parameters(), **self.config["optimiser_kwargs"]
        )
        lr_scheduler = self.config["lr_scheduler"](optimiser)  # type: ignore
        return {"optimizer": optimiser, "lr_scheduler": lr_scheduler}
