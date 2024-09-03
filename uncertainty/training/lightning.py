import inspect
import os
from typing import Optional


from torch.func import functional_call, stack_module_state  # type: ignore
import toolz as tz
from toolz import curried
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

    def __init__(self, ensemble_size: int, config: Configuration):
        super().__init__()
        self.config = config
        self.train_fname = os.path.join(config["staging_dir"], config["train_fname"])
        self.test_fname = os.path.join(config["staging_dir"], config["test_fname"])
        self.augmentations = augmentations(p=1)
        self.ensemble_size = ensemble_size

        indices = list(range(len(H5Dataset(self.train_fname))))
        self.train_indices, self.val_indices = random_split(
            indices, [1 - config["val_split"], config["val_split"]]  # type: ignore
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
                ensemble_size=self.ensemble_size,
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
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        self.config = config
        self.class_weights = class_weights
        # Original dice, used for evaluation
        self.dice_eval = MultilabelF1Score(num_labels=config["output_channel"])
        self.running_loss = RunningMean(window=10)
        self.val_counter = 0

        self.model = model
        self.deep_supervision = (
            hasattr(model, "deep_supervision") and model.deep_supervision
        )
        if self.deep_supervision:
            self.loss = DeepSupervisionLoss(DiceBCELoss(class_weights, logits=True))
        else:
            self.loss = DiceBCELoss(class_weights, logits=True)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch: torch.Tensor):
        x, y = batch
        y_pred = self.model(x, logits=True)
        loss = self.loss(y_pred, y)

        self.running_loss(loss.detach())
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

        # TODO: get dice of each organ

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


class LitDeepEnsemble(lit.LightningModule):
    """
    Train an ensemble of models.

    `config["n_batches"]` is the number of batches to train for.
    """

    def __init__(
        self,
        model,
        ensemble_size: int,
        config: Configuration,
        class_weights: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        self.class_weights = class_weights
        self.ensemble_size = ensemble_size

        self.models = self.__init_ensemble(model)
        self.ensemble_params, self.ensemble_buffers = stack_module_state(self.models)
        self.deep_supervision = (
            hasattr(model, "deep_supervision") and model.deep_supervision
        )
        if self.deep_supervision:
            self.loss = DeepSupervisionLoss(DiceBCELoss(class_weights, logits=True))
        else:
            self.loss = DiceBCELoss(class_weights, logits=True)
        self.dice_eval = MultilabelF1Score(num_labels=config["output_channel"])
        self.running_loss = RunningMean(window=10)
        self.val_counter = 0

        self.automatic_optimization = False  # activate manual optimization

    def __init_ensemble(self, model) -> nn.ModuleList:
        cls = type(model)
        return tz.pipe(
            cls,
            lambda cls: inspect.signature(cls.__init__).parameters,
            curried.keyfilter(lambda k: k != "self"),
            lambda sig: sig.values(),
            curried.map(lambda param: getattr(model, param.name)),
            tuple,
            lambda args: nn.ModuleList([cls(*args) for _ in range(self.ensemble_size)]).to("cuda"),
        )  # type: ignore

    def forward(self, x, logits=True):
        def call_model(params: dict, buffers: dict, x: torch.Tensor) -> torch.Tensor:
            return functional_call(self.models[0], (params, buffers), (x, logits))
        # each model should have different randomness (dropout)
        return vmap(call_model, in_dims=(0, 0, None), randomness="different")(
            self.ensemble_params, self.ensemble_buffers, x
        )

    def optimise_single_model(
        self,
        y: torch.Tensor,
        y_pred: torch.Tensor,
        optimiser: nn.Module,
        lr_scheduler: nn.Module,
    ) -> torch.Tensor:
        loss = self.loss(y_pred, y)
        optimiser.zero_grad()
        self.manual_backward(loss)
        optimiser.step()
        lr_scheduler.step()
        return loss

    def training_step(self, batch: torch.Tensor):
        optimisers, lr_schedulers = self.optimizers()  # type: ignore
        x, y = batch
        y_preds = torch.split(self.forward(x), x.shape[0] // self.ensemble_size, dim=0)
        ys = torch.split(y, x.shape[0] // self.ensemble_size, dim=0)
        loss_dict = {
            f"loss_{i}": self.optimise_single_model(y, y_pred, optimiser, lr_scheduler)
            for i, (y, y_pred, optimiser, lr_scheduler) in enumerate(
                zip(ys, y_preds, optimisers, lr_schedulers)
            )
        }
        self.log_dict(loss_dict, prog_bar=False)
        self.running_loss(sum(loss_dict.values().detach()) / len(loss_dict))
        self.log(
            "train_loss",
            self.running_loss.compute(),
            sync_dist=True,
            prog_bar=True,
        )

    @torch.no_grad()
    def validation_step(self, batch: torch.Tensor):
        x, y = batch
        y_preds = self.forward(torch.repeat_interleave(x, self.ensemble_size, dim=0))

        loss_dict = {
            f"val_loss_{i}": self.loss(y_pred, y) for i, y_pred in enumerate(y_preds)
        }
        if self.deep_supervision:
            y_preds = [y_pred[-1] for y_pred in y_preds]
        dice_dict = {
            f"val_dice_{i}": self.dice_eval(y_pred, y)
            for i, y_pred in enumerate(y_preds)
        }
        if self.val_counter % 10:
            [
                __dump_tensors(x, y, y_pred, dice, loss, self.val_counter, i)
                for i, (y_pred, dice, loss) in enumerate(
                    zip(y_preds, dice_dict.values(), loss_dict.values())
                )
            ]
            self.val_counter += 1

        self.log_dict(loss_dict, prog_bar=False)
        self.log_dict(dice_dict, prog_bar=False)
        self.log(
            "val_loss",
            sum(loss_dict.values()) / len(loss_dict),
            sync_dist=True,
            prog_bar=True,
        )
        self.log(
            "val_dice",
            sum(dice_dict.values()) / len(dice_dict),
            sync_dist=True,
            prog_bar=True,
        )

        return loss_dict

    @torch.no_grad()
    def test_step(self, batch: torch.Tensor):
        x, y = batch
        y_preds = self.forward(torch.repeat_interleave(x, self.ensemble_size, dim=0))

        loss_dict = {
            f"test_loss_{i}": self.loss(y_pred, y) for i, y_pred in enumerate(y_preds)
        }
        if self.deep_supervision:
            y_preds = [y_pred[-1] for y_pred in y_preds]
        dice_dict = {
            f"test_dice_{i}": self.dice_eval(y_pred, y)
            for i, y_pred in enumerate(y_preds)
        }

        self.log_dict(loss_dict, prog_bar=False)
        self.log_dict(dice_dict, prog_bar=False)
        self.log(
            "test_loss",
            sum(loss_dict.values()) / len(loss_dict),
            sync_dist=True,
            prog_bar=True,
        )
        self.log(
            "test_dice",
            sum(dice_dict.values()) / len(dice_dict),
            sync_dist=True,
            prog_bar=True,
        )

        return loss_dict

    def configure_optimizers(self):  # type: ignore
        optimiser_fn = lambda params: self.config["optimiser"](
            params, **self.config["optimiser_kwargs"]
        )
        opts = [optimiser_fn(model.parameters()) for model in self.models]
        lr_schedulers = [self.config["lr_scheduler"](optimiser) for optimiser in opts]  # type: ignore
        return opts, lr_schedulers
