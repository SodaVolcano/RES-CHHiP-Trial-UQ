from typing import Callable, Optional

import lightning as lit
from torchmetrics.classification import MultilabelF1Score
import numpy as np
import toolz as tz
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split

from ..config import Configuration
from ..data.patient_scan import PatientScan, from_h5_dir
from .data_handling import augmentations, preprocess_dataset


class PatientScanDataset(Dataset):
    def __init__(
        self,
        patient_scans: list["PatientScan"],
        transform: Optional[
            Callable[[tuple[np.ndarray, np.ndarray]], tuple[np.ndarray, np.ndarray]]
        ] = tz.identity,
    ):
        """
        Parameters
        ----------
        patient_scans : list[PatientScan]
            List of PatientScan objects to be converted to
            (volume, masks) pairs. The volume and masks will have shape
            (H, W, D, C) where C is the number of channels which
            PatientScanDataset will convert to (C, H, W, D).
        transform : Optional[Callable]
            A function that take in a (volume, masks) pair and returns a new
            (volume, masks) pair. Default is the identity function.
        buffer_size : int
            Size of the buffer used by the Shuffler to randomly shuffle the dataset.
            Set to 1 to disable shuffling.
        """
        super().__init__()
        self.data: list[tuple[np.ndarray, np.ndarray]] = preprocess_dataset(
            patient_scans
        )
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return tz.pipe(
            self.data[index],
            self.transform,
            lambda vol_mask: (
                # float16 to reduce memory usage
                torch.tensor(vol_mask[0], dtype=torch.float16),
                torch.tensor(vol_mask[1], dtype=torch.float16),
            ),
        )  # type: ignore


class LitSegmentation(lit.LightningModule):
    """
    Wrapper class for PyTorch model to be used with PyTorch Lightning
    """

    def __init__(
        self,
        model: nn.Module,
        deep_supervision: bool,
        config: Configuration,
        class_weights: Optional[list[float]] = None,
    ):
        """
        Parameters
        ----------
        model : nn.Module
            PyTorch model to be trained
        deep_supervision : bool
            Whether to use deep supervision. If enabled, for each output
            from the levels of the U-Net, a loss is calculated and summed
            as
                L = w1 * L1 + w2 * L2 + ... + wn * Ln
            Where the weights halve for each level and are normalised to sum to 1.
            Output from the two levels in the lowest resolution are not used.
            SEE https://arxiv.org/abs/1809.10486
        config : Configuration
            Configuration object
        class_weights : Optional[list[float]]
            Weights for each class in the loss function. Default is None.
            Weights are typically calculated using the number of pixels as
                n_background / n_foreground
        """
        super().__init__()
        self.model = model
        self.config = config
        self.deep_supervision = deep_supervision
        self.class_weights = class_weights
        if class_weights is None:
            self.loss_fns = [nn.BCEWithLogitsLoss()] * config["output_channel"]
        else:
            self.bce_fns = [
                nn.BCEWithLogitsLoss(weight=torch.tensor(weight))
                for weight in class_weights
            ]
        self.dice_fn = MultilabelF1Score(num_labels=config["output_channel"])

    def forward(self, x):
        return self.model(x)

    def __calc_loss_deep_supervision(
        self, y_preds: list[torch.Tensor], y: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate the loss for each level of the U-Net

        Parameters
        ----------
        y_preds : list[torch.Tensor]
            List of outputs from the U-Net ordered from the lowest resolution
            to the highest resolution
        """
        # ignore the two lowest resolution outputs and order from highest to lowest
        y_preds = y_preds[-3::-1]
        weights = [1 / (2**i) for i in range(len(y_preds))]
        # normalise weights to sum to 1
        weights = [weight / sum(weights) for weight in weights]
        return torch.Tensor(
            sum(
                [
                    weight * self.calc_loss(y_pred, y)
                    for weight, y_pred in zip(weights, y_preds)
                ]
            )
        )

    def calc_loss(self, y_pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute the sum of the binary cross entropy loss and the dice loss
        """
        # Sums BCE loss for each class, y have shape (batch, n_classes, H, W, D)
        bce = tz.pipe(
            [
                self.loss_fns[i](y_pred[:, i], y[:, i])
                for i in range(len(self.loss_fns))
            ],
            sum,
            torch.Tensor,
        )
        dice = self.dice_fn(y_pred, y)
        return bce + (1 - dice)  # scale dice to match bce?

    def training_step(self, batch: torch.Tensor):
        x, y = batch
        y_pred = self.model(x)
        loss = (
            self.calc_loss(y_pred, y)
            if not self.deep_supervision
            else self.__calc_loss_deep_supervision(y_pred, y)
        )

        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch: torch.Tensor):
        x, y = batch
        y_pred = self.model(x)
        loss = self.calc_loss(y_pred, y)

        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):  # type: ignore
        optimiser = self.config["optimiser"](
            self.model.parameters(), **self.config["optimiser_kwargs"]
        )
        lr_scheduler = self.config["lr_scheduler"](self.optimizer)
        return {"optimizer": optimiser, "lr_scheduler": lr_scheduler}


class SegmentationData(lit.LightningDataModule):
    """
    Wrapper class for PyTorch dataset to be used with PyTorch Lightning

    WARNING: Only loads h5 files
    """

    def __init__(self, config: Configuration):
        super().__init__()
        self.config = config
        self.data_dir = config["staging_dir"]
        self.batch_size = config["batch_size"]
        self.val_split = config["val_split"]
        self.augmentations = augmentations(p=1)

        scans = list(filter(lambda x: x is not None, from_h5_dir(self.data_dir)))
        self.train, self.val = random_split(
            scans, (1 - config["val_split"]) * len(scans), config["val_split"] * len(scans)  # type: ignore
        )

    def train_dataloader(self):
        dataset = PatientScanDataset(list(self.train), transform=self.augmentations)  # type: ignore
        return DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True, num_workers=2
        )

    def val_dataloader(self):
        val = list(self.val)  # type: ignore
        dataset = PatientScanDataset(val)
        return DataLoader(dataset, batch_size=len(val), num_workers=2)

    def test_dataloader(self):
        pass