import os
from typing import Callable, Optional

import lightning as lit
from torchmetrics.classification import MultilabelF1Score
import numpy as np
import toolz as tz
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split
from toolz import curried
import h5py as h5

from ..config import Configuration, configuration
from .augmentations import augmentations
from ..data.preprocessing import _preprocess_data_configurable


class H5Dataset(Dataset):
    """
    PyTorch Dataset for loading (x, y) pairs from an H5 file.

    Parameters
    ----------
    h5_file_path : str
        Path to the H5 file containing the dataset.
    transform : Callable[[tuple[np.ndarray, np.ndarray]], tuple[np.ndarray, np.ndarray]]
        Function to apply to the (x, y) pair. Default is the identity function.
        Intended to be used for data augmentation.
    indices : Optional[list[int]]
        List of indices to load from the H5 file. Default is None, which loads
        all the data.

    Attributes
    ----------
    h5_file_path : str
        The file path of the H5 file.

    h5_file : h5py.File
        The opened H5 file object.

    keys : list of str
        List of keys corresponding to the dataset entries in the H5 file.
    """

    def __init__(
        self,
        h5_file_path: str,
        config: Configuration = configuration(),
        transform: Callable[
            [tuple[np.ndarray, np.ndarray]], tuple[np.ndarray, np.ndarray]
        ] = tz.identity,
        indices: Optional[list[int]] = None,
    ):
        self.h5_file_path = h5_file_path
        self.h5_file = h5.File(self.h5_file_path, "r")
        self.transform = transform
        self.keys = list(self.h5_file.keys())
        self.config = config
        if indices is not None:
            self.keys = [self.keys[i] for i in indices]

    def __len__(self) -> int:
        return len(self.keys)

    def __getitem__(self, index: int):
        return tz.pipe(
            self.h5_file[self.keys[index]],
            lambda group: (group["x"][:], group["y"][:]),
            _preprocess_data_configurable(config=self.config),
            self.transform,
            curried.map(torch.tensor),
            tuple,
        )

    def __del__(self):
        self.h5_file.close()


class SegmentationData(lit.LightningDataModule):
    """
    Wrapper class for PyTorch dataset to be used with PyTorch Lightning
    """

    def __init__(self, config: Configuration):
        super().__init__()
        self.config = config
        self.fname = os.path.join(config["staging_dir"], config["staging_fname"])
        self.batch_size = config["batch_size"]
        self.val_split = config["val_split"]
        self.augmentations = augmentations(p=1)

        dataset = H5Dataset(self.fname, transform=_preprocess_data_configurable(config))

        self.train, self.val = random_split(
            dataset, [1 - round(config["val_split"]), config["val_split"]]
        )

    def train_dataloader(self):
        return DataLoader(
            self.train, batch_size=self.batch_size, shuffle=True, num_workers=2
        )

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=len(self.val), num_workers=2)

    def test_dataloader(self):
        pass


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
        self, y_preds: list[torch.Tensor], ys: torch.Tensor
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
        ys = torch.tensor(
            [nn.Upsample(size=y_pred.shape)(y) for y, y_pred in zip(ys, y_preds)]
        )

        weights = [1 / (2**i) for i in range(len(y_preds))]  # halve for each level
        # normalise weights to sum to 1
        weights = [weight / sum(weights) for weight in weights]

        return torch.Tensor(
            sum(
                [
                    weight * self.calc_loss(y_pred, ys)
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
