import os
from typing import Callable, Optional

import lightning as lit
from torchmetrics.classification import MultilabelF1Score
import numpy as np
import toolz as tz
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split, IterableDataset
from toolz import curried
import h5py as h5

from ..config import Configuration, configuration
from .augmentations import augmentations
from ..data.preprocessing import _preprocess_data_configurable


class H5DatasetPatch(IterableDataset):
    """
    Wrapper for H5Dataset that allows for patch-based training



    Parameters
    ----------
    h5_file_path : str
        Path to the H5 file containing the dataset.
    patch_size : tuple[int]
        Size of the patch to extract from the dataset.
    stride : tuple[int]
        Stride to move the patch window.
    transform : Callable[[tuple[np.ndarray, np.ndarray]], tuple[np.ndarray, np.ndarray]]
        Function to apply to the (x, y) pair. Default is the identity function.
        Intended to be used for data augmentation.
    config : Configuration
        Configuration object
    """

    def __init__(
        self,
        h5_file_path: str,
        patch_size: tuple[int],
        stride: tuple[int],
        transform: Callable[
            [tuple[np.ndarray, np.ndarray]], tuple[np.ndarray, np.ndarray]
        ] = tz.identity,
        config: Configuration = configuration(),
    ):
        self.h5_file_path = h5_file_path
        self.patch_size = patch_size
        self.stride = stride
        self.transform = transform
        self.config = config
        self.dataset = H5Dataset(h5_file_path, config=config, transform=transform)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        pass

    def __del__(self):
        del self.dataset


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
    ):
        self.h5_file_path = h5_file_path
        self.transform = transform
        with h5.File(h5_file_path, "r") as f:
            self.keys = list(f.keys())
        self.config = config
        self.h5_file = None

    def __len__(self) -> int:
        return len(self.keys)

    def __getitem__(self, index: int):
        # opened HDF5 is not pickleable, so don't open in __init__
        # open once to prevent overhead
        if self.h5_file is None:
            self.h5_file = h5.File(self.h5_file_path, "r")

        return tz.pipe(
            self.h5_file[self.keys[index]],
            # [:] change data from dataset to numpy array
            lambda group: (group["x"][:], group["y"][:]),
            _preprocess_data_configurable(config=self.config),
            self.transform,
            curried.map(
                # torch complains if using tensor(tensor, dtype...)
                lambda x: (
                    torch.tensor(x, dtype=torch.float32)
                    if isinstance(x, np.ndarray)
                    else x.to(torch.float32)
                )
            ),
            tuple,
        )

    def __del__(self):
        if self.h5_file is not None:
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

        # Apply augmentations to validation set too
        dataset = H5Dataset(self.fname, transform=augmentations(p=1))

        self.train, self.val = random_split(
            dataset, [1 - config["val_split"], config["val_split"]]
        )

    def train_dataloader(self):
        return DataLoader(
            self.train, batch_size=self.batch_size, shuffle=True, num_workers=2
        )

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, num_workers=2)

    def test_dataloader(self):
        pass


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
        PyTorch model to be trained
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
        class_weights: Optional[list[float]] = None,
    ):
        super().__init__()
        self.model = model
        self.config = config
        self.deep_supervision = (
            hasattr(model, "deep_supervision") and model.deep_supervision
        )
        self.class_weights = class_weights
        if class_weights is None:
            self.bce_fns = nn.ModuleList(
                [nn.BCEWithLogitsLoss()] * config["output_channel"]
            )
        else:
            self.bce_fns = nn.ModuleList(
                [
                    nn.BCEWithLogitsLoss(weight=torch.tensor(weight))
                    for weight in class_weights
                ]
            )
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
        # ignore the TWO lowest resolution outputs, [1:] as encoder already excludes bottleneck
        y_preds = y_preds[1:]
        ys = [nn.Upsample(size=y_pred.shape[2:])(y) for y_pred in y_preds]

        # Reverse to match y_preds; halves weights as resolution decreases
        weights = list(reversed([1 / (2**i) for i in range(len(y_preds))]))
        # normalise weights to sum to 1
        weights = [weight / sum(weights) for weight in weights]

        return torch.Tensor(
            sum(
                [
                    weight * self.calc_loss_single(y_pred, y_scaled)
                    for weight, y_pred, y_scaled in zip(weights, y_preds, ys)
                ]
            )
        )

    def calc_loss_single(self, y_pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute the sum of the binary cross entropy loss and the dice loss for one output
        """
        # Sums BCE loss for each class, y have shape (batch, n_classes, H, W, D)
        bce = tz.pipe(
            [self.bce_fns[i](y_pred[:, i], y[:, i]) for i in range(len(self.bce_fns))],
            sum,
            torch.Tensor,
        )
        dice = self.dice_fn(y_pred, y)
        return bce + (1 - dice)  # scale dice to match bce?

    def calc_loss(
        self, y_pred: torch.Tensor | list[torch.Tensor], y: torch.Tensor
    ) -> torch.Tensor:
        return (
            self.calc_loss_single(y_pred, y)
            if not self.deep_supervision and not isinstance(y_pred, list)
            else self.__calc_loss_deep_supervision(y_pred, y)  # type: ignore
        )

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
        lr_scheduler = self.config["lr_scheduler"](optimiser)  # type: ignore
        return {"optimizer": optimiser, "lr_scheduler": lr_scheduler}
