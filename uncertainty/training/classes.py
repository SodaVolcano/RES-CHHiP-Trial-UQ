from itertools import islice, cycle
import os
from typing import Callable, Iterable, Optional
import random

import lightning as lit
from torchmetrics.classification import MultilabelF1Score
import numpy as np
import toolz as tz
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split, Subset, IterableDataset
from toolz import curried
import h5py as h5
from skimage.util import view_as_windows
from torchmetrics.aggregation import RunningMean

from ..config import Configuration, configuration
from .augmentations import augmentations
from ..data.preprocessing import _preprocess_data_configurable


class SlidingPatchDataset(IterableDataset):
    """
    Produce rolling window of patches from a dataset of 3D images.
    """

    def __init__(
        self,
        dataset: "H5Dataset",
        patch_size: tuple[int, int, int],
        patch_step: int,
        transform: Callable[
            [tuple[np.ndarray, np.ndarray]], tuple[np.ndarray, np.ndarray]
        ] = tz.identity,
    ):
        self.dataset = dataset
        self.transform = transform
        self.patch_size = patch_size
        self.patch_step = patch_step

    def __patch_iter(self, idx: int):
        """
        Return an iterator of patches for data at index `idx`
        """
        x_patches, y_patches = tz.pipe(
            self.dataset[idx],
            curried.map(lambda arr: arr.numpy()),
            curried.map(
                lambda arr: view_as_windows(
                    arr, (arr.shape[0], *self.patch_size), self.patch_step
                )
            ),
            tuple,
        )  # type: ignore
        for idx in np.ndindex(x_patches.shape[:-4]):  # type: ignore
            yield torch.tensor(x_patches[idx]), torch.tensor(y_patches[idx])  # type: ignore

    def __iter__(self) -> Iterable[tuple[np.ndarray, np.ndarray]]:  # type: ignore
        """
        Return iterator of patches over all data in the dataset
        """
        return tz.pipe(
            range(len(self.dataset)),
            curried.map(self.__patch_iter),
            tz.concat,
        )  # type: ignore

    def __del__(self):
        del self.dataset


class RandomPatchDataset(IterableDataset):
    """
    Randomly sample patches from a dataset of 3D images.

    Given a `batch_size`, each `batch_size` chunk of data returned
    by the iterator is guaranteed to have `max(1, fg_oversample_ratio)`
    number of patches with at least one foreground class in it.

    Parameters
    ----------
    dataset:

    batch_size:
        Number of patches in a single batch. Each batch is guaranteed
        to have a certain number of patches with a foreground class (oversampling)
    patch_size : tuple[int]
        Size of the patch to extract from the dataset.
    transform : Callable[[tuple[np.ndarray, np.ndarray]], tuple[np.ndarray, np.ndarray]]
        Function to apply to the (x, y) pair. Default is the identity function.
        Intended to be used for data augmentation.
    config : Configuration
        Configuration object
    """

    def __init__(
        self,
        dataset: "H5Dataset | Subset",
        batch_size: int,
        patch_size: tuple[int, int, int],
        fg_oversample_ratio: float,
        transform: Callable[
            [tuple[np.ndarray, np.ndarray]], tuple[np.ndarray, np.ndarray]
        ] = tz.identity,
    ):
        self.dataset = dataset
        self.transform = transform
        self.patch_size = patch_size
        self.fg_ratio = fg_oversample_ratio
        self.batch_size = batch_size

        # Force at least 1 sample have foreground
        n_fg_samples = max(1, int(round(self.batch_size * self.fg_ratio)))
        self.data_stream = tz.pipe(
            (
                self.__fg_patch_iter(n_fg_samples),
                self.__random_patch_iter(self.batch_size - n_fg_samples),
            ),
            tz.concat,
            cycle,
        )

    def __patch_iter(self):
        """
        Return an infinite stream of randomly augmented and sampled patches
        """
        while True:
            yield self.__sample_patch()

    def __sample_patch(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Randomly pick and augment a (x, y) pair from the dataset and sample a patch from it.
        """
        x, y = self.dataset[np.random.randint(len(self.dataset))]  # type: ignore
        x, y = self.transform((x, y))  # type: ignore
        h_coord, w_coord, d_coord = [
            np.random.randint(0, dim - patch_dim + 1)
            for dim, patch_dim in zip(x.shape[1:], self.patch_size)  # type: ignore
        ]

        def _extract_patch(arr: np.ndarray) -> np.ndarray:
            return arr[
                :,
                h_coord : h_coord + self.patch_size[0],
                w_coord : w_coord + self.patch_size[1],
                d_coord : d_coord + self.patch_size[2],
            ]

        return (_extract_patch(x), _extract_patch(y))  # type: ignore

    def __fg_patch_iter(self, length: int):
        """
        Iterator of `length` patches guaranteed to contain a foreground
        """
        return tz.pipe(
            self.__patch_iter(),
            curried.filter(lambda x: torch.any(x[1])),
            lambda x: islice(x, length),
        )

    def __random_patch_iter(self, length: int):
        """
        Iterator of `length` patches randomly sampled
        """
        return islice(self.__patch_iter(), length)

    def __iter__(self):
        """
        Return infinite stream of sampled patches

        Each `batch_size` is guaranteed to have at least one patch with the
        foreground class, and number of such patch in the batch is guaranteed
        to be `round(fg_oversample_ratio * batch_size)`
        """
        buffer = []
        while True:
            if len(buffer) == 0:
                buffer = list(islice(self.data_stream, self.batch_size))
                random.shuffle(buffer)
            yield buffer.pop()

    def __del__(self):
        del self.dataset


class H5Dataset(Dataset):
    """
    PyTorch Dataset for loading (x, y) pairs from an H5 file.

    Parameters
    ----------
    h5_file_path : str
        Path to the H5 file containing the dataset.

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
    ):
        self.h5_file_path = h5_file_path
        self.dataset = None
        with h5.File(h5_file_path, "r") as f:
            self.keys = list(f.keys())
        self.config = config

    def __len__(self) -> int:
        return len(self.keys)

    def __getitem__(self, index: int):
        # opened HDF5 is not pickleable, so don't open in __init__
        # open once to prevent overhead
        if self.dataset is None:
            self.dataset = h5.File(self.h5_file_path, "r")

        return tz.pipe(
            self.dataset[self.keys[index]],
            # [:] change data from dataset to numpy array
            lambda group: (group["x"][:], group["y"][:]),
            _preprocess_data_configurable(config=self.config),
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
        if self.dataset is not None:
            self.dataset.close()


class SegmentationData(lit.LightningDataModule):
    """
    Wrapper class for PyTorch dataset to be used with PyTorch Lightning
    """

    def __init__(self, config: Configuration):
        super().__init__()
        self.config = config
        self.train_fname = os.path.join(config["staging_dir"], config["train_fname"])
        self.test_fname = os.path.join(config["staging_dir"], config["test_fname"])
        self.augmentations = augmentations(p=1)

        self.test = H5Dataset(self.test_fname)

        train = H5Dataset(self.train_fname)
        self.train, self.val = random_split(
            train, [1 - config["val_split"], config["val_split"]]
        )

    def train_dataloader(self):
        return DataLoader(
            RandomPatchDataset(
                self.train,
                self.config["batch_size"],
                patch_size=self.config["patch_size"],
                fg_oversample_ratio=self.config["foreground_oversample_ratio"],
                transform=self.augmentations,
            ),
            num_workers=4,
            batch_size=self.config["batch_size"],
            prefetch_factor=3,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            RandomPatchDataset(
                self.val,
                self.config["batch_size"],
                self.config["patch_size"],
                fg_oversample_ratio=self.config["foreground_oversample_ratio"],
            ),
            num_workers=4,
            batch_size=self.config["batch_size_eval"],
            prefetch_factor=2,
            persistent_workers=True,
        )

    def test_dataloader(self):
        return DataLoader(
            SlidingPatchDataset(
                self.test,
                self.config["patch_size"],
                self.config["patch_step"],
            ),
            num_workers=4,
            batch_size=self.config["batch_size_eval"],
            prefetch_factor=3,
            persistent_workers=True,
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
        self.running_loss = RunningMean(window=10)

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
        y_pred = self.model.last_activation(y_pred)
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
        y_pred = self.model(x, logits=True)
        loss = (
            self.calc_loss(y_pred, y)
            if not self.deep_supervision
            else self.__calc_loss_deep_supervision(y_pred, y)
        )

        self.running_loss(loss)
        self.log(
            "train_loss", self.running_loss.compute(), sync_dist=True, prog_bar=True
        )
        return loss

    def validation_step(self, batch: torch.Tensor):
        x, y = batch
        y_pred = self.model(x, logits=True)
        loss = self.calc_loss(y_pred, y)
        if self.deep_supervision:
            y_pred = y_pred[-1]
        y_pred = self.model.last_activation(y_pred)
        dice = self.dice_fn(y_pred, y)

        self.log("val_loss", loss, sync_dist=True, prog_bar=True)
        self.log("val_dice", dice, sync_dist=True, prog_bar=True)
        return loss

    def test_step(self, batch: torch.Tensor):
        x, y = batch
        y_pred = self.model(x, logits=False)
        loss = self.calc_loss(y_pred, y)
        if self.deep_supervision:
            y_pred = y_pred[-1]
        y_pred = self.model.last_activation(y_pred)
        dice = self.dice_fn(y_pred, y)

        self.log("test_loss", loss, sync_dist=True, prog_bar=True)
        self.log("test_dice", dice, sync_dist=True, prog_bar=True)
        return loss

    def configure_optimizers(self):  # type: ignore
        optimiser = self.config["optimiser"](
            self.model.parameters(), **self.config["optimiser_kwargs"]
        )
        lr_scheduler = self.config["lr_scheduler"](optimiser)  # type: ignore
        return {"optimizer": optimiser, "lr_scheduler": lr_scheduler}
