from itertools import islice
import os
from typing import Callable, Optional
import random


import torch.utils
from kornia.augmentation import RandomAffine3D
from pytorch_lightning import seed_everything
import time
import lightning as lit
import numpy as np
import toolz as tz
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split, IterableDataset
from toolz import curried
import h5py as h5
from skimage.util import view_as_windows
from torchmetrics.aggregation import RunningMean
from torchmetrics.classification import MultilabelF1Score

from .loss import SmoothDiceLoss
from ..config import Configuration, configuration
from .augmentations import augmentations


def get_unique_filename(base_path):
    """Generate a unique filename by appending (copy) if the file already exists."""
    if not os.path.exists(base_path):
        return base_path

    base, ext = os.path.splitext(base_path)
    counter = 1
    while True:
        new_filename = f"{base} (copy{counter}){ext}"
        if not os.path.exists(new_filename):
            return new_filename
        counter += 1


class SlidingPatchDataset(IterableDataset):
    """
    Produce rolling window of patches from a dataset of 3D images.
    """

    def __init__(
        self,
        h5_file_path: str,
        config: Configuration,
        patch_size: int,
        patch_step: int,
        transform: Callable[
            [tuple[np.ndarray, np.ndarray]], tuple[np.ndarray, np.ndarray]
        ] = tz.identity,
    ):
        self.h5_file_path = h5_file_path
        self.config = config
        self.dataset = None
        self.transform = transform
        self.patch_size = patch_size
        self.patch_step = patch_step

    @torch.no_grad()
    def __patch_iter(self, idx: int):
        """
        Return an iterator of patches for data at index `idx`
        """
        if self.dataset is None:
            self.dataset = H5Dataset(self.h5_file_path, self.config)

        x_patches, y_patches = tz.pipe(
            self.dataset[idx],
            curried.map(lambda arr: arr.numpy()),
            curried.map(
                lambda arr: view_as_windows(
                    arr, (arr.shape[0], *self.patch_size), self.patch_step
                )
            ),
            tuple,
        )
        for idx in np.ndindex(x_patches.shape[:-4]):
            yield torch.tensor(x_patches[idx]), torch.tensor(y_patches[idx])

    @torch.no_grad()
    def __iter__(self):
        """
        Return iterator of patches over all data in the dataset
        """
        if self.dataset is None:
            self.dataset = H5Dataset(self.h5_file_path, self.config)

        return tz.pipe(
            range(len(self.dataset)),
            curried.map(self.__patch_iter),
            tz.concat,
            curried.map(self.transform),
        )

    def __del__(self):
        if self.dataset is not None:
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
        Function to apply to the (x, y) patch pair. Default is the identity function.
        Intended to be used for data augmentation.
    config : Configuration
        Configuration object
    """

    def __init__(
        self,
        h5_file_path: str,
        config: Configuration,
        indices: list[int],
        batch_size: int,
        patch_size: int,
        fg_oversample_ratio: float,
        transform: Callable[
            [tuple[np.ndarray, np.ndarray]], tuple[np.ndarray, np.ndarray]
        ] = tz.identity,
    ):
        self.indices = indices
        self.config = config
        self.h5_file_path = h5_file_path
        self.dataset = None
        self.transform = transform
        self.patch_size = patch_size
        self.fg_ratio = fg_oversample_ratio
        self.batch_size = batch_size
        self.affine = RandomAffine3D(
            5, align_corners=True, shears=0, scale=(0.9, 1.1), p=0.15
        )
        self.affine_mask = RandomAffine3D(
            5,
            align_corners=True,
            shears=0,
            scale=(0.9, 1.1),
            resample="nearest",
            p=0.15,
        )

    @torch.no_grad()
    def __patch_iter(self):
        """
        Return an infinite stream of randomly augmented and sampled patches
        """
        if self.dataset is None:
            self.dataset = H5Dataset(self.h5_file_path, self.config)

        while True:
            # type: ignore
            yield tz.pipe(
                self.dataset[random.choice(self.indices)],
                lambda xy: self.__sample_patch(*xy),
                curried.map(lambda arr: torch.tensor(arr)),
                tuple,
            )

    @torch.no_grad()
    def __sample_patch(self, x, y) -> tuple[np.ndarray, np.ndarray]:
        """
        Randomly sample a patch from (x, y)
        """
        h_coord, w_coord, d_coord = [
            random.randint(0, dim - patch_dim)
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

    @torch.no_grad()
    def __fg_patch_iter(self, length: int):
        """
        Iterator of `length` patches guaranteed to contain a foreground
        """
        return tz.pipe(
            self.__patch_iter(),
            curried.filter(lambda xy: torch.any(xy[1])),
            lambda x: islice(x, length),
        )

    @torch.no_grad()
    def __random_patch_iter(self, length: int):
        """
        Iterator of `length` patches randomly sampled
        """
        return islice(self.__patch_iter(), length)

    @torch.no_grad()
    def __oversampled_iter(self):
        """
        Iterator of length `batch_size` with oversampled foreground examples
        """
        n_fg_samples = max(1, int(round(self.batch_size * self.fg_ratio)))
        return tz.pipe(
            (
                self.__fg_patch_iter(n_fg_samples),
                self.__random_patch_iter(self.batch_size - n_fg_samples),
            ),
            tz.concat,
        )

    @torch.no_grad()
    def __iter__(self):
        """
        Return infinite stream of sampled patches

        Each `batch_size` is guaranteed to have at least one patch with the
        foreground class, and number of such patch in the batch is guaranteed
        to be `round(fg_oversample_ratio * batch_size)`
        """
        x, y = [], []

        while True:
            if len(x) == 0:
                batch = (vol_mask for vol_mask in self.__oversampled_iter())
                x, y = zip(*batch)
                # Apply augmentation batch-wise, more efficient
                x = self.affine(torch.stack(x))
                y = self.affine_mask(torch.stack(y), self.affine._params)

            yield next(map(self.transform, zip(x, y)))

    @torch.no_grad()
    def __del__(self):
        if self.dataset is not None:
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

    @torch.no_grad()
    def __getitem__(self, index: int):
        # opened HDF5 is not pickleable, so don't open in __init__
        # open once to prevent overhead
        if self.dataset is None:
            self.dataset = h5.File(self.h5_file_path, "r")

        return tz.pipe(
            self.dataset[self.keys[index]],
            # [:] changes data from dataset to numpy array
            lambda group: (group["x"][:], group["y"][:]),
            tuple,
        )

    @torch.no_grad()
    def __del__(self):
        if self.dataset is not None:
            self.dataset.close()


def _seed_with_time(id: int):
    seed_everything(time.time() + (id * 3), verbose=False)


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

        indices = list(range(len(H5Dataset(self.train_fname))))
        self.train_indices, self.val_indices = random_split(
            indices, [1 - config["val_split"], config["val_split"]]
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
            num_workers=14,
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
                self.test,
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
        class_weights: Optional[torch.Tensor] = None,
        n_models: int = 1,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        self.model = model
        self.config = config
        self.deep_supervision = (
            hasattr(model, "deep_supervision") and model.deep_supervision
        )
        self.class_weights = class_weights
        self.bce_loss = nn.BCEWithLogitsLoss(class_weights)
        # Smooth, differentiable approximation
        self.dice_loss = SmoothDiceLoss()
        # Original dice, used for evaluation
        self.dice_eval = MultilabelF1Score(num_labels=config["output_channel"])
        self.running_loss = RunningMean(window=10)
        self.val_counter = 0

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
        ys = [nn.Upsample(size=y_pred.shape[2:])(y) for y_pred in y_preds]

        weights = [1 / (2**i) for i in range(len(y_preds))]
        weights = [weight / sum(weights) for weight in weights]  # normalise to sum to 1

        # Reverse weight to match y_preds; halves weights as resolution decreases
        return sum(
            weight * self.calc_loss_single(y_pred, y_scaled)
            for weight, y_pred, y_scaled in zip(reversed(weights), y_preds, ys)
        )

    def calc_loss_single(self, y_pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute the sum of the binary cross entropy loss and the dice loss for one output
        """
        # Sums BCE loss for each class, y have shape (batch, n_classes, H, W, D)
        bce = self.bce_loss(y_pred, y)
        # y_pred = (
        #    self.model.last_activation(y_pred) > self.config["classification_threshold"]
        # )
        dice = self.dice_loss(y_pred, y)

        self.log("bce", bce.detach(), sync_dist=True, prog_bar=True)
        self.log("dice", 1 - dice.detach(), sync_dist=True, prog_bar=True)
        return bce + dice  # scale dice to match bce?

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
        loss = self.calc_loss(y_pred, y)

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
        loss = self.calc_loss(y_pred, y)
        if self.deep_supervision:
            y_pred = y_pred[-1]
        dice = self.dice_eval(y_pred, y)

        # TODO: get dice of each organ

        if self.val_counter % 10:
            name = os.path.join("./batches", f"batch_{self.counter}.pt")
            torch.save(
                {"x": x, "y": y, "y_pred": y_pred, "dice": dice, "loss": loss}, name
            )
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
        loss = self.calc_loss(y_pred, y)
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
