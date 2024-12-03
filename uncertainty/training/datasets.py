"""
PyTorch Dataset classes for loading data from H5 files.
"""

import os
import random
import time
from itertools import islice
from typing import Callable

import h5py as h5
import numpy as np
import toolz as tz
import torch
import torch.utils
from kornia.augmentation import RandomAffine3D
from toolz import curried
from torch.utils.data import Dataset, IterableDataset, Subset


import lightning as lit
from pytorch_lightning import seed_everything
from torch.utils.data import DataLoader

from uncertainty.config import auto_match_config

from ..data import augmentations


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


class RandomPatchDataset(IterableDataset):
    """
    Randomly sample patches from a dataset of 3D images.

    Given a `batch_size`, each `batch_size` chunk of data returned
    by the iterator is guaranteed to have `max(1, fg_oversample_ratio)`
    number of patches with at least one foreground class in it.

    Parameters
    ----------
    h5_path : str
        Path to the H5 file containing the PatientScan dataset.
    indices : list[int]
        List of indices to sample from the dataset.
    batch_size:
        Number of patches in a single batch.
    patch_size : tuple[int, int, int]
        Size of the patch to extract from the dataset.
    fg_oversample_ratio : float
        Ratio of patches guaranteed to contain the foreground class in each batch.
        Hard fixed with a minimum of 1 per batch.
    transform : Callable[[tuple[np.ndarray, np.ndarray]], tuple[np.ndarray, np.ndarray]]
        Function to apply to the (x, y) patch pair. Default is the identity function.
        Intended to be used for data augmentation.
    """

    @auto_match_config(prefixes=["training"])
    def __init__(
        self,
        h5_path: str,
        indices: list[int],
        batch_size: int,
        patch_size: tuple[int, int, int],
        foreground_oversample_ratio: float,
        transform: Callable[
            [tuple[np.ndarray, np.ndarray]], tuple[np.ndarray, np.ndarray]
        ] = tz.identity,
    ):
        self.indices = indices
        self.h5_path = h5_path
        self.dataset = None  # initialised later to avoid expensive pickling
        self.transform = transform
        self.patch_size = patch_size
        self.fg_ratio = foreground_oversample_ratio
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
            self.dataset = H5Dataset(self.h5_path, self.indices)

        while True:
            yield tz.pipe(
                self.dataset[random.choice(self.indices)],
                lambda xy: self.__sample_patch(*xy),
                curried.map(lambda arr: torch.tensor(arr)),
                tuple,
            )

    @torch.no_grad()
    def __sample_patch(
        self, x: np.ndarray, y: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
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

        return (_extract_patch(x), _extract_patch(y))

    @torch.no_grad()
    def __fg_patch_iter(self, n_patches: int):
        """
        Iterator of `n_patches` patches guaranteed to contain a foreground
        """
        return tz.pipe(
            self.__patch_iter(),
            curried.filter(lambda xy: torch.any(xy[1])),
            lambda x: islice(x, n_patches),
        )

    @torch.no_grad()
    def __random_patch_iter(self, n_patches: int):
        """
        Iterator of `n_patches` patches randomly sampled
        """
        return islice(self.__patch_iter(), n_patches)

    @torch.no_grad()
    def __oversampled_iter(self):
        """
        Iterator of length `batch_size` with oversampled foreground examples
        """
        n_fg_samples = max(1, int(round(self.batch_size * self.fg_ratio)))
        return tz.concat(
            (
                self.__fg_patch_iter(n_fg_samples),
                self.__random_patch_iter(self.batch_size - n_fg_samples),
            )
        )

    @torch.no_grad()
    def __iter__(self):
        """
        Return infinite stream of sampled patches

        Each `batch_size` is guaranteed to have at least one patch with the
        foreground class, and number of such patch in the batch is guaranteed
        to be `round(fg_oversample_ratio * batch_size)`
        """
        batch = []

        while True:
            if len(batch) == 0:
                batch = (vol_mask for vol_mask in self.__oversampled_iter())
                x, y = zip(*batch)
                # Apply augmentation batch-wise, more efficient than element-wise
                x = self.affine(torch.stack(x))
                y = self.affine_mask(torch.stack(y), self.affine._params)

                batch = list(map(self.transform, zip(x, y)))
            yield batch.pop()

    @torch.no_grad()
    def __del__(self):
        if self.dataset is not None:
            del self.dataset


def _seed_with_time(id: int):
    """Seed everything with current time and id"""
    seed_everything(int(time.time() + (id * 3)), verbose=False)


class H5Dataset(Dataset):
    """
    PyTorch Dataset for loading (x, y) pairs from an H5 file.

    H5 file must contain preprocessed data where the masks are stored as
    a single numpy array where the channels correspond to the different
    organs.

    Parameters
    ----------
    h5_path : str
        Path to the H5 file containing the dataset.
    indices : list[str] | None
        List of indices (patient IDs) that can be fetched from the dataset.
    """

    def __init__(
        self,
        h5_path: str,
        indices: list[str] | None = None,
    ):
        self.h5_path = h5_path
        self.dataset = None
        with h5.File(h5_path, "r") as f:
            self.indices = indices or list(f.keys())

    def __len__(self) -> int:
        return len(self.indices)

    def __iter__(self):
        for idx in self.indices:
            yield self[idx]

    @torch.no_grad()
    def __getitem__(self, idx: str) -> tuple[np.ndarray, np.ndarray]:
        """
        Get the (x, y) pair for the given index.
        """
        # opened HDF5 is not pickleable, so don't open in __init__!
        # open once here to prevent overhead
        if self.dataset is None:
            self.dataset = h5.File(self.h5_path, "r")

        return tz.pipe(
            self.dataset[idx],
            lambda group: (group["volume"][:], group["masks"][:]),
            tuple,
        )  # type: ignore

    @torch.no_grad()
    def __del__(self):
        if self.dataset is not None:
            self.dataset.close()


class SegmentationData(lit.LightningDataModule):
    """
    Prepares the train and validation DataLoader for the segmentation task.
    """

    def __init__(
        self,
        train_indices: list[str] | None = None,
        val_indices: list[str] | None = None,
        augmentations: Callable = augmentations(),
    ):
        """
        Pass in checkpoint_path if you want to dump the indices
        """
        super().__init__()
        self.augmentations = augmentations(p=1)
        self.train_indices = train_indices
        self.val_indices = val_indices

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
