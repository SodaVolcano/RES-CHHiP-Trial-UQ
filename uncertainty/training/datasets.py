"""
PyTorch Dataset classes for loading data from H5 files.
"""

import random
import time
from itertools import islice
from typing import Callable, Generator, Iterable

import h5py as h5
import lightning as lit
import numpy as np
import toolz as tz
import torch
import torch.utils
from kornia.augmentation import RandomAffine3D
from pytorch_lightning import seed_everything
from toolz import curried
from torch.utils.data import DataLoader, Dataset, IterableDataset

from uncertainty.config import auto_match_config

from ..data import augmentations


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
    indices : list[str] | None
        List of indices (patient IDs) that can be fetched from the dataset.
        If None, the whole dataset is used.
    batch_size: int
        Number of patches in a single batch.
    patch_size : tuple[int, int, int]
        Size of the patch to extract from the dataset.
    foreground_oversample_ratio : float
        Ratio of patches guaranteed to contain the foreground class in each batch.
        Hard fixed with a minimum of 1 per batch.
    transform : Callable[[tuple[torch.Tensor, torch.Tensor]], tuple[torch.Tensor, torch.Tensor]]
        Function to apply to the (x, y) patch pair. Default is the identity function.
        Intended to be used for data augmentation.
    """

    @auto_match_config(prefixes=["data", "training"])
    def __init__(
        self,
        h5_path: str,
        indices: list[str] | None,
        batch_size: int,
        patch_size: tuple[int, int, int],
        foreground_oversample_ratio: float,
        transform: Callable[
            [tuple[torch.Tensor, torch.Tensor]], tuple[torch.Tensor, torch.Tensor]
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
    def __patch_iter(self) -> Generator[tuple[torch.Tensor, torch.Tensor], None, None]:
        """
        Return an infinite stream of randomly augmented and sampled patches
        """
        if self.dataset is None:
            self.dataset = H5Dataset(self.h5_path, self.indices)

        while True:
            yield tz.pipe(
                self.dataset[random.choice(self.dataset.indices)],
                lambda xy: self.__sample_patch(*xy),
                curried.map(lambda arr: torch.tensor(arr)),
                tuple,
            )  # type: ignore

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
    def __fg_patch_iter(
        self, n_patches: int
    ) -> Iterable[tuple[torch.Tensor, torch.Tensor]]:
        """
        Iterator of `n_patches` patches guaranteed to contain a foreground
        """
        return tz.pipe(
            self.__patch_iter(),
            curried.filter(lambda x_y: torch.any(x_y[1])),
            lambda x: islice(x, n_patches),
        )  # type: ignore

    @torch.no_grad()
    def __random_patch_iter(
        self, n_patches: int
    ) -> Iterable[tuple[torch.Tensor, torch.Tensor]]:
        """
        Iterator of `n_patches` patches randomly sampled
        """
        return islice(self.__patch_iter(), n_patches)

    @torch.no_grad()
    def __oversampled_iter(self) -> Iterable[tuple[torch.Tensor, torch.Tensor]]:
        """
        Iterator of length `batch_size` with oversampled foreground examples
        """
        n_fg_samples = max(1, int(round(self.batch_size * self.fg_ratio)))
        return tz.concat(
            (
                self.__fg_patch_iter(n_fg_samples),
                self.__random_patch_iter(self.batch_size - n_fg_samples),
            )  # type: ignore
        )

    @torch.no_grad()
    def __iter__(self) -> Generator[tuple[torch.Tensor, torch.Tensor], None, None]:
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
                # transform == tz.identity implies no augmentation
                if self.transform != tz.identity:
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
        If None, the whole dataset is used.
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
    Prepares the train and validation DataLoaders for the segmentation task.


    Note: The test DataLoader is not implemented, use the inference functions from
    `uncertainty.evaluation` instead.

    Tip: It's recommended to pass most parameters using a configuration dictionary,
    i.e. `SegmentationData(**config, train_indices=..., val_indices=...)`. Parameters
    required by the function that are present in `config` will be automatically
    matched and passed in.

    Parameters
    -----------
    h5_path : str
        Path to the H5 file containing the dataset.
    train_indices : list[str] | None
        List of indices (patient IDs) assigned to the training set. If None,
        the whole dataset will be used.
    val_indices : list[str] | None
        List of indices (patient IDs) assigned to the validation set. If None,
        the whole dataset will be used.
    batch_size : int
        Batch size for the training DataLoader.
    batch_size_eval : int
        Batch size for the validation DataLoader.
    patch_size : tuple[int, int, int]
        Size of the patch to extract from the dataset.
    foreground_oversample_ratio : float
        Ratio of patches guaranteed to contain the foreground class in each batch.
    num_workers_train : int
        Number of workers for the training DataLoader.
    num_workers_val : int
        Number of workers for the validation DataLoader.
    prefetch_factor_train : int
        Number of samples loaded in advance by the training DataLoader.
    prefetch_factor_val : int
        Number of samples loaded in advance by the validation DataLoader.
    persistent_workers_train : bool
        Whether to keep the workers alive between epochs for the training DataLoader.
    persistent_workers_val : bool
        Whether to keep the workers alive between epochs for the validation DataLoader.
    pin_memory_train : bool
        Whether to pin memory for the training DataLoader.
    pin_memory_val : bool
        Whether to pin memory for the validation DataLoader.
    """

    @auto_match_config(prefixes=["data", "training"])
    def __init__(
        self,
        h5_path: str,
        batch_size: int,
        batch_size_eval: int,
        patch_size: tuple[int, int, int],
        foreground_oversample_ratio: float,
        num_workers_train: int,
        num_workers_val: int,
        prefetch_factor_train: int = 0,
        prefetch_factor_val: int = 0,
        persistent_workers_train: bool = False,
        persistent_workers_val: bool = False,
        pin_memory_train: bool = False,
        pin_memory_val: bool = False,
        train_indices: list[str] | None = None,
        val_indices: list[str] | None = None,
        augmentations: Callable = augmentations(),
    ):
        """
        Pass in checkpoint_path if you want to dump the indices
        """
        super().__init__()
        self.augmentations = augmentations(p=1)
        self.h5_path = h5_path
        self.batch_size = batch_size
        self.batch_size_eval = batch_size_eval
        self.patch_size = patch_size
        self.fg_ratio = foreground_oversample_ratio
        self.train_indices = train_indices
        self.val_indices = val_indices
        self.num_workers_train = num_workers_train
        self.num_workers_val = num_workers_val
        self.prefetch_factor_train = prefetch_factor_train
        self.prefetch_factor_val = prefetch_factor_val
        self.persistent_workers_train = persistent_workers_train
        self.persistent_workers_val = persistent_workers_val
        self.pin_memory_train = pin_memory_train
        self.pin_memory_val = pin_memory_val

    def train_dataloader(self):
        return DataLoader(
            RandomPatchDataset(
                self.h5_path,
                self.train_indices,
                self.batch_size,
                self.patch_size,
                self.fg_ratio,
                transform=self.augmentations,
            ),
            num_workers=self.num_workers_train,
            batch_size=self.batch_size,
            prefetch_factor=self.prefetch_factor_train,
            persistent_workers=self.persistent_workers_train,
            pin_memory=self.pin_memory_train,
            worker_init_fn=_seed_with_time,
        )

    def val_dataloader(self):
        return DataLoader(
            RandomPatchDataset(
                self.h5_path,
                self.val_indices,
                self.batch_size,
                self.patch_size,
                self.fg_ratio,
            ),
            num_workers=self.num_workers_val,
            batch_size=self.batch_size_eval,
            prefetch_factor=self.prefetch_factor_val,
            pin_memory=self.pin_memory_val,
            persistent_workers=self.persistent_workers_val,
            worker_init_fn=_seed_with_time,
        )

    def test_dataloader(self):
        raise NotImplementedError(
            "Don't use the test loader, use functions from uncertainty.evaluation instead"
        )
