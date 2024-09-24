from itertools import islice
import os
from typing import Callable
import random


import torch.utils
from kornia.augmentation import RandomAffine3D
import numpy as np
import toolz as tz
import torch
from torch.utils.data import Dataset, IterableDataset, Subset
from toolz import curried
import h5py as h5
from skimage.util import view_as_windows

from ..config import Configuration, configuration


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
        patch_size: tuple[int, ...],
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
            curried.map(
                lambda arr: arr.numpy() if isinstance(arr, torch.Tensor) else arr
            ),
            curried.map(
                lambda arr: view_as_windows(
                    arr, (arr.shape[0], *self.patch_size), self.patch_step
                )
            ),  # type: ignore
            tuple,
        )
        assert x_patches is not None and y_patches is not None
        for idx in np.ndindex(x_patches.shape[:-4]):  # type: ignore
            yield torch.tensor(x_patches[idx]), torch.tensor(y_patches[idx])

    @torch.no_grad()
    def __iter__(self):  # type: ignore
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
    fg_oversample_ratio : float
        Ratio of patches with guaranteed foreground class to include in each batch. Hard
        fixed with a minimum of 1 per batch.
    ensemble_size : int
        Number of models in the ensemble to fetch batch for. Batch size is `batch_size * ensemble_size`
        intended to feed each ensemble member with `batch_size` number of patches, each with
        oversampled patches.
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
        indices: list[int] | Subset,
        batch_size: int,
        patch_size: tuple[int, ...],
        fg_oversample_ratio: float,
        ensemble_size: int = 1,
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
        self.ensemble_size = ensemble_size
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
        batch = []

        while True:
            if len(batch) == 0:
                batch = (vol_mask for vol_mask in self.__oversampled_iter())
                x, y = zip(*batch)
                # Apply augmentation batch-wise, more efficient
                x = self.affine(torch.stack(x))
                y = self.affine_mask(torch.stack(y), self.affine._params)

                batch = list(map(self.transform, zip(x, y)))
            yield batch.pop()

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
    config: Configuration
        Dictionary containing configurations.
    indices: list[int] | None
        Optinal list of indices. If specified, only elements
        with the specified indices are fetched. Good for specifying
        train-validation split. If the indices are shuffled, the
        dataset will also be shuffled.

    Attributes
    ----------
    h5_file_path : str
        The file path of the H5 file.
    h5_file : h5py.File
        The opened H5 file object.
    keys : list of str
        List of keys corresponding to the dataset entries in the H5 file.
    indices: list[int] | None
        List of indices that can be fetched from the dataset.
    """

    def __init__(
        self,
        h5_file_path: str,
        config: Configuration = configuration(),
        indices: list[int] | None = None,
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
