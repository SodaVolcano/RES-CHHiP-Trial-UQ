from itertools import cycle, tee
from typing import Callable, Iterable, Optional
import torch
from torch.utils.data import IterableDataset

from ..data.patient_scan import PatientScan
from .data_handling import preprocess_dataset
import numpy as np
from toolz import curried
import toolz as tz


class PatientScanDataset(IterableDataset):
    def __init__(
        self,
        patient_scans: Iterable["PatientScan"],
        transform: Optional[
            Callable[[tuple[np.ndarray, np.ndarray]], tuple[np.ndarray, np.ndarray]]
        ] = tz.identity,
    ):
        """
        Parameters
        ----------
        patient_scans : Iterable[PatientScan]
            Iterable of PatientScan objects to be converted to
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
        self.data: Iterable[tuple[np.ndarray, np.ndarray]] = preprocess_dataset(
            patient_scans
        )
        self.transform = transform

    def __iter__(self):  # type: ignore
        self.data, it = tee(self.data, 2)
        return tz.pipe(
            it,
            curried.map(self.transform),
            curried.map(
                lambda vol_mask: (torch.tensor(vol_mask[0]), torch.tensor(vol_mask[1]))
            ),
            cycle,
            lambda it: tz.random_sample(0.1, it),
        )
