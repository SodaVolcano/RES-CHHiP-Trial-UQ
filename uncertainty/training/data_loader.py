from itertools import cycle, tee
from typing import Callable, Iterable, Optional
import torch
from torch.utils.data import Dataset

from ..data.patient_scan import PatientScan
from .data_handling import preprocess_dataset
import numpy as np
from toolz import curried
import toolz as tz


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
            lambda vol_mask: (torch.tensor(vol_mask[0]), torch.tensor(vol_mask)[1]),
        )  # type: ignore
