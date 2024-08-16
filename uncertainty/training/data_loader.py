from typing import Iterable, Optional
import torch
from torch.utils.data import IterableDataset

from ..data.patient_scan import PatientScan
from .data_handling import preprocess_dataset
import numpy as np
from volumentations import Compose
from toolz import curried
import toolz as tz


class PatientScanDataset(IterableDataset):
    def __init__(
        self,
        patient_scans: Iterable["PatientScan"],
        transform: Optional[Compose] = None,
    ):
        """
        Parameters
        ----------
        patient_scans : Iterable[PatientScan]
            Iterable of PatientScan objects to be converted to
            (volume, masks) pairs. The volume and masks will have shape
            (H, W, D, C) where C is the number of channels which
            PatientScanDataset will convert to (C, H, W, D).
        transform
        """
        self.data: Iterable[tuple[np.ndarray, np.ndarray]] = preprocess_dataset(
            patient_scans
        )
        self.transform = transform if transform is not None else tz.identity

    def __iter__(self):
        return tz.pipe(
            iter(self.data),
            curried.map(self.transform),
            curried.map(lambda x: torch.tensor(x)),
        )
