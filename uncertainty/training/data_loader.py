from itertools import cycle
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
        buffer_size: int = 100,
    ):
        """
        Parameters
        ----------
        patient_scans : Iterable[PatientScan]
            Iterable of PatientScan objects to be converted to
            (volume, masks) pairs. The volume and masks will have shape
            (H, W, D, C) where C is the number of channels which
            PatientScanDataset will convert to (C, H, W, D).
        transform : Optional[Compose]
            A Volumentations-3D Compose object to apply transformations to
            the volume and masks (e.g. data augmentation). If None, no
            transformations are applied.
        """
        super().__init__()
        self.data: Iterable[tuple[np.ndarray, np.ndarray]] = preprocess_dataset(
            patient_scans
        )
        self.transform = transform if transform is not None else tz.identity
        self.buffer_size = buffer_size

    def __iter__(self):  # type: ignore
        return tz.pipe(
            self.data,
            cycle,
            lambda it: tz.random_sample(0.2, it),
            curried.map(self.transform),
            curried.map(
                lambda vol_mask: (torch.tensor(vol_mask[0]), torch.tensor(vol_mask[1]))
            ),
        )
