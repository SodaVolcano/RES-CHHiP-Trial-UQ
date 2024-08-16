from typing import Iterable
from torch.utils.data import IterableDataset

from ..data.patient_scan import PatientScan
from .data_handling import preprocess_dataset
import numpy as np


class PatientScanDataset(IterableDataset):
    def __init__(self, patient_scans: Iterable["PatientScan"], transform=None):
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

    def __iter__(self):
        return iter(self.data)
