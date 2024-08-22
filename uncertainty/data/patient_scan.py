import os
from typing import List, NamedTuple

import numpy as np

from .mask import Mask


class PatientScan(NamedTuple):
    """
    Volume-mask pair
    """

    patient_id: str
    volume: np.ndarray
    mask_list: List[Mask]

    @property
    def masks(self) -> dict[str, Mask]:
        """
        Return list of masks
        """
        return {mask.observer: mask for mask in self.mask_list}

    @property
    def n_masks(self) -> int:
        """
        Return number of masks
        """
        return len(self.masks)

    @property
    def mask_observers(self) -> List[str]:
        """
        Return list of observers
        """
        return [mask.observer for mask in self.mask_list]

    def __getitem__(self, idx: int) -> Mask:  # type: ignore
        """
        Return mask object for given observer
        """
        return next(mask for mask in self.mask_list if mask.observer == idx)

    def __repr__(self) -> str:
        return f"PatientScan(patient_id='{self.patient_id}', mask_observers={self.mask_observers})"
