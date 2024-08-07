"""
Data type Mask and functions that operate on Mask
"""

from typing import NamedTuple, override
import numpy as np

from uncertainty.utils.wrappers import curry


class Mask(NamedTuple):
    """Organ mask for a single observer, allow multiple organs"""

    organs: dict[str, np.ndarray]
    observer: str = ""

    def __getitem__(self, idx: str) -> np.ndarray:
        """
        Return mask array for given organ

        Side effect: evaluates mask array if it's a generator
        """
        return self.organs[idx]

    def __repr__(self) -> str:
        return f"Mask(observer='{self.observer}', organs={get_organ_names(self)}"

    def keys(self) -> list[str]:
        """
        Return list of organ names
        """
        return list(self.organs.keys())


@curry
def get_organ_mask(mask: Mask, organ: str) -> np.ndarray:
    """
    Return mask array for given organ
    """
    return mask[organ]


def get_organ_names(mask: Mask) -> list[str]:
    """
    Return list of organs
    """
    return list(mask.keys())


@curry
def masks_as_array(mask: Mask, organ_ordering: list[str]) -> np.ndarray:
    """
    Return array of N masks with shape `(H, W, D, n_organs)`, ordered by `organ_ordering`
    """
    return np.stack([mask[organ] for organ in organ_ordering], axis=-1)
