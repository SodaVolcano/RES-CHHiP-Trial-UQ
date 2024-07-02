"""
Collection of class encapsulating volume data, supporting lazy evaluation
"""

from typing import TypeVar
from typing import Any, Generator, Iterable, Iterator, List

from uncertainty.common.constants import MaskType, VolumeType

T = TypeVar("T")
GeneratorOrConcrete = Generator[T, None, None] | T


class Mask:
    """
    Organ mask for a single observer, allow multiple organs
    """

    def __init__(
        self, organs: dict[str, GeneratorOrConcrete[MaskType]], observer: str = ""
    ):
        # dictionary of organ-mask pairs
        self.__organs = organs
        self.observer = observer

    def get_organ_mask(self, organ: str) -> MaskType:
        """
        Return mask array for given organ
        """
        if isinstance(self.__organs[organ], Generator):
            self.__organs[organ] = next(self.__organs[organ])
        return self.__organs[organ]

    def __getitem__(self, idx: str) -> MaskType:
        """
        Return mask array for given organ
        """
        return self.get_organ_mask(idx)

    def get_organ_names(self) -> List[str]:
        """
        Return list of organs
        """
        return list(self.__organs.keys())

    def __repr__(self) -> str:
        return f"Mask(observer='{self.observer}', organs={self.get_organ_names()}"


class PatientScan:
    """
    Volume-mask pair
    """

    patient_id: str

    # 3D array of the CT/MRI scan
    __volume: Generator[VolumeType, None, None] | VolumeType
    # One or multiple masks
    __masks: List[Mask]

    def __init__(
        self,
        patient_id: str,
        volume: GeneratorOrConcrete[VolumeType],
        masks: List[Mask],
    ):
        self.patient_id = patient_id
        self.__volume = volume
        self.__masks = masks

    @property
    def volume(self) -> VolumeType:
        """
        Return volume attribute
        """
        if isinstance(self.__volume, Generator):
            self.__volume = next(self.__volume)
        return self.__volume

    @property
    def n_masks(self) -> int:
        """
        Return number of masks
        """
        return len(self.__masks)

    @property
    def masks(self) -> dict[str, Mask]:
        """
        Return dictionary of observer-mask pairs
        """
        return {mask.observer: mask for mask in self.__masks}

    @property
    def mask_observers(self) -> List[str]:
        """
        Return list of observers
        """
        return [mask.observer for mask in self.__masks]

    def get_mask(self, observer: str) -> Mask:
        """
        Return mask object for given observer
        """
        return next(mask for mask in self.__masks if mask.observer == observer)

    def __repr__(self) -> str:
        return f"PatientScan(patient_id='{self.patient_id}', mask_observers={self.mask_observers})"
