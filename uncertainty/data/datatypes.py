"""
Collection of class encapsulating volume data, supporting lazy evaluation
"""

import shutil
from typing import TypeVar
from typing import Generator, List
import os
import tempfile

from loguru import logger
import numpy as np
import h5py

from uncertainty.utils.logging import logger_wraps

T = TypeVar("T")
GeneratorOrConcrete = Generator[T, None, None] | T


class Mask:
    """
    Organ mask for a single observer, allow multiple organs
    """

    def __init__(
        self, organs: dict[str, GeneratorOrConcrete[np.ndarray]], observer: str = ""
    ):
        # dictionary of organ-mask pairs
        self.__organs = organs
        self.observer = observer

    def get_organ_mask(self, organ: str) -> np.ndarray:
        """
        Return mask array for given organ
        """
        if isinstance(self.__organs[organ], Generator):

            self.__organs[organ] = next(self.__organs[organ])
        return self.__organs[organ]

    def __getitem__(self, idx: str) -> np.ndarray:
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

    def as_array(self, organ_ordering: List[str]) -> np.array:
        """
        Return array of N masks with shape (H, W, D, N), ordered by organ_ordering
        """
        return np.stack([self[organ] for organ in organ_ordering], axis=-1)


class PatientScan:
    """
    Volume-mask pair
    """

    patient_id: str

    # 3D array of the CT/MRI scan
    __volume: Generator[np.array, None, None] | np.array
    # One or multiple masks
    __masks: List[Mask]

    def __init__(
        self,
        patient_id: str,
        volume: GeneratorOrConcrete[np.array],
        masks: List[Mask],
    ):
        self.patient_id = patient_id
        self.__volume = volume
        self.__masks = masks

    @property
    def volume(self) -> np.array:
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

    @classmethod
    @logger.catch
    def load_h5py(cls, file_path: str):
        with h5py.File(file_path, "r") as f:
            masks = []
            for observer in f.attrs["mask_observers"]:
                mask_group = f[f"mask_{observer}"]
                organs = {name: mask_group[name][:] for name in mask_group.keys()}
                masks.append(Mask(organs=organs, observer=observer))

            return PatientScan(
                patient_id=f.attrs["patient_id"], volume=f["volume"][:], masks=masks
            )

    @logger.catch
    def save_h5py(self, save_dir: str):
        """
        Save the current object in a file at save_dir/patient_id.h5

        No file is created if exception occurs
        """
        temp_path = tempfile.NamedTemporaryFile(delete=False).name
        file_path = os.path.join(save_dir, f"{self.patient_id}.h5")

        try:
            with h5py.File(temp_path, "w") as f:
                f.create_dataset("volume", data=self.volume, compression="gzip")

                for mask in self.masks.values():
                    group = f.create_group(f"mask_{mask.observer}")
                    for organ_name in mask.get_organ_names():
                        group.create_dataset(
                            organ_name, data=mask[organ_name], compression="gzip"
                        )

                f.attrs["patient_id"] = self.patient_id
                f.attrs["mask_observers"] = self.mask_observers

            self.load_h5py(temp_path)  # Check if loading works
            shutil.move(temp_path, file_path)
        except Exception as e:
            os.remove(temp_path)
            logger.error(f"Error saving PatientScan {self.patient_id}: {e}")

    def __repr__(self) -> str:
        return f"PatientScan(patient_id='{self.patient_id}', mask_observers={self.mask_observers})"

    @property
    def training_instance(self) -> tuple[np.array, np.array]:
        """
        Return tuple of volume and mask for training
        """
        pass
