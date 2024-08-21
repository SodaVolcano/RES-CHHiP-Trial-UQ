import os
from typing import Generator, Iterable, List, NamedTuple, Optional

import h5py
import numpy as np
import toolz as tz
from loguru import logger
from toolz import curried

from uncertainty.utils.path import generate_full_paths

from .mask import Mask, get_organ_names


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

    def __getitem__(self, idx: int) -> Mask:
        """
        Return mask object for given observer
        """
        return next(mask for mask in self.mask_list if mask.observer == idx)

    def __repr__(self) -> str:
        return f"PatientScan(patient_id='{self.patient_id}', mask_observers={self.mask_observers})"


def from_h5(file_path: str) -> Optional[PatientScan]:
    """
    Load PatientScan object from h5 file, None if error
    """
    try:
        with h5py.File(file_path, "r") as f:
            masks = []
            for observer in f.attrs["mask_observers"]:
                mask_group = f[f"mask_{observer}"]
                organs = {name: mask_group[name][:] for name in mask_group.keys()}
                masks.append(Mask(organs=organs, observer=observer))

            return PatientScan(
                patient_id=f.attrs["patient_id"], volume=f["volume"][:], mask_list=masks
            )
    except Exception as e:
        logger.error(f"Error loading PatientScan from {file_path}: {e}")
        return None


def from_h5_dir(dir_path: str) -> Iterable[Optional[PatientScan]]:
    """
    Load all PatientScan objects from h5 files in dir_path

    None is returned in place of an object if an error occurs
    """
    return tz.pipe(
        generate_full_paths(dir_path, os.listdir),
        curried.map(from_h5),
    )


def save_h5(scan: PatientScan, save_dir: str) -> None:
    """
    Save the current object in a file at save_dir/patient_id.h5

    The file is removed if an error occurs during saving
    """
    file_path = os.path.join(save_dir, f"{scan.patient_id}.h5")

    try:
        with h5py.File(file_path, "w") as f:
            f.create_dataset("volume", data=scan.volume, compression="gzip")

            for mask in scan.masks.values():
                group = f.create_group(f"mask_{mask.observer}")
                for organ_name in get_organ_names(mask):
                    # Replace with "|" to avoid incompatible object error, "" also causes error
                    group_name = (
                        organ_name.replace("/", "|")
                        if len(organ_name) > 0
                        else "_empty_"
                    )
                    group.create_dataset(
                        group_name, data=mask[organ_name], compression="gzip"
                    )

            f.attrs["patient_id"] = scan.patient_id
            f.attrs["mask_observers"] = scan.mask_observers

        from_h5(file_path)  # Check if loading works
    except Exception as e:
        logger.error(f"Error saving PatientScan {scan.patient_id}: {e}")
        os.remove(file_path)
