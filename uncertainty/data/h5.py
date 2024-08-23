import os
import h5py as h5
from loguru import logger
import numpy as np
from typing import Iterable, Optional

from tqdm import tqdm
import toolz as tz
from toolz import curried

from uncertainty.data.patient_scan import PatientScan
from uncertainty.utils.parallel import pmap

from .mask import Mask, get_organ_names
from ..utils.logging import logger_wraps
from ..utils.path import generate_full_paths
from ..utils.wrappers import curry


@logger_wraps(level="INFO")
@curry
def save_xy_to_h5(
    dataset: Iterable[tuple[np.ndarray, np.ndarray]],
    path: str,
    name: str = "dataset.h5",
) -> None:
    """
    Save a list of instance-label (volume, mask) pairs to an h5 file

    Each tuple is given a numerical key as a string in the h5 file.
    """

    def create_datasets(x: np.ndarray, y: np.ndarray, idx: int, hf: h5.File) -> None:
        group = hf.create_group(f"{idx}")
        group.create_dataset("x", data=x, compression="gzip")
        group.create_dataset("y", data=y, compression="gzip")

    with h5.File(os.path.join(path, name), "w") as hf:
        [
            create_datasets(tup[0], tup[1], i, hf)
            for i, tup in tqdm(enumerate(dataset), desc="Saving to H5")
        ]


@curry
def save_scan_to_h5(scan: PatientScan, save_dir: str) -> None:
    """
    Save the current object in a file at save_dir/patient_id.h5

    The file is removed if an error occurs during saving
    """
    file_path = os.path.join(save_dir, f"{scan.patient_id}.h5")

    try:
        with h5.File(file_path, "w") as f:
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

        load_scan_from_h5(file_path)  # Check if loading works
    except Exception as e:
        logger.error(f"Error saving PatientScan {scan.patient_id}: {e}")
        os.remove(file_path)


def load_scan_from_h5(file_path: str) -> Optional[PatientScan]:
    """
    Load PatientScan object from h5 file, None if error
    """
    try:
        with h5.File(file_path, "r") as f:
            masks = []
            for observer in f.attrs["mask_observers"]:  # type: ignore
                mask_group = f[f"mask_{observer}"]
                organs = {name: mask_group[name][:] for name in mask_group.keys()}  # type: ignore
                masks.append(Mask(organs=organs, observer=observer))  # type: ignore

            return PatientScan(
                patient_id=f.attrs["patient_id"], volume=f["volume"][:], mask_list=masks  # type: ignore
            )
    except Exception as e:
        logger.error(f"Error loading PatientScan from {file_path}: {e}")
        return None


@curry
def load_scans_from_h5(
    dir_path: str, n_workers: int = 1
) -> Iterable[Optional[PatientScan]]:
    """
    Load all PatientScan objects from h5 files in dir_path

    None is returned in place of an object if an error occurs

    Parameters
    ----------
    dir_path : str
        Path to directory containing h5 files
    n_workers : int
        Number of parallel processes to use, set to 1 to disable, by default 1
    """
    mapper = pmap(n_workers=n_workers) if n_workers > 1 else curried.map
    return tz.pipe(
        generate_full_paths(dir_path, os.listdir),
        mapper(load_scan_from_h5),
    )
