import os
from typing import Iterable, Optional

import h5py as h5
import numpy as np
import toolz as tz
import torch
from loguru import logger
from toolz import curried
from tqdm import tqdm

from uncertainty.data.patient_scan import PatientScan
from uncertainty.utils.parallel import pmap

from ..utils.logging import logger_wraps
from ..utils.path import generate_full_paths
from ..utils.wrappers import curry
from .mask import Mask, get_organ_names


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


@logger_wraps(level="INFO")
@curry
def save_xy_pred_to_h5(
    xy_preds: Iterable[tuple[np.ndarray, np.ndarray, np.ndarray]],
    path: str,
    name: str = "predictions.h5",
) -> None:
    """
    Save stream of (volume, mask, predicted_mask) to an h5 file

    Each tuple is given a numerical key as a string in the h5 file.
    """

    def create_datasets(
        x: np.ndarray, y: np.ndarray, y_pred: np.ndarray, idx: int, hf: h5.File
    ) -> None:
        group = hf.create_group(f"{idx}")
        group.create_dataset("x", data=x, compression="gzip")
        group.create_dataset("y", data=y, compression="gzip")
        group.create_dataset("y_pred", data=y_pred, compression="gzip")

    with h5.File(os.path.join(path, name), "w") as hf:
        [
            create_datasets(*xy_pred, i, hf)
            for i, xy_pred in tqdm(enumerate(xy_preds), desc="Saving to H5")
        ]


@logger_wraps(level="INFO")
@curry
def save_pred_to_h5(
    xy_preds: Iterable[tuple[np.ndarray, np.ndarray, np.ndarray]],
    indices: list[int],
    path: str,
    name: str = "predictions.h5",
) -> None:
    """
    Save predictions and their index to an h5 file

    """

    def create_datasets(idx: int, y_pred: np.ndarray, hf: h5.File) -> None:
        group = hf.create_group(f"{idx}")
        if isinstance(y_pred, torch.Tensor):
            group.create_dataset("y_pred", data=y_pred, compression="gzip")
            return

        for i, pred in enumerate(y_pred):
            group.create_dataset(f"y_pred_{i}", data=pred, compression="gzip")

    with h5.File(os.path.join(path, name), "w") as hf:
        [
            create_datasets(i, pred, hf)
            for i, pred in tqdm(zip(indices, xy_preds), desc="Saving to H5")
        ]


@logger_wraps(level="INFO")
def load_xy_from_h5(fname: str) -> Iterable[tuple[np.ndarray, np.ndarray]]:
    """
    Load instance-label pairs from an h5 file
    """
    with h5.File(fname, "r") as hf:
        for key in hf.keys():
            yield hf[key]["x"][:], hf[key]["y"][:]  # type: ignore


@logger_wraps(level="INFO")
def load_xy_pred_from_h5(fname: str) -> Iterable[tuple[np.ndarray, np.ndarray]]:
    """
    Load instance-label-prediction from an h5 file
    """
    with h5.File(fname, "r") as hf:
        for key in hf.keys():
            yield hf[key]["x"][:], hf[key]["y"][:], hf[key]["y_pred"][:]  # type: ignore


@logger_wraps(level="INFO")
def load_pred_from_h5(
    fname: str, keys: list[str] | None = None
) -> Iterable[tuple[np.ndarray, np.ndarray]]:
    """
    Load predictions from an h5 file

    keys: list of order of keys to load the data in,
    used to ensure laod order is the same as dump order
    """
    with h5.File(fname, "r") as hf:
        for key in keys or hf.keys():
            if list(hf[key].keys()) == ["y_pred"]:
                yield hf[key]["y_pred"][:]
            else:
                yield [hf[key][f"y_pred_{i}"][:] for i in range(len(hf[key].keys()))]


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
