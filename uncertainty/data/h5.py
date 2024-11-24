"""
Utility functions for loading and saving data to HDF5 files
"""
import os
from typing import Generator, Iterable
import h5py as h5
import numpy as np
import torch
from tqdm import tqdm

from ..utils import logger_wraps, curry
from .datatypes import PatientScan, PatientScanPreprocessed


@logger_wraps(level="INFO")
@curry
def save_scans_to_h5(
    dataset: Iterable[PatientScan] | Iterable[PatientScanPreprocessed],
    path: str = "./data/dataset.h5",
) -> None:
    """
    Save a list of patient scan dictionaries to an h5 file with patient ID as key
    """

    with h5.File(path, "w") as hf:
        create_dataset = lambda scan: hf.create_dataset(f"{scan["patient_id"]}", data=scan, compression="gzip")
        map(create_dataset, tqdm(dataset, desc="Saving to H5"))


@logger_wraps(level="INFO")
def load_scans_from_h5(path: str) -> Generator[h5.Dataset | h5.Group | h5.Datatype, None, None]:
    """
    Load patient scan dictionaries from an h5 file
    """
    with h5.File(path, "r") as hf:
        for key in hf.keys():
            yield hf[key]



# TODO: refactor
@logger_wraps(level="INFO")
@curry
def __save_xy_pred_to_h5(
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
def __save_pred_to_h5(
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
def __load_xy_pred_from_h5(fname: str) -> Iterable[tuple[np.ndarray, np.ndarray]]:
    """
    Load instance-label-prediction from an h5 file
    """
    with h5.File(fname, "r") as hf:
        for key in hf.keys():
            yield hf[key]["x"][:], hf[key]["y"][:], hf[key]["y_pred"][:]  # type: ignore


@logger_wraps(level="INFO")
def __load_pred_from_h5(
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

