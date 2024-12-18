"""
Utility functions for loading and saving data to HDF5 files
"""

import os
from datetime import date
from typing import Generator, Iterable, Literal

import h5py as h5
import numpy as np
import torch
from loguru import logger
from tqdm import tqdm

from ..metrics import entropy_map, probability_map, variance_map
from ..utils import curry, iterate_while, logger_wraps
from .datatypes import PatientScan, PatientScanPreprocessed


@curry
def _create_group(
    dict_: dict,
    name: str,
    hf: h5.File | h5.Group,
    duplicate_name_strategy: Literal["skip", "overwrite", "rename"] = "skip",
) -> None:
    """
    Create a group in an h5 file with the given name and add the dictionary items as datasets or groups

    Parameters
    ----------
    dict_: dict
        Dictionary to save
    name: str
        Name of the group
    hf: h5.File | h5.Group
        H5 file or group to save to
    duplicate_name_strategy: Literal["skip", "allow"]
        Strategy to handle duplicate names. If "skip", skip the key, if "overwrite", overwrite the key, if
        "rename", rename the key by appending `"-<int>"` to the key
    """
    if name in hf.keys():
        if duplicate_name_strategy == "skip":
            logger.warning(f"Key {name} already exists, skipping")
            return
        elif duplicate_name_strategy == "overwrite":
            logger.warning(f"Key {name} already exists, overwriting")
            del hf[name]
        elif duplicate_name_strategy == "rename":
            idx = iterate_while(
                lambda idx: idx + 1, lambda idx: f"{name}-{idx}" in hf.keys(), 1
            )
            name = f"{name}-{idx}"
            logger.warning(f"Key {name} already exists, renaming to {name}")
        else:
            raise ValueError(f"Unsupported strategy {duplicate_name_strategy}")

    group = hf.create_group(name)
    for key, val in dict_.items():
        match val:
            case x if isinstance(x, str | int | float):
                group[key] = val
            case x if isinstance(x, date):
                group[key] = str(val)
            case x if isinstance(x, dict):
                _create_group(val, key, group, duplicate_name_strategy)
            case x if isinstance(x, np.ndarray | tuple | torch.Tensor | list):
                group.create_dataset(key, data=val, compression="gzip")
            case x if x is None:
                pass
            case _:
                logger.warning(f"Unsupported type {type(val)} for key {key}, skipping")


@logger_wraps(level="INFO")
@curry
def save_scans_to_h5(
    dataset: Iterable[PatientScan] | Iterable[PatientScanPreprocessed],
    path: str = "./data/dataset.h5",
    duplicate_name_strategy: Literal["skip", "overwrite", "rename"] = "skip",
) -> None:
    """
    Save a list of patient scan dictionaries to an h5 file with patient ID as key

    Parameters
    ----------
    dataset: Iterable[PatientScan] | Iterable[PatientScanPreprocessed]
        List of patient scan dictionaries
    path: str
        Path to the h5 file
    duplicate_name_strategy: Literal["skip", "overwrite", "rename"]
        Strategy to handle duplicate names. If "skip", skip the key, if "overwrite", overwrite the key, if
        "rename", rename the key by appending `"-<int>"` to the key
    """

    with h5.File(path, "w") as hf:
        for scan in tqdm(dataset, desc="Saving to H5"):
            _create_group(
                scan,
                name=f"{scan["patient_id"]}",
                hf=hf,
                duplicate_name_strategy=duplicate_name_strategy,
            )


@logger_wraps(level="INFO")
@curry
def load_scans_from_h5(
    path: str,
    indices: list[str] | None = None,
) -> (
    Generator[PatientScan, None, None] | Generator[PatientScanPreprocessed, None, None]
):
    """
    Load patient scan dictionaries from an h5 file

    Parameters
    ----------
    path: str
        Path to the h5 file
    indices: list[str] | None
        List of keys to load from the h5 file
    """
    with h5.File(path, "r") as hf:
        for key in indices or hf.keys():
            scan = hf[key]
            yield {
                "patient_id": scan["patient_id"][()],  # type: ignore
                "volume": scan["volume"][:],  # type: ignore
                "dimension_original": tuple(scan["dimension_original"][()]),  # type: ignore
                "spacings": tuple(scan["spacings"][()]),  # type: ignore
                "modality": scan["modality"][()].decode(),  # type: ignore
                "manufacturer": scan["manufacturer"][()].decode(),  # type: ignore
                "scanner": scan["scanner"][()].decode(),  # type: ignore
                "study_date": date.fromisoformat(scan["study_date"][()].decode()),  # type: ignore
                "masks": {
                    organ: scan[f"masks/{organ}"][:]  # type: ignore
                    for organ in scan["masks"].keys()  # type: ignore
                },
            }


@logger_wraps(level="INFO")
@curry
def save_prediction_to_h5(
    h5_path: str,
    group_name: str,
    x: torch.Tensor | None,
    y: torch.Tensor | None,
    y_pred: torch.Tensor,
    duplicate_name_strategy: Literal["skip", "overwrite", "rename"] = "skip",
):
    """
    Save x, y and prediction to a H5 file, appending if the file already exists

    Parameters
    ----------
    h5_path: str
        Path to the H5 file
    group_name: str
        Name of the group to save in the H5 file
    x: torch.Tensor | None
        Input tensor
    y: torch.Tensor | None
        Target tensor
    y_pred: torch.Tensor
        Prediction tensor
    duplicate_name_strategy: Literal["skip", "overwrite", "rename"]
        Strategy to handle duplicate names. If "skip", skip the key, if "overwrite", overwrite the key, if
        "rename", rename the key by appending `"-<int>"` to the key
    """
    with h5.File(h5_path, "a" if os.path.exists(h5_path) else "w") as h5_file:
        _create_group(
            {"x": x, "y": y, "y_pred": y_pred},
            group_name,
            h5_file,
            duplicate_name_strategy=duplicate_name_strategy,
        )


@logger_wraps(level="INFO")
@curry
def save_predictions_to_h5(
    h5_path: str,
    group_name: str,
    x: torch.Tensor | None,
    y: torch.Tensor | None,
    y_preds: list[torch.Tensor],
    compute_aggregation: bool = True,
    duplicate_name_strategy: Literal["skip", "overwrite", "rename"] = "skip",
):
    """
    Save x, y, and list of predictions to H5 file, optionally saving probability, entropy and variance map

    Parameters
    ----------
    h5_path: str
        Path to the H5 file
    group_name: str
        Name of the group to save in the H5 file
    x: torch.Tensor | None
        Input tensor
    y: torch.Tensor | None
        Target tensor
    y_preds: list[torch.Tensor]
        List of predictions
    compute_aggregation: bool
        Whether to compute probability, entropy and variance maps
    duplicate_name_strategy: Literal["skip", "overwrite", "rename"]
        Strategy to handle duplicate names. If "skip", skip the key, if "overwrite", overwrite the key, if
        "rename", rename the key by appending `"-<int>"` to the key
    """
    dict_ = {"x": x, "y": y, "y_preds": y_preds}
    if compute_aggregation:
        dict_ |= {
            "probability_map": probability_map(y_preds),
            "variance_map": variance_map(y_preds),
            "entropy_map": entropy_map(y_preds),
        }
    with h5.File(h5_path, "a" if os.path.exists(h5_path) else "w") as h5_file:
        _create_group(
            dict_, group_name, h5_file, duplicate_name_strategy=duplicate_name_strategy
        )
