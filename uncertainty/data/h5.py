"""
Utility functions for loading and saving data to HDF5 files
"""

import os
from datetime import date
from typing import Generator, Iterable

import h5py as h5
import numpy as np
import torch
from loguru import logger
from tqdm import tqdm

from ..metrics import entropy_map, probability_map, variance_map
from ..utils import curry, logger_wraps
from .datatypes import PatientScan, PatientScanPreprocessed


@curry
def _create_group(
    dict_: dict,
    name: str,
    hf: h5.File | h5.Group,
) -> None:
    group = hf.create_group(name)
    for key, val in dict_.items():
        match val:
            case x if isinstance(x, str | int | float):
                group[key] = val
            case x if isinstance(x, date):
                group[key] = str(val)
            case x if isinstance(x, dict):
                _create_group(val, key, group)
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
) -> None:
    """
    Save a list of patient scan dictionaries to an h5 file with patient ID as key
    """

    with h5.File(path, "w") as hf:
        for scan in tqdm(dataset, desc="Saving to H5"):
            _create_group(scan, name=f"{scan["patient_id"]}", hf=hf)


@logger_wraps(level="INFO")
def load_scans_from_h5(
    path: str,
    indices: list[str] | None = None,
) -> (
    Generator[PatientScan, None, None] | Generator[PatientScanPreprocessed, None, None]
):
    """
    Load patient scan dictionaries from an h5 file
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


def save_prediction_to_h5(
    h5_path: str,
    group_name: str,
    x: torch.Tensor | None,
    y: torch.Tensor | None,
    y_pred: torch.Tensor,
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
    """
    with h5.File(h5_path, "a" if os.path.exists(h5_path) else "w") as h5_file:
        _create_group({"x": x, "y": y, "y_pred": y_pred}, group_name, h5_file)


def save_predictions_to_h5(
    h5_path: str,
    group_name: str,
    x: torch.Tensor | None,
    y: torch.Tensor | None,
    y_preds: list[torch.Tensor],
    compute_aggregation: bool = True,
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
    """
    dict_ = {"x": x, "y": y, "y_preds": y_preds}
    if compute_aggregation:
        dict_ |= {
            "probability_map": probability_map(y_preds),
            "variance_map": variance_map(y_preds),
            "entropy_map": entropy_map(y_preds),
        }
    with h5.File(h5_path, "a" if os.path.exists(h5_path) else "w") as h5_file:
        _create_group(dict_, group_name, h5_file)
