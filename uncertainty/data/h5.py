"""
Utility functions for loading and saving data to HDF5 files
"""

from datetime import date
from typing import Generator, Iterable
import os
import torch

import h5py as h5
import numpy as np
from loguru import logger
from tqdm import tqdm

from ..metrics import probability_map, variance_map, entropy_map

from ..utils import curry, logger_wraps
from .datatypes import MaskDict, PatientScan, PatientScanPreprocessed

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
            case x if isinstance(x, list):
                [_create_group({i: val}, str(i)) for i, val in enumerate(x)]
            case x if isinstance(x, np.ndarray | tuple | torch.Tensor):
                group.create_dataset(key, data=val, compression="gzip")
            case _:
                logger.warning(
                    f"Unsupported type {type(val)} for key {key}, skipping"
                )



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
) -> (
    Generator[PatientScan, None, None] | Generator[PatientScanPreprocessed, None, None]
):
    """
    Load patient scan dictionaries from an h5 file
    """
    with h5.File(path, "r") as hf:
        for key in hf.keys():
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


def save_prediction_to_h5(name: str, h5_path: str, x: np.ndarray, y: np.ndarray, y_pred: np.ndarray):
    """
    Save x, y and prediction to a H5 file, appending if the file already exists
    """
    with h5.File(h5_path, 'a' if os.path.exists(h5_path) else 'w') as h5_file:
        _create_group({'x': x, 'y': y, 'y_pred': y_pred}, name, h5_file)

def save_predictions_to_h5(name: str, h5_path: str, x: torch.Tensor, y: torch.Tensor, y_preds: list[torch.Tensor], compute_aggregation: bool = True):
    """
    Save x, y, and list of predictions to H5 file, optionally saving probability, entropy and variance map

    Parameters
    ----------
    name: str
        Name of the group to save in the H5 file
    h5_path: str
        Path to the H5 file
    x: torch.Tensor
        Input tensor
    y: torch.Tensor
        Target tensor
    y_preds: list[torch.Tensor]
        List of predictions
    compute_aggregation: bool
        Whether to compute probability, entropy and variance maps
    #TODO
    """
    dict_ = {'x': x, 'y': y, 'y_preds': y_preds}
    dict_ |= {'probability_map': probability_map(y_preds), "variance_map": variance_map(y_preds), "entropy_map": entropy_map(y_preds)} if compute_aggregation else {}
    with h5.File(h5_path, 'a' if os.path.exists(h5_path) else 'w') as h5_file:
        _create_group(dict_, name, h5_file)


"""
list(preds), y

add y
add group: preds
    for pred in preds, add to h5

load prediction for A class
save probability map
save entropy map
save variance map

"""
