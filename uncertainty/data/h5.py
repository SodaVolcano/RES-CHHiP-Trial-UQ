"""
Utility functions for loading and saving data to HDF5 files
"""

from datetime import date
from typing import Generator, Iterable
from loguru import logger

import h5py as h5
import numpy as np
from tqdm import tqdm

from ..utils import curry, logger_wraps
from .datatypes import PatientScan, PatientScanPreprocessed, MaskDict


@logger_wraps(level="INFO")
@curry
def save_scans_to_h5(
    dataset: Iterable[PatientScan] | Iterable[PatientScanPreprocessed],
    path: str = "./data/dataset.h5",
) -> None:
    """
    Save a list of patient scan dictionaries to an h5 file with patient ID as key
    """

    @curry
    def create_dataset(
        dict_: PatientScan | PatientScanPreprocessed | MaskDict,
        name: str,
        hf: h5.File | h5.Group,
    ) -> None:
        group = hf.create_group(name)
        for key, val in dict_.items():
            match val:
                case x if isinstance(x, str | int):
                    group[key] = val
                case x if isinstance(x, date):
                    group[key] = str(val)
                case x if isinstance(x, dict):
                    create_dataset(val, key, group)
                case x if isinstance(x, np.ndarray | tuple):
                    group.create_dataset(key, data=val, compression="gzip")
                case _:
                    logger.warning(f"Unsupported type {type(val)} for key {key}, skipping")

    with h5.File(path, "w") as hf:
        for scan in tqdm(dataset, desc="Saving to H5"):
            create_dataset(scan, name=f"{scan["patient_id"]}", hf=hf)


@logger_wraps(level="INFO")
def load_scans_from_h5(
    path: str,
) -> Generator[PatientScan, None, None] | Generator[PatientScanPreprocessed, None, None]:
    """
    Load patient scan dictionaries from an h5 file
    """
    with h5.File(path, "r") as hf:
        for key in hf.keys():
            scan = hf[key]
            yield {
                "patient_id": scan['patient_id'][()], # type: ignore
                "volume": scan['volume'][:], # type: ignore
                "dimension_original": tuple(scan['dimension_original'][()]), # type: ignore
                "spacings": tuple(scan['spacings'][()]), # type: ignore
                "modality": scan['modality'][()].decode(), # type: ignore
                "manufacturer": scan['manufacturer'][()].decode(), # type: ignore
                "scanner": scan['scanner'][()].decode(), # type: ignore
                "study_date": date.fromisoformat(scan['study_date'][()].decode()), # type: ignore
                "masks": {
                    organ: scan[f"masks/{organ}"][:] # type: ignore
                    for organ in scan["masks"].keys() # type: ignore
                },
            } 
