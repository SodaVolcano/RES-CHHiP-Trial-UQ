"""
Datatype definitions for masks and patient scans
"""

from datetime import date
from typing import Annotated, TypedDict

import numpy as np

MaskDict = dict[Annotated[str, "organ name"], np.ndarray]

PatientScan = TypedDict(
    "PatientScan",
    {
        "patient_id": int,
        "volume": np.ndarray,
        "dimension_original": tuple[int, int, int],
        "spacings": tuple[float, float, float],
        "modality": str,
        "manufacturer": str,
        "scanner": str,
        "study_date": date,
        "masks": MaskDict,
    },
)

PatientScanPreprocessed = TypedDict(
    "PatientScanPreprocessed",
    {
        "patient_id": int,
        "volume": np.ndarray,
        "dimension_original": tuple[int, int, int],
        "spacings": tuple[float, float, float],
        "modality": str,
        "manufacturer": str,
        "scanner": str,
        "study_date": date,
        "masks": np.ndarray,
        "organ_ordering": list[str],
    },
)
