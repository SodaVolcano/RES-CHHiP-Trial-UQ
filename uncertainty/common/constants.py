"""
Collection of global constants
"""

from typing import Final
import numpy as np

# SOP Class UIDs for different types of DICOM files
# https://dicom.nema.org/dicom/2013/output/chtml/part04/sect_B.5.html
CT_IMAGE: Final[str] = "1.2.840.10008.5.1.4.1.1.2"
RT_STRUCTURE_SET: Final[str] = "1.2.840.10008.5.1.4.1.1.481.3"
RT_DOSE: Final[str] = "1.2.840.10008.5.1.4.1.1.481.2"
RT_PLAN: Final[str] = "1.2.840.10008.5.1.4.1.1.481.5"

# regex pattern to match any valid python identifier names
VALID_IDENTIFIER: Final[str] = "[a-zA-Z_][a-zA-Z0-9_]*"

# Hounsfield Units (HU), intensity range for CT images
# wrONG
CT_RANGE: Final[tuple[int, int]] = (-1000, 30)

# ROI keep lists, all names containing these as substring will be kept
ROI_KEEP_LIST: Final[list[str]] = [
    "bladder",
    "rectum",
    "p+sv",
    "pros",
    "prossv",
    "p_only",
    "p_+_base_sv",
    "p_+_sv",
    "ctv",
]

ROI_EXCLUSION_LIST: Final[list[str]] = [
    "ptv",
    "gtv",
    "bowel",
    "trigone",
    "ant_block",
    "boost",
    "arrow",
    "hip",
    "fem",
    "llat",
    "dose",
    "rfh",
    "surface",
    "lfh",
    "body",
    "copy",
    "seed",
    "bulb",
    "hot",
    "exactigrt_thick",
    "sigmoid",
    "gas",
    "bone",
    "couchouter",
    "tattoo",
    "old",
    "103%",
    "ref",
    "rlat",
    "105%",
    "target",
    "pb",
    "do_not_use",
    "ureter",
    "gtc_cds",
    "external",
    "herniae",
    "patient_outline",
    "ac",
    "ub",
    "tatt",
    "recover",
    "was_",
    "couchinner",
    "air",
    "s1",
    "s2",
    "s3",
    "anal_canal",
]
