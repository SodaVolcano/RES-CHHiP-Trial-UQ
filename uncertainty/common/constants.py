"""
Collection of global constants
"""

from typing import Final
import nptyping as npt

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

VolumeType: Final = npt.NDArray[npt.Shape["* z, * x, * y"], npt.Float]
MaskType: Final = npt.NDArray[npt.Shape["* z, * x, * y"], npt.Bool]

# Prevent polluting the namespace
del npt
