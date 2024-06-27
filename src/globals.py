"""
Collection of global constants
"""

import nptyping as npt

# SOP Class UIDs for different types of DICOM files
# https://dicom.nema.org/dicom/2013/output/chtml/part04/sect_B.5.html
CT_IMAGE = "1.2.840.10008.5.1.4.1.1.2"
RT_STRUCTURE_SET = "1.2.840.10008.5.1.4.1.1.481.3"
RT_DOSE = "1.2.840.10008.5.1.4.1.1.481.2"
RT_PLAN = "1.2.840.10008.5.1.4.1.1.481.5"

# regex pattern to match any valid python identifier names
VALID_IDENTIFIER = "[a-zA-Z_][a-zA-Z0-9_]*"

# Hounsfield Units (HU), intensity range for CT images
# wrONG
CT_RANGE = (-1000, 30)

VolumeType = npt.NDArray[npt.Shape["* z, * x, * y"], npt.Float]
MaskType = npt.NDArray[npt.Shape["* z, * x, * y"], npt.Bool]
