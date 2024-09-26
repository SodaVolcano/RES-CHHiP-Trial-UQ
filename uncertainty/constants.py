"""
Collection of global constants
"""

from typing import Final

# SOP Class UIDs for different types of DICOM files
# https://dicom.nema.org/dicom/2013/output/chtml/part04/sect_B.5.html
CT_IMAGE: Final[str] = "1.2.840.10008.5.1.4.1.1.2"
RT_STRUCTURE_SET: Final[str] = "1.2.840.10008.5.1.4.1.1.481.3"
RT_DOSE: Final[str] = "1.2.840.10008.5.1.4.1.1.481.2"
RT_PLAN: Final[str] = "1.2.840.10008.5.1.4.1.1.481.5"

# regex pattern to match any valid python identifier names
VALID_IDENTIFIER: Final[str] = "[a-zA-Z_][a-zA-Z0-9_]*"

# Hounsfield Units (HU), intensity range for CT images
HU_RANGE: Final[tuple[int, int]] = (-1000, 3000)
# Threshold value (HU) to binarise the scan to get mask of the body
BODY_THRESH = -800


# ROI keep list, all names containing these as substring will be kept
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

# ROI exclusion list, remove ROI name before matching keep list
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


# list of ROI name variants in the patient scans for organs of interest
# Preprocessed masks will have same name order as this list
ORGAN_MATCHES: Final[dict[str, list[str]]] = {
    "prostate": [
        "prostate",
        "prostate+sv",
        "ctv1",
        "ctv2",
        "ctv3",
        "prostate_+_sv",
        "p+sv",
        "prostate_&_sv",
        "pros_new",
        "prossv_new",
        "p_+_base_sv",
        "p_only_js",
        "prostate_only",
        "prostate_sv",
        "prostate_and_base_svs",
        "prostate_only",
        "prostate_alone",
        "prostate_and_svs",
        "p_+_sv",
        "2_prostate",
        "3prost_semves",
        "2_prost_sv",
        "3prostate",
        "ctv_pros+sv",
        "ctv_pros_only",
        "pros+sem",
        "prostate_only",
        "prostate_and_sv",
    ],
    "bladder": ["bladder", "bladder_jp", "bladder_c", "bladder_db"],
    "rectum": [
        "rectum",
        "rectum_kc",
        "rectum_aw",
        "rectum_kc",
        "rectumaw_kc",
        "rectum,_nos",
        "rectum_kf",
        "rectumaw_kf",
        "rectum_jp",
        "rectumaw_jp",
        "rectum_rb",
        "rectumaw_rb",
    ],
}
