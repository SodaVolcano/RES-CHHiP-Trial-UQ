"""
Collection of global constants
"""

from typing import Final

import tensorflow.keras as keras
from toolz import memoize

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


@memoize
def model_config() -> dict:
    config = {
        # --------- Data settings ---------
        "data_dir": "/content/gdrive/MyDrive/dataset/Data",
        "input_width": 500,
        "input_height": 500,
        "input_depth": 360,
        "input_dim": 3,  # Number of organs
        # ------- ConvolutionBlock settings  --------
        "kernel_size": (3, 3),
        "n_convolutions_per_block": 1,
        "use_batch_norm": False,
        "activation": "gelu",
        # ------- Encoder/Decoder settings -------
        # Number of kernels in first level of Encoder, doubles/halves at each level in Encoder/Decoder
        "n_kernels_init": 64,
        # Number of resolutions/blocks; height of U-Net
        "n_levels": 5,
        # Number of class to predict
        "n_kernels_last": 1,
        # Use sigmoid if using binary crossentropy, softmax if using categorical crossentropy
        # Note if using softmax, n_kernels_last should be equal to number of classes (e.g. 2 for binary segmentation)
        # If None, loss will use from_logits=True
        "final_layer_activation": "sigmoid",
        # ------- Overall U-Net settings -----------------
        "model_checkpoint_path": "./checkpoints/checkpoint.model.keras",
        "n_epochs": 1,
        "batch_size": 32,
        "metrics": ["accuracy"],
        "initializer": "he_normal",  # For kernel initialisation
        "optimizer": keras.optimizers.Adam,
        "loss": keras.losses.BinaryCrossentropy,
        # Learning rate scheduler, decrease learning rate at certain epochs
        "lr_scheduler": keras.optimizers.schedules.PiecewiseConstantDecay,
        # Percentage of training where learning rate is decreased
        "lr_schedule_percentages": [0.2, 0.5, 0.8],
        # Gradually decrease learning rate, starting at first value
        "lr_schedule_values": [3e-4, 1e-4, 1e-5, 1e-6],
    }
    # Explicitly calculate number of kernels per block (for Encoder)
    config["n_kernels_per_block"] = [
        config["n_kernels_init"] * (2**block_level)
        for block_level in range(config["n_levels"])
    ]

    return config


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
