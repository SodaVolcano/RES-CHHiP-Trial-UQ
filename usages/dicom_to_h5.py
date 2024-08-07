"""
Load and save folders of DICOM files to h5 files.
"""

import argparse

from uncertainty.data.dicom import save_dicom_scans_to_h5
from loguru import logger
from uncertainty.utils.logging import config_logger


def main(
    dicom_path: str, save_path: str, preprocess: bool, n_workers: int, logging: bool
):
    if logging:
        logger.enable("uncertainty")
        config_logger()

    save_dicom_scans_to_h5(
        dicom_path, save_path, n_workers=n_workers, preprocess=preprocess
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Load and save folders of DICOM files to h5 files."
    )
    parser.add_argument(
        "in_path",
        type=str,
        help="Path to the folder containing folders of DICOM files.",
    )
    parser.add_argument(
        "out_path", type=str, help="Output directory to store the h5 files."
    )
    parser.add_argument(
        "--preprocess",
        action="store_true",
        help="Whether to preprocess the volume and the masks (convert to the HU scale and make them isotropic). Will take significant resources.",
        default=True,
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes to use. Set to 1 to disable multiprocessing.",
    )
    parser.add_argument(
        "--logging",
        action="store_true",
        help="Enable logging, output is saved to /logs.",
        default=True,
    )

    args = parser.parse_args()

    main(args.in_path, args.out_path, args.preprocess, args.workers, args.logging)
