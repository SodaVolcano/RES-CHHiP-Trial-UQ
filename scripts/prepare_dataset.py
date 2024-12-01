"""
Load folders of DICOM files to H5, preprocess, and perform data splitting.
"""

import sys

from loguru import logger

sys.path.append("..")
sys.path.append(".")
from __helpful_parser import HelpfulParser
from uncertainty.utils import config_logger
from uncertainty.config import configuration
from uncertainty.data import (
    preprocess_dataset,
    load_all_patient_scans,
    save_scans_to_h5,
)


def main(dicom_path: str, save_path: str, preprocess: bool, n_workers: int):

    scans = load_all_patient_scans(dicom_path)
    if preprocess:
        scans = preprocess_dataset(scans, n_workers=n_workers)
    save_scans_to_h5(scans, save_path)


if __name__ == "__main__":
    parser = HelpfulParser(
        description="Load and save folders of DICOM files to a h5 file."
    )
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        help="Path to the configuration file.",
        default="configuration.yaml",
    )
    parser.add_argument(
        "--in_path",
        "-i",
        type=str,
        help="Path to the folder containing folders of DICOM files.",
        default=None,
    )
    parser.add_argument(
        "--out_path",
        "-o",
        type=str,
        help="Output path to store the h5 file.",
        default=None,
    )
    parser.add_argument(
        "--preprocess",
        "-p",
        action="store_true",
        help="Whether to preprocess the volume and the masks. Will take significant resources. Default is True.",
        default=True,
    )
    parser.add_argument(
        "--workers",
        "-w",
        type=int,
        default=1,
        help="Number of worker processes to use. Set to 1 to disable multiprocessing. Default is 1.",
    )
    parser.add_argument(
        "--logging",
        "-l",
        action="store_true",
        help="Enable logging. Default is False.",
        default=True,
    )

    args = parser.parse_args()
    config = configuration(args.config)

    if args.logging:
        logger.enable("uncertainty")
        config_logger()

    main(
        args.in_path or config["data__data_dir"],
        args.out_path or config["data__h5_path"],
        args.preprocess,
        args.workers,
    )
