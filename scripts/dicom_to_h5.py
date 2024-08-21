"""
Load and save folders of DICOM files to h5 files.
"""

import argparse
import sys
from typing import override
from loguru import logger
from context import uncertainty as un


def main(
    dicom_path: str, save_path: str, preprocess: bool, n_workers: int, logging: bool
):

    if logging:
        logger.enable("uncertainty")
        un.utils.logging.config_logger()

    un.data.dicom.save_dicom_scans_to_h5(
        dicom_path, save_path, n_workers=n_workers, preprocess=preprocess
    )


if __name__ == "__main__":
    config = un.config.configuration()

    class HelpfulParser(argparse.ArgumentParser):
        """
        Print help message when an error occurs.
        """

        @override
        def error(self, message):
            sys.stderr.write("error: %s\n" % message)
            self.print_help()
            sys.exit(2)

    parser = HelpfulParser(
        description="Load and save folders of DICOM files to h5 files."
    )
    parser.add_argument(
        "--in_path",
        type=str,
        help="Path to the folder containing folders of DICOM files.",
        default=config["data_dir"],
    )
    parser.add_argument(
        "out_path",
        type=str,
        help="Output directory to store the h5 files.",
        default=config["staging_dir"],
    )
    parser.add_argument(
        "--preprocess",
        action="store_true",
        help="Whether to preprocess the volume and the masks (convert to the HU scale and make them isotropic). Will take significant resources. Default is True.",
        default=True,
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes to use. Set to 1 to disable multiprocessing. Default is 1.",
    )
    parser.add_argument(
        "--logging",
        action="store_true",
        help="Enable logging, output is saved to /logs. Default is False.",
        default=True,
    )

    args = parser.parse_args()

    main(args.in_path, args.out_path, args.preprocess, args.workers, args.logging)
