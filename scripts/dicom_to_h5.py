"""
Load and save folders of DICOM files to h5 file(s).
"""

from loguru import logger
from context import uncertainty as un
from __helpful_parser import HelpfulParser


def main(
    dicom_path: str, save_path: str, preprocess: bool, n_workers: int, logging: bool
):

    if logging:
        logger.enable("uncertainty")
        un.utils.logging.config_logger()

    scans = un.data.dicom.load_patient_scans(dicom_path, preprocess=preprocess)
    if preprocess:
        un.data.preprocessing.preprocess_dataset(scans, n_workers=n_workers)
        un.data.h5.save_xy_to_h5(scans, save_path)
        return

    un.data.h5.save_scan_to_h5(scans, save_path)


if __name__ == "__main__":
    config = un.config.configuration()

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
        help="Whether to preprocess the volume and the masks. Will take significant resources. If True, a single h5 file containing (volume, mask) pairs is produced. If False, multiple h5 files for each raw PatientScan object is produced. Default is True.",
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
