import os

import sys
import torch
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from loguru import logger

sys.path.append("..")
sys.path.append(".")
import dill
import torch
from lightning.pytorch.loggers import TensorBoardLogger
from uncertainty import configuration
from uncertainty.utils import config_logger


from __helpful_parser import HelpfulParser


def main(config: dict, in_path: str, checkpoint_path: str):
    torch.set_float32_matmul_precision("medium")
    torch.autograd.set_detect_anomaly(True)
    pass


if __name__ == "__main__":
    parser = HelpfulParser(description="Train a model on a dataset h5 files.")
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        help="Path to the configuration file.",
        default="configuration.yaml",
    )
    parser.add_argument(
        "--in-path",
        type=str,
        help="Path to the H5 dataset of PatientScan dictionaries. If not provided, the h5_path from the configuration file will be used.",
        optional=True,
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        help="Path to save or load model checkpoints. A folder container folders for each fold will be created, and the checkpoints for each model will be saved in the respective fold folder. If not provided, the checkpoint_dir from the configuration file will be used.",
        optional=True,
    )
    # parser.add_argument(
    #     "--retrain",
    #     action="store_true",
    #     help="Whether to retrain a model saved in the model checkpoint",
    #     default=False,
    # )
    args = parser.parse_args()
    config = configuration(args.config)
    parser.add_argument(
        "--logging",
        "-l",
        action="store_true",
        help="Enable logging. Default is True.",
        default=True,
    )
    if args.logging:
        logger.enable("uncertainty")
        config_logger()

    # if args.retrain:
    #     with open(os.path.join(checkpoint_path, "config.pkl"), "rb") as f:
    #         config = dill.load(f)

    # TODO: add retrain argument
    main(
        config=config,
        in_path=args.in_path,
        checkpoint_path=args.checkpoint_path,
        # retrain=args.retrain,
    )
