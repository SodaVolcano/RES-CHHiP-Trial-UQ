from context import uncertainty as un
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning import Trainer
import os
import torch

import dill
from scripts.__helpful_parser import HelpfulParser
from uncertainty.config import Configuration
import torch
from lightning.pytorch.loggers import TensorBoardLogger


def main():
    pass


if __name__ == "__main__":
    config = un.config.configuration()
    parser = HelpfulParser(
        description="Train a model on a dataset of DICOM files or h5 files."
    )
    parser.add_argument(
        "--data_path",
        type=str,
        help="Path to the dataset.h5 file containing list of (x, y) pairs.",
        default=os.path.join(config["staging_dir"], config["staging_fname"]),
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        help="Path to save or load model checkpoints.",
        default=config["model_checkpoint_path"],
    )
    parser.add_argument(
        "--retrain",
        action="store_true",
        help="Whether to retrain a model saved in the model checkpoint",
        default=False,
    )
    parser.add_argument(
        "--no_deep_supervision",
        action="store_false",
        help="Disable deep supervision during training.",
        default=not config["deep_supervision"],
    )

    args = parser.parse_args()
    config["staging_dir"] = os.path.dirname(args.data_path)
    config["staging_fname"] = os.path.basename(args.data_path)
    checkpoint_path = args.checkpoint_path
    if args.retrain:
        with open(os.path.join(checkpoint_path, "config.pkl"), "rb") as f:
            config = dill.load(f)

    main(
        config,
        checkpoint_path,
        args.retrain,
        deep_supervision=not args.no_deep_supervision,
    )
