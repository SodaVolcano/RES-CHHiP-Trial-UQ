import os
import sys

import torch
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from loguru import logger

from uncertainty.training.datasets import H5Dataset, SegmentationData
from uncertainty.training.training import (
    init_training_dir,
    read_training_fold_file,
    train_models,
)

sys.path.append("..")
sys.path.append(".")
import dill
import torch
from __helpful_parser import HelpfulParser
from lightning.pytorch.loggers import TensorBoardLogger

from uncertainty import configuration
from uncertainty.utils import config_logger


def main(
    config: dict, h5_path: str, train_dir: str, config_path: str, models: list[str]
):
    torch.set_float32_matmul_precision("medium")
    torch.autograd.set_detect_anomaly(True)

    # split into test set
    dataset = H5Dataset(h5_path)

    res = init_training_dir(train_dir, config_path, dataset.indices, **config)
    assert (
        res != None
    ), f"Failed to initialise training directory, specified configuration file at {config_path} already exist in training directory!"
    _, fold_split_path, checkpoint_paths = res

    for fold_idx, ckpt_path in enumerate(checkpoint_paths):
        train_indices, val_indices, seed = read_training_fold_file(
            fold_split_path, fold_idx
        )
        dataset = SegmentationData(
            h5_path,
            train_indices=train_indices,  # type: ignore
            val_indices=val_indices,  # type: ignore
            **config,
        )
        train_models(
            models,
            dataset=dataset,
            checkpoint_dir=ckpt_path,
            experiment_name=f"fold_{fold_idx}",
            seed=seed,  # type: ignore
            **config,
        )


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
        "-i",
        type=str,
        help="Path to the H5 dataset of PatientScan dictionaries. If not provided, the h5_path from the configuration file will be used.",
        optional=True,
    )
    parser.add_argument(
        "--train-dir",
        "-o",
        type=str,
        help="Model training directory to save or load model checkpoints. A folder container folders for each fold will be created, and the checkpoints for each model will be saved in the respective fold folder. If not provided, the checkpoint_dir from the configuration file will be used.",
        optional=True,
    )
    parser.add_argument(
        "--logging",
        "-l",
        action="store_true",
        help="Enable logging. Default is True.",
        default=True,
    )
    parser.add_argument(
        "--models",
        "-m",
        type=str,
        help="List of models to train, separated by comma.",
        default="",
    )

    args = parser.parse_args()
    config = configuration(args.config)

    if args.logging:
        logger.enable("uncertainty")
        config_logger(**config)

    # TODO: add retrain argument
    main(
        config=config,
        config_path=args.config,
        h5_path=args.in_path or config["data__h5_path"],
        train_dir=args.train_dir or config["training__train_dir"],
        models=args.models.split(",") if args.models else config["training__models"],
    )
