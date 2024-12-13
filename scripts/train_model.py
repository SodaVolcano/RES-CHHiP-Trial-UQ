"""
Train model(s) on a h5 dataset.

usage - training a single unet model on fold 0:
    uv run python3 path/to/train_model.py --fold 0 unet
usage - training 3 unet models on fold 2:
    uv run python3 path/to/train_model.py --fold 2 unet_3
usage - training 3 unet models and 1 confidnet on all folds:
    uv run python3 path/to/train_model.py unet_3 confidnet
"""

import sys
from pathlib import Path

import torch
from __helpful_parser import HelpfulParser
from loguru import logger

sys.path.append("..")
sys.path.append(".")

from uncertainty import configuration
from uncertainty.training import (
    H5Dataset,
    SegmentationData,
    init_training_dir,
    read_fold_splits_file,
    train_models,
)
from uncertainty.utils import config_logger


def train_one_fold(
    fold_idx: int,
    h5_path: str,
    config: dict,
    models: list[str],
    fold_split_path: str | Path,
    ckpt_path: str | Path,
):
    train_indices, val_indices, seed = read_fold_splits_file(fold_split_path, fold_idx)
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


def setup(h5_path: str, train_dir: str, config_path: str, config: dict):
    torch.set_float32_matmul_precision("medium")
    torch.autograd.set_detect_anomaly(True)

    # split into test set
    dataset = H5Dataset(h5_path)
    res = init_training_dir(train_dir, config_path, dataset.indices, **config)

    assert (
        res != None
    ), f"Failed to initialise training directory, specified configuration file at {config_path} already exist in training directory!"
    _, train_test_path, fold_split_path, fold_dirs = res
    return fold_split_path, checkpoint_paths


def train_models_for_one_fold(
    config: dict,
    config_path: str,
    h5_path: str,
    train_dir: str,
    fold_idx: int,
    models: list[str],
):
    fold_split_path, checkpoint_paths = setup(h5_path, train_dir, config_path, config)
    train_one_fold(
        fold_idx, h5_path, config, models, fold_split_path, checkpoint_paths[fold_idx]
    )


def train_models_for_all_folds(
    config: dict,
    config_path: str,
    h5_path: str,
    train_dir: str,
    models: list[str],
):
    fold_split_path, checkpoint_paths = setup(h5_path, train_dir, config_path, config)

    for fold_idx, ckpt_path in enumerate(checkpoint_paths):
        train_one_fold(fold_idx, h5_path, config, models, fold_split_path, ckpt_path)


if __name__ == "__main__":
    parser = HelpfulParser(description="Train model(s) on a dataset h5 files.")
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
        required=False,
    )
    parser.add_argument(
        "--train-dir",
        "-o",
        type=str,
        help="Model training directory to save or load model checkpoints. A folder container folders for each fold will be created, and the checkpoints for each model will be saved in the respective fold folder. If not provided, the checkpoint_dir from the configuration file will be used.",
        required=False,
    )
    parser.add_argument(
        "--logging",
        "-l",
        action="store_true",
        help="Enable logging. Default is True.",
        default=True,
    )
    parser.add_argument(
        "models",
        type=str,
        nargs="+",
        help="List of model names to train. To train multiple models with the same name, specify quantity as '<model_name>_<int>. e.g. 'unet_3' will train 3 UNet models.",
    )
    parser.add_argument(
        "--fold",
        "-f",
        type=int,
        help="Fold number to train the model on. If not specified, all folds will be trained.",
        required=False,
    )

    args = parser.parse_args()
    config = configuration(args.config)

    if args.logging:
        logger.enable("uncertainty")
        config_logger(**config)

    if args.fold is not None:
        train_models_for_one_fold(
            config,
            args.config,
            args.in_path or config["data__h5_path"],
            args.train_dir or config["training__train_dir"],
            args.fold,
            args.models,
        )
    else:
        train_models_for_all_folds(
            config,
            args.config,
            args.in_path or config["data__h5_path"],
            args.train_dir or config["training__train_dir"],
            args.models,
        )
