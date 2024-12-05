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
from __helpful_parser import HelpfulParser
from lightning.pytorch.loggers import TensorBoardLogger

from uncertainty import configuration
from uncertainty.utils import config_logger

"""
1. init_training(train_dir: str, config_path: str, n_folds):
    0. if not exist: create train_dir
    1. if config exists and config_path != that config, FAIL, else copy config over -> configuration.yaml
    2. if exist, use it, else perform test-train split -> train-test-split.pkl
    3. if exist, use it, else perform k-fold split -> validation_folds.pkl
    4. if not exist: create checkpoint folders for each fold
    5. train models for each fold paths
    
2. Train models(models: list[str], checkpoint_dir: str, train_val_split, config)
    1. for each model in models:
        - parse quantity and model class from model
        - create model folder names using model name and quantity  [model_name_1, model_name_2, ...]
            [(model_fn, "model_name_1"), (model_fn, "model_name_2"), ...]
        - for each model folder:
            - create model folder
            - train_model
                - init model class
                - init data class using train_val_split
                - train!



"""


def main(config: dict, in_path: str, checkpoint_dir: str):
    torch.set_float32_matmul_precision("medium")
    torch.autograd.set_detect_anomaly(True)

    # split into test set

    # split data into folds
    # for each fold,
    # set the path to checkpoint
    # get the model function and init it
    # make dataset using only those indices
    # train model


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
        "--checkpoint-dir",
        "-o",
        type=str,
        help="Directory to save or load model checkpoints. A folder container folders for each fold will be created, and the checkpoints for each model will be saved in the respective fold folder. If not provided, the checkpoint_dir from the configuration file will be used.",
        optional=True,
    )
    parser.add_argument(
        "--logging",
        "-l",
        action="store_true",
        help="Enable logging. Default is True.",
        default=True,
    )
    # parser.add_argument(
    #     "--retrain",
    #     action="store_true",
    #     help="Whether to retrain a model saved in the model checkpoint",
    #     default=False,
    # )

    args = parser.parse_args()
    config = configuration(args.config)

    if args.logging:
        logger.enable("uncertainty")
        config_logger(**config)

    # if args.retrain:
    #     with open(os.path.join(checkpoint_path, "config.pkl"), "rb") as f:
    #         config = dill.load(f)

    # TODO: add retrain argument
    main(
        config=config,
        in_path=args.in_path or config["data__h5_path"],
        checkpoint_dir=args.checkpoint_path or config["training__checkpoint_dir"],
        # retrain=args.retrain,
    )
