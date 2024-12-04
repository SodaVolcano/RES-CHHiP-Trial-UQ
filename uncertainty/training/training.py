"""
Functions for managing training, such as checkpointing and data splitting.
"""

import os
import pickle
import shutil
from pathlib import Path
from random import randint
from typing import Iterable, Literal, Sequence, TypedDict

import dill
import lightning
import toolz as tz
import torch
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from loguru import logger
from sklearn.model_selection import KFold
from toolz import curried
from torch import nn

from uncertainty.config import auto_match_config

from ..models.unet import UNet
from ..utils import curry, logger_wraps
from ..utils.common import unpack_args
from .datasets import H5Dataset, SegmentationData
from .lightning import LitSegmentation


@auto_match_config(prefixes=["training", "data"])
def train_model(
    model: lightning.LightningModule,
    dataset: SegmentationData,
    # Logging parameters
    log_dir: str | Path,
    experiment_name: str,
    # Checkpoint parameters
    checkpoint_path: str,
    checkpoint_name: str,
    checkpoint_every_n_epochs: int,
    # Training parameters
    n_epochs: int,
    n_batches_per_epoch: int,
    n_batches_val: int,
    check_val_every_n_epoch: int,
    save_last: bool = True,
    strategy: str = "ddp",
    accelerator: str = "gpu" if torch.cuda.is_available() else "cpu",
    precision: str = "16-mixed",
    enable_progress_bar: bool = True,
    enable_model_summary: bool = True,
):
    """
    Train a model and save checkpoints in `<checkpoint_path>/<experiment_name>/<model_classname>-<int>`.

    Parameters
    ----------
    model : lightning.LightningModule
        The PyTorch Lightning model to be trained. Should define the training, validation, and optional test steps.
    dataset : SegmentationData
        The dataset containing training and validation dataloaders.
    log_dir : str | Path
        Directory path where TensorBoard logs will be saved. Used for monitoring training progress and metrics.
    experiment_name : str
        Name of the experiment, which will be used to organise logs under the specified `log_dir`.
    checkpoint_path : str
        Directory path where model checkpoints will be saved.
    checkpoint_name : str
        Base name for the checkpoint files. Each checkpoint will be saved with this name followed by an epoch or step indicator.
    checkpoint_every_n_epochs : int
        Frequency (in epochs) at which model checkpoints are saved during training.
    n_epochs : int
        Total number of epochs to train the model.
    n_batches_per_epoch : int
        Number of training batches to run per epoch. Useful for limiting training iterations in debugging or resource-constrained settings.
    n_batches_val : int
        Number of batches to use during validation. Limits the validation set size for efficiency.
    check_val_every_n_epoch : int
        Frequency (in epochs) at which validation is performed. A value of 0 disables validation.
    save_last : bool, optional
        Whether to save the final model checkpoint at the end of training, even if it's not the best-performing one. Default is `True`.
    strategy : str, optional
        Specifies the distributed training strategy, see https://lightning.ai/docs/pytorch/stable/extensions/strategy.html
    accelerator : str, optional
        Device type to run the model on. Options include `"gpu"`, `"cpu"`, or `"tpu"`. Defaults to `"gpu"` if available.
    precision : str, optional
        Training precision of float values.
    enable_progress_bar : bool, optional
        Whether to show a progress bar during training.
    enable_model_summary : bool, optional
        Whether to print a model summary before training starts.
    """
    run_validation = check_val_every_n_epoch > 0

    tb_logger = TensorBoardLogger(save_dir=log_dir, name=experiment_name)

    model_checkpoint = curry(ModelCheckpoint)(
        monitor="val_loss" if run_validation else "train_loss",
        mode="min",
        dirpath=checkpoint_path,
        save_on_train_epoch_end=not run_validation,
    )

    checkpoint = model_checkpoint(
        filename=checkpoint_name,
        every_n_epochs=checkpoint_every_n_epochs,
        save_top_k=-1,
    )
    checkpoint_last = model_checkpoint(
        filename="last",
        save_top_k=1,
    )

    trainer = Trainer(
        max_epochs=n_epochs,
        limit_train_batches=n_batches_per_epoch,
        limit_val_batches=n_batches_val if run_validation else 0,
        num_sanity_val_steps=0 if not run_validation else 2,
        callbacks=[checkpoint, checkpoint_last] if save_last else [checkpoint],
        strategy=strategy,
        check_val_every_n_epoch=check_val_every_n_epoch,
        accelerator=accelerator,
        precision=precision,  # type: ignore
        enable_progress_bar=enable_progress_bar,
        enable_model_summary=enable_model_summary,
        logger=tb_logger,
    )
    # TODO: add retrain
    # if retrain:
    #     trainer.fit(model, data, ckpt_path=checkpoint_path)
    # else:
    trainer.fit(model, dataset)


DataSplitDict = TypedDict(
    "DataSplitDict", {"train": list[int], "val": list[int], "seed": int}
)


@logger_wraps(level="INFO")
def split_into_folds[
    T
](dataset: Sequence[T], n_folds: int, return_indices: bool = False) -> Iterable[
    tuple[list[int] | Sequence[T], list[int] | Sequence[T]]
]:
    """
    Split a dataset into `n_folds` folds and return an Iterable of the split.

    Parameters
    ----------
    dataset : Sequence
        The dataset to split.
    n_folds : int
        The number of folds to split the dataset into.
    return_indices : bool, optional
        If True, return the indices of the dataset split instead of the actual data.
    """
    if n_folds <= 1:
        logger.warning(
            f"Number of folds specified is {n_folds}. No splitting is performed."
        )
        split = [list(range(len(dataset)))] if return_indices else [dataset]
        return zip(split, [[]])  # validation set is empty

    indices = KFold(n_splits=n_folds).split(dataset)  # type: ignore
    if return_indices:
        # cast from numpy array to list
        return map(
            unpack_args(lambda train, val: (train.tolist(), val.tolist())), indices
        )
    get_with_indices = lambda indices: [dataset[i] for i in indices]
    return map(
        unpack_args(
            lambda train_idx, val_idx: (
                get_with_indices(train_idx),
                get_with_indices(val_idx),
            )
        ),
        indices,
    )


@logger_wraps(level="INFO")
def write_training_fold_file(
    path: str | Path,
    fold_indices: Iterable[tuple[list[int], list[int]]],
    seed: bool = True,
    force: bool = False,
) -> None:
    """
    Write a file containing indices of each fold split (and optionally, fold-specific seed)

    Parameters
    ----------
    path : str
        The path to write the file to.
    fold_indices : Iterable[tuple[list[int], list[int]]]
        An iterable of tuples containing the training and validation indices for each fold.
    seed : bool, optional
        If True, add a random seed to each fold.
    force : bool, optional
        If True, overwrite the file if it exists.
    """
    content = tz.pipe(
        {
            f"fold_{i}": {"train": train, "val": val}
            for i, (train, val) in enumerate(fold_indices)
        },
        # add seed to each fold if specified
        (curried.valmap(lambda x: x | {"seed": randint(0, 2**32 - 1)} if seed else x)),
    )

    if Path(path).exists():
        if not force:
            logger.warning(
                f"File {path} already exists. If you wish to ovewrite, use `force=True`."
            )
            return

    with open(path, "wb") as f:
        pickle.dump(content, f)


def read_training_fold_file(
    path: str | Path,
    fold: int | None = None,
) -> DataSplitDict | dict[str, DataSplitDict]:
    """
    Read configuration for a fold from the training fold file at `path`.

    If `fold` is not provided, the entire file is read and returned.
    """
    with open(path, "rb") as f:
        content = pickle.load(f)
    if fold is not None:
        return content[f"fold_{fold}"]
    return content


@logger_wraps(level="INFO")
def init_checkpoint_dir(
    checkpoint_path: str,
    config_path: str,
    n_folds: int,
    dataset: Sequence,
    force: bool = False,
) -> tuple[Path, list[Path]] | None:
    """
    Initialise the checkpoint directory.

    Parameters
    ----------
    checkpoint_path : str
        The path to the checkpoint directory.
    config_path : str
        The path to the configuration file to be copied.
    n_folds : int
        The number of folds to split the dataset into.
    dataset : Sequence
        The dataset to split into folds.
    force : bool, optional
        If True, overwrite the checkpoint directory if it exists.

    Returns
    -------
    tuple[Path, list[Path]] | None
        A tuple containing the data split path and a list of fold directories.
        If the checkpoint directory already exists and `force` is False, None is returned.
    """
    checkpoint_dir = Path(checkpoint_path)

    if checkpoint_dir.exists():
        if not force:
            logger.warning(
                f"Directory {checkpoint_dir} already exists. If you wish to ovewrite, use `force=True`."
            )
            return
        shutil.rmtree(checkpoint_dir)  # Remove the existing directory if force is True

    # Create the checkpoint directory and fold subdirectories
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    for i in range(1, n_folds + 1):
        (checkpoint_dir / f"fold_{i}").mkdir()

    config_name = os.path.basename(config_path)
    shutil.copy(config_path, checkpoint_dir / config_name)

    # Split dataset into folds and write to validation_folds.pkl
    fold_indices = split_into_folds(dataset, n_folds, return_indices=True)
    data_split_path = checkpoint_dir / "validation_folds.pkl"
    write_training_fold_file(data_split_path, fold_indices)  # type: ignore

    return data_split_path, list(checkpoint_dir.glob("fold_*"))


def load_checkpoint(
    ckpt_dir: None | str = None,
    model_path: str | None = None,
    config_path: str | None = None,
    indices_path: str | None = None,
) -> tuple[
    nn.Module,
    dict,
    dict[Literal["train_indices", "val_indices"], list[int]],
    H5Dataset,
    H5Dataset,
]:
    """
    Load the model, configuration, and its dataset from a checkpoint folder.

    Parameters
    ----------
    ckpt_dir : str, optional
        Folder containing "latest.ckpt", "config.pkl", and "indices.pt" for
        the model weights, configuration object, and train-validation split
        indices respectively. If they are not named as those files, pass in
        `model_path`, `config_path`, and `indices_path` instead of this parameter.
    model_path : str, optional
        Path to the model checkpoint file. If not provided, defaults to
        "latest.ckpt" in the `ckpt_dir`.
    config_path : str, optional
        Path to the configuration file. If not provided, defaults to
        "config.pkl" in the `ckpt_dir`.
    indices_path : str, optional
        Path to the indices file. If not provided, defaults to
        "indices.pt" in the `ckpt_dir`.

    Returns
    -------
    tuple
        A tuple containing:
        - ckpt : The loaded model checkpoint.
        - config : The configuration object.
        - indices: A dictionary with keys "train_indices" and "val_indices" with a list
                   of integers indicating indices in the total dataset allocated to the
                   train and validation set respectively.
        - train_dataset: H5Dataset containing the training data.
        - val_dataset: H5Dataset containing the validation data.

    Raises
    ------
    FileNotFoundError
        If any of the specified paths do not exist.
    """
    config_path = config_path or os.path.join(ckpt_dir, "config.pkl")
    indices_path = indices_path or os.path.join(ckpt_dir, "indices.pt")
    model_path = model_path or os.path.join(ckpt_dir, "latest.ckpt")
    with open(config_path, "rb") as f:
        config = dill.load(f)
    indices = torch.load(indices_path, weights_only=True)
    train_dataset = H5Dataset(
        os.path.join(config["staging_dir"], config["train_fname"]),
        indices=indices["train_indices"],
    )
    val_dataset = H5Dataset(
        os.path.join(config["staging_dir"], config["train_fname"]),
        indices=indices["val_indices"],
    )
    model = UNet(config)
    model = LitSegmentation.load_from_checkpoint(
        model_path, model=model, config=config, save_hyperparams=False
    ).model
    return model, config, indices, train_dataset, val_dataset


def checkpoint_dir_type(
    path,
    required_files: list[str] | set[str] = ["latest.ckpt", "indices.pt", "config.pkl"],
) -> Literal["single", "multiple", "invalid"]:
    """
    Return type of the checkpoint directory - "single", "multiple", or "invalid"

    This function returns if the given directory contains a single model's checkpoint or
    folders of model checkpoints. "single" is defined as a folder with at least the
    required files while "multiple" is defined as a directory without the required files,
    but have directories that **all** have the required files.

    Parameters
    ----------
    path : str
        The path to the checkpoint directory to be validated.

    Returns
    -------
    Literal["single", "multiple", "invalid"]
        'single' if all required files are found in the main directory,
        'multiple' if required files are found in subdirectories,
        and "invalid" if otherwise.
    """
    required_files = set(required_files)

    if not os.path.isdir(path):
        logger.critical(f"The path {path} is not a valid directory.")
        return "invalid"

    if required_files.issubset(os.listdir(path)):
        return "single"

    subdir_paths = [os.path.join(path, entry) for entry in os.listdir(path)]
    bad_dirs = list(
        filter(
            lambda x: not (os.path.isdir(x) and required_files.issubset(os.listdir(x))),
            subdir_paths,
        )
    )
    if not len(bad_dirs) == 0:
        logger.critical(
            f"Folders of checkpoint detected but the following folder have bad structure: {bad_dirs}"
        )
        return "invalid"

    return "multiple"
