"""
Functions for managing training, such as checkpointing and data splitting.
"""

import os
import pickle
import random
import re
import shutil
from pathlib import Path
from typing import Callable, Iterable, Sequence, TypedDict

import lightning
import toolz as tz
import torch
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from loguru import logger
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split as sk_train_test_split
from toolz import curried

from ..config import auto_match_config, configuration
from ..models import get_model
from ..training import LitModel
from ..utils import list_files, logger_wraps, next_available_path, unpack_args, starmap
from .datasets import SegmentationData

DataSplitDict = TypedDict("DataSplitDict", {"train": list[int], "val": list[int]})
FoldSplitsDict = dict[str, DataSplitDict]
TrainDirDict = dict[str, dict[str, Iterable[LitModel]]]


@auto_match_config(prefixes=["training", "data"])
def train_model(
    model: lightning.LightningModule,
    dataset: SegmentationData,
    # Logging parameters
    log_dir: str | Path,
    experiment_name: str,
    # Checkpoint parameters
    checkpoint_path: str | Path,
    checkpoint_name: str,
    checkpoint_every_n_epochs: int,
    # Training parameters
    n_epochs: int,
    n_batches_per_epoch: int,
    n_batches_val: int,
    check_val_every_n_epoch: int,
    num_sanity_val_steps: int,
    precision: str,
    save_last_checkpoint: bool,
    strategy: str = "ddp",
    accelerator: str = "auto",
    enable_progress_bar: bool = True,
    enable_model_summary: bool = True,
):
    """
    Train a model and save checkpoints in `checkpoint_path`.

    Model checkpoints are saved in the specified `checkpoint_path` directory. The `torch.nn.Module`
    model is saved as `torch-module.pt` in the same directory. A checkpoint called `last.ckpt` is
    saved at the end of training Logs are saved in `log_dir` under the `experiment_name` directory.

    If the checkpoint directory already exists, an integer is appended to the directory name.

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
    num_sanity_val_steps : int
        Number of validation steps to run before training starts. 0 disables the sanity check.
    save_last : bool, optional
        Whether to save the final model checkpoint at the end of training, even if it's not the best-performing one. Default is `True`.
    strategy : str, optional
        Specifies the distributed training strategy, see https://lightning.ai/docs/pytorch/stable/extensions/strategy.html
    accelerator : str, optional
        Device type to run the model on. Options include `"auto"`, `"gpu"`, `"cpu"`, or `"tpu"`. Defaults to `"auto"` if available.
    precision : str, optional
        Training precision of float values.
    enable_progress_bar : bool, optional
        Whether to show a progress bar during training.
    enable_model_summary : bool, optional
        Whether to print a model summary before training starts.
    """
    run_validation = check_val_every_n_epoch > 0

    if (next_path := next_available_path(checkpoint_path)) != checkpoint_path:
        logger.warning(
            f"Checkpoint path {checkpoint_path} already exists. Saving to {next_path} instead."
        )
        checkpoint_path = next_path

    checkpoint_path = Path(checkpoint_path)
    os.makedirs(checkpoint_path)
    torch.save(model.model, checkpoint_path / "torch-module.pt")

    tb_logger = TensorBoardLogger(
        save_dir=log_dir, name=experiment_name, version=checkpoint_path.name
    )

    checkpoint = ModelCheckpoint(
        monitor="val_loss" if run_validation else "train_loss",
        mode="min",
        dirpath=checkpoint_path,
        save_on_train_epoch_end=not run_validation,
        filename=checkpoint_name,
        every_n_epochs=checkpoint_every_n_epochs,
        save_top_k=-1,
    )
    checkpoint_last = ModelCheckpoint(
        monitor="val_loss" if run_validation else "train_loss",
        mode="min",
        dirpath=checkpoint_path,
        save_on_train_epoch_end=not run_validation,
        filename="last",
        save_top_k=1,
    )

    trainer = Trainer(
        max_epochs=n_epochs,
        limit_train_batches=n_batches_per_epoch,
        limit_val_batches=n_batches_val if run_validation else 0,
        num_sanity_val_steps=num_sanity_val_steps,
        callbacks=(
            [checkpoint, checkpoint_last] if save_last_checkpoint else [checkpoint]
        ),
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


@auto_match_config(prefixes=["training", "data", "unet", "confidnet"])
def train_models(
    models: list[str],
    dataset: SegmentationData,
    checkpoint_dir: Path | str,
    experiment_name: str,
    **kwargs,
):
    """
    Train multiple models on the same dataset and save checkpoints in the specified directory.

    Parameters
    ----------
    models : list[str]
        A list of model names to train. Each model name can be a single model name or a string
        with the format "<model_name>_<quantity>", where `<model_name>` is the model class name
        in lowercase and `<quantity>` is the number of models to train. e.g. "unet_3" will train
        3 UNet models.
    dataset : SegmentationData
        The dataset containing training and validation dataloaders.
    checkpoint_dir : Path | str
        Directory path where model checkpoints will be saved.
    experiment_name : str
        Name of the experiment, which will be used to organise logs under the specified `log_dir`.
        e.g. "fold_0", "fold_1", etc.
    **kwargs
        Additional keyword arguments used to initialise the models and configure the training process.
    """
    checkpoint_dir = Path(checkpoint_dir)

    def parse_model_str(model: str) -> list[tuple[Callable, Path]]:
        """Get model function and path where model checkpoints will be stored"""
        if "_" not in model:
            return [(get_model(model), Path(model))]
        return tz.pipe(
            model.split("_"),
            unpack_args(
                lambda name, quantity: [
                    (get_model(name), checkpoint_dir / f"{name}-{i}")
                    for i in range(int(quantity))
                ]
            ),
        )  # type: ignore

    tz.pipe(
        models,
        curried.map(parse_model_str),
        tz.concat,
        starmap(
            lambda model_fn, model_path: (
                train_model(
                    LitModel(model=model_fn(**kwargs), **kwargs),
                    dataset,
                    checkpoint_path=checkpoint_dir / model_path,
                    experiment_name=experiment_name,
                    **kwargs,
                )
            )
        ),
        list,
    )


def _get_with_indices[
    T
](indices: list[int] | list[str], dataset: Sequence[T] | dict[str, T]) -> Sequence[T]:
    return [dataset[i] for i in indices]  # type: ignore


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
        return starmap(lambda train, val: (train.tolist(), val.tolist()), indices)
    return starmap(
        lambda train_idx, val_idx: (
            _get_with_indices(train_idx, dataset),
            _get_with_indices(val_idx, dataset),
        ),
        indices,
    )


@logger_wraps(level="INFO")
def write_fold_splits_file(
    path: str | Path,
    fold_indices: Iterable[tuple[list[int], list[int]]],
    force: bool = False,
) -> None:
    """
    Write a file containing indices of each fold split

    Parameters
    ----------
    path : str
        The path to write the file to.
    fold_indices : Iterable[tuple[list[int], list[int]]]
        An iterable of tuples containing the training and validation indices for each fold.
    force : bool, optional
        If True, overwrite the file if it exists.
    """
    content = tz.pipe(
        {
            f"fold_{i}": {"train": train, "val": val}
            for i, (train, val) in enumerate(fold_indices)
        },
    )

    if Path(path).exists():
        if not force:
            logger.warning(
                f"File {path} already exists. If you wish to ovewrite, use `force=True`."
            )
            return

    with open(path, "wb") as f:
        pickle.dump(content, f)


def read_fold_splits_file(
    path: str | Path,
    fold: int | None = None,
) -> tuple[list[int] | list[str], list[int] | list[str]] | FoldSplitsDict:
    """
    Read configuration for a fold from the training fold file at `path`.

    If `fold` is not provided, the entire file is read and returned. Else,
    the function returns the train and validation indices for the specified fold.
    """
    with open(path, "rb") as f:
        content = pickle.load(f)
    if fold is None:
        return content
    return content[f"fold_{fold}"]["train"], content[f"fold_{fold}"]["val"]


def train_test_split(
    dataset: Sequence,
    test_split: float,
    return_indices: bool = False,
    seed: int | None = None,
) -> tuple[Sequence, Sequence]:
    """
    Split a dataset into training and test sets.

    Parameters
    ----------
    dataset : Sequence
        The dataset to split.
    test_split : float
        The proportion of the dataset to allocate to the test set.
    return_indices : bool, optional
        If True, return the indices of the dataset split instead of the actual data.
    seed : int, optional
        The random seed to use for the split.

    Returns
    -------
    tuple[Sequence, Sequence]
        A tuple containing the training and test sets.
    """
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    train_indices, test_indices = sk_train_test_split(
        indices, test_size=test_split, random_state=seed
    )
    if not return_indices:
        return (
            _get_with_indices(train_indices, dataset),
            _get_with_indices(test_indices, dataset),
        )
    return train_indices, test_indices


@auto_match_config(prefixes=["training"])
def init_training_dir(
    train_dir: str | Path,
    config_path: str,
    dataset_indices: Sequence[int] | Sequence[str],
    n_folds: int,
    test_split: float,
) -> tuple[Path, Path, Path, list[Path]] | None:
    """
    Initialise directory for training and perform data splitting.

    The function will create a new directory at `train_dir` (if it doesn't exists) and copy the
    configuration file at `config_path` to it. It will then perform a test-train split on the dataset
    indices and split the remaining training indices into folds, saving the indices to
    `train-test-split.pkl` and `validation-fold-splits.pkl` respectively. Lastly, it will create
    checkpoint folders for each fold.

    Parameters
    ----------
    train_dir : str | Path
        The path to the training directory. If it doesn't exist, a new directory will be created.
    config_path : str
        The path to the configuration file to be copied to `train_dir`. If a configuration file
        already exists in `train_dir` and its path is not the same as `config_path`, the function
        will fail.
    dataset_indices : Sequence[int] | Sequence[str]
        The indices from the dataset to be split into test and train sets.
    n_folds : int
        The number of folds to split the training set into after splitting into test and train sets.
    test_split : float
        The proportion of the dataset to allocate to the test set before splitting into folds.

    Returns
    -------
    tuple[Path, Path, list[Path]] | None
        A tuple containing paths to...
         - the copied configuration file
         - test-train split file
         - validation fold splits file
         - the checkpoint folders for each fold
        None is returned if the configuration file already exists in the directory but
        the specified configuration path is different.
    """
    train_dir = Path(train_dir)
    train_dir.mkdir(exist_ok=True)

    # Copy configuration file to train_dir, fail if it already exists
    config_copy_path = train_dir / "configuration.yaml"
    if os.path.exists(config_copy_path) and config_path != config_copy_path:
        logger.error(
            f"configuration.yaml already exists in {train_dir} but specified different configuration path {config_path}. Please remove it or use the same configuration file."
        )
        return
    shutil.copy(config_path, config_copy_path)

    # Perform test-train split if not already done
    if not os.path.exists(train_test_path := train_dir / "train-test-split.pkl"):
        with open(train_test_path, "wb") as f:
            train_indices, test_indices = train_test_split(dataset_indices, test_split)
            pickle.dump((train_indices, test_indices), f)
    else:
        with open(train_test_path, "rb") as f:
            train_indices, test_indices = pickle.load(f)

    # Perform k-fold split if not already done
    if not os.path.exists(data_split_path := train_dir / "validation-fold-splits.pkl"):
        fold_indices = split_into_folds(train_indices, n_folds)
        write_fold_splits_file(data_split_path, fold_indices)  # type: ignore

    # Create checkpoint folders for each fold
    fold_dirs = [train_dir / f"fold_{i}" for i in range(n_folds)]
    for fold in fold_dirs:
        fold.mkdir(exist_ok=True)
    return config_copy_path, train_test_path, data_split_path, fold_dirs


def load_model(checkpoint_path: Path | str) -> LitModel:
    """
    Load `LitModel` from a checkpoint path.
    """
    base_model = torch.load(Path(checkpoint_path).parent / "torch-module.pt")
    return LitModel.load_from_checkpoint(
        checkpoint_path=checkpoint_path, model=base_model
    )


def load_models(
    checkpoint_dir: str | Path, checkpoint_regex: str = "last.ckpt"
) -> dict[str, LitModel]:
    """
    Load multiple `LitModel` from a directory containing multiple checkpoints.

    Parameters
    ----------
    checkpoint_dir : str | Path
        The path to the directory containing the model checkpoints. The checkpoints
        can be in subdirectories.
    checkpoint_regex : str, optional
        The regex pattern to match the checkpoint filenames. Defaults to "last.ckpt".

    Returns
    -------
    dict[str, LitModel]
        A dictionary containing the folder name (not full path) and the
        corresponding `LitModel` loaded from the checkpoint in the folder.
    """
    return tz.pipe(
        checkpoint_dir,
        list_files,
        curried.filter(
            lambda fname: re.match(checkpoint_regex, os.path.basename(fname))
        ),
        lambda ckpt_file_paths: {
            os.path.basename(os.path.dirname(f)): load_model(f) for f in ckpt_file_paths
        },
    )  # type: ignore


def load_training_dir(
    train_dir: str | Path,
    checkpoint_regex: str = "last.ckpt",
) -> tuple[dict, FoldSplitsDict, tuple[list[int], list[int]], TrainDirDict]:
    """
    Load the configuration, data splits, train and test indices, and model checkpoints from training directory.

    The training directory is created by `init_training_dir` and contains:
    - `configuration.yaml`: the configuration file used for training
    - `train-test-split.pkl`: the indices for the training and test sets, produced
        using `uncertainty.trianing.train_test_split()`
    - `validation-fold-splits.pkl`: the indices for the training and validation sets for each fold,
        produced using `uncertainty.training.split_into_folds()`
    - `fold_<int>`: directories for each validation fold, each containing directories with model checkpoints.

    Parameters
    ----------
    train_dir : str | Path
        The path to the training directory.
    checkpoint_regex : str, optional
        The regex pattern to match and identify which checkpoint file to load
        from the checkpoint directories. Defaults to "last.ckpt".

    Returns
    -------
    tuple[dict, FoldSplitsDict, tuple[list[int], list[int]], TrainDirDict]
        A tuple containing:
        - the configuration dictionary
        - the fold splits dictionary containing train and validation indices
          for each fold of the form `{'fold_0': {'train': [...], 'val': [...]}, ...}`
        - the training and test set indices, where the traning indices are used to
          form the folds in the fold splits dictionary (and hence may not be needed)
        - a dictionary containing fold names as the keys and a dictionary of
          `(model_names, LitModel)` pairs as the values. The dictionary is in the format
            `{'fold_0': {'model1': LitModel, 'model2': LitModel, ...}, 'fold_1': {...}, ...}`
    """

    train_dir = Path(train_dir)
    config = configuration(train_dir / "configuration.yaml")
    data_split = read_fold_splits_file(train_dir / "validation-fold-splits.pkl")
    assert isinstance(data_split, dict)

    with open(train_dir / "train-test-split.pkl", "rb") as f:
        train_indices, test_indices = pickle.load(f)

    checkpoints = tz.pipe(
        train_dir,
        os.listdir,
        curried.filter(
            lambda fname: re.match(r"fold_\d+", fname)
            and os.path.isdir(train_dir / fname)
        ),
        curried.sorted(key=lambda x: int(x.split("_")[1])),  # sort by fold number
        lambda fold_dirs: {
            fold: load_models(train_dir / fold, checkpoint_regex) for fold in fold_dirs
        },
    )
    return config, data_split, (train_indices, test_indices), checkpoints  # type: ignore
