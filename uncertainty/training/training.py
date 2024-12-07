"""
Functions for managing training, such as checkpointing and data splitting.
"""

import os
import pickle
import random
import shutil
from pathlib import Path
from random import randint
from typing import Callable, Iterable, Sequence, TypedDict

import lightning
import toolz as tz
from lightning import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from loguru import logger
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split as sk_train_test_split
from toolz import curried

from ..config import auto_match_config
from ..models import get_model
from ..utils import logger_wraps, unpack_args, unpacked_map
from .datasets import SegmentationData

DataSplitDict = TypedDict(
    "DataSplitDict", {"train": list[int], "val": list[int], "seed": int}
)


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
    precision: str,
    save_last_checkpoint: bool,
    strategy: str = "ddp",
    accelerator: str = "auto",
    enable_progress_bar: bool = True,
    enable_model_summary: bool = True,
):
    """
    Train a model and save checkpoints in `<checkpoint_path>/<experiment_name>/<model_classname_lowercase>[-<int>]`.

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
        Device type to run the model on. Options include `"auto"`, `"gpu"`, `"cpu"`, or `"tpu"`. Defaults to `"auto"` if available.
    precision : str, optional
        Training precision of float values.
    enable_progress_bar : bool, optional
        Whether to show a progress bar during training.
    enable_model_summary : bool, optional
        Whether to print a model summary before training starts.
    """
    run_validation = check_val_every_n_epoch > 0

    tb_logger = TensorBoardLogger(save_dir=log_dir, name=experiment_name)

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
        num_sanity_val_steps=0 if not run_validation else 2,
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
    seed: int | None = None,
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
    seed : int, optional
        Random seed to use for reproducibility. Used at the beginning before training all models.
    **kwargs
        Additional keyword arguments used to initialise the models and configure the training process.
    """
    checkpoint_dir = Path(checkpoint_dir)
    if seed is not None:
        seed_everything(seed)

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
        unpacked_map(
            lambda model_fn, model_path: (
                train_model(
                    model_fn(**kwargs),
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
        return unpacked_map(lambda train, val: (train.tolist(), val.tolist()), indices)
    return unpacked_map(
        lambda train_idx, val_idx: (
            _get_with_indices(train_idx, dataset),
            _get_with_indices(val_idx, dataset),
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
) -> (
    tuple[list[int] | list[str], list[int] | list[str], int | None]
    | dict[str, DataSplitDict]
):
    """
    Read configuration for a fold from the training fold file at `path`.

    If `fold` is not provided, the entire file is read and returned. Else,
    the function returns the train, validation indices and the seed value
    for the specified fold (if the seed exists).
    """
    with open(path, "rb") as f:
        content = pickle.load(f)
    if fold is not None:
        return content[f"fold_{fold}"]
    return content["train"], content["val"], content.get("seed", None)


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
) -> tuple[Path, Path, list[Path]] | None:
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
    if not os.path.exists(split_path := train_dir / "train-test-split.pkl"):
        with open(split_path, "wb") as f:
            pickle.dump(train_test_split(dataset_indices, test_split), f)

    # Perform k-fold split if not already done
    if not os.path.exists(data_split_path := train_dir / "validation-fold-splits.pkl"):
        fold_indices = split_into_folds(dataset_indices, n_folds)
        write_training_fold_file(data_split_path, fold_indices)  # type: ignore

    # Create checkpoint folders for each fold
    fold_dirs = [train_dir / f"fold_{i}" for i in range(n_folds)]
    for fold in fold_dirs:
        fold.mkdir(exist_ok=True)
    return config_copy_path, data_split_path, fold_dirs


# def load_checkpoint(
#     ckpt_dir: None | str = None,
#     model_path: str | None = None,
#     config_path: str | None = None,
#     indices_path: str | None = None,
# ) -> tuple[
#     nn.Module,
#     dict,
#     dict[Literal["train_indices", "val_indices"], list[int]],
#     H5Dataset,
#     H5Dataset,
# ]:
#     """
#     Load the model, configuration, and its dataset from a checkpoint folder.

#     Parameters
#     ----------
#     ckpt_dir : str, optional
#         Folder containing "latest.ckpt", "config.pkl", and "indices.pt" for
#         the model weights, configuration object, and train-validation split
#         indices respectively. If they are not named as those files, pass in
#         `model_path`, `config_path`, and `indices_path` instead of this parameter.
#     model_path : str, optional
#         Path to the model checkpoint file. If not provided, defaults to
#         "latest.ckpt" in the `ckpt_dir`.
#     config_path : str, optional
#         Path to the configuration file. If not provided, defaults to
#         "config.pkl" in the `ckpt_dir`.
#     indices_path : str, optional
#         Path to the indices file. If not provided, defaults to
#         "indices.pt" in the `ckpt_dir`.

#     Returns
#     -------
#     tuple
#         A tuple containing:
#         - ckpt : The loaded model checkpoint.
#         - config : The configuration object.
#         - indices: A dictionary with keys "train_indices" and "val_indices" with a list
#                    of integers indicating indices in the total dataset allocated to the
#                    train and validation set respectively.
#         - train_dataset: H5Dataset containing the training data.
#         - val_dataset: H5Dataset containing the validation data.

#     Raises
#     ------
#     FileNotFoundError
#         If any of the specified paths do not exist.
#     """
#     config_path = config_path or os.path.join(ckpt_dir, "config.pkl")
#     indices_path = indices_path or os.path.join(ckpt_dir, "indices.pt")
#     model_path = model_path or os.path.join(ckpt_dir, "latest.ckpt")
#     with open(config_path, "rb") as f:
#         config = dill.load(f)
#     indices = torch.load(indices_path, weights_only=True)
#     train_dataset = H5Dataset(
#         os.path.join(config["staging_dir"], config["train_fname"]),
#         indices=indices["train_indices"],
#     )
#     val_dataset = H5Dataset(
#         os.path.join(config["staging_dir"], config["train_fname"]),
#         indices=indices["val_indices"],
#     )
#     model = UNet(config)
#     model = LitSegmentation.load_from_checkpoint(
#         model_path, model=model, config=config, save_hyperparams=False
#     ).model
#     return model, config, indices, train_dataset, val_dataset


# def checkpoint_dir_type(
#     path,
#     required_files: list[str] | set[str] = ["latest.ckpt", "indices.pt", "config.pkl"],
# ) -> Literal["single", "multiple", "invalid"]:
#     """
#     Return type of the checkpoint directory - "single", "multiple", or "invalid"

#     This function returns if the given directory contains a single model's checkpoint or
#     folders of model checkpoints. "single" is defined as a folder with at least the
#     required files while "multiple" is defined as a directory without the required files,
#     but have directories that **all** have the required files.

#     Parameters
#     ----------
#     path : str
#         The path to the checkpoint directory to be validated.

#     Returns
#     -------
#     Literal["single", "multiple", "invalid"]
#         'single' if all required files are found in the main directory,
#         'multiple' if required files are found in subdirectories,
#         and "invalid" if otherwise.
#     """
#     required_files = set(required_files)

#     if not os.path.isdir(path):
#         logger.critical(f"The path {path} is not a valid directory.")
#         return "invalid"

#     if required_files.issubset(os.listdir(path)):
#         return "single"

#     subdir_paths = [os.path.join(path, entry) for entry in os.listdir(path)]
#     bad_dirs = list(
#         filter(
#             lambda x: not (os.path.isdir(x) and required_files.issubset(os.listdir(x))),
#             subdir_paths,
#         )
#     )
#     if not len(bad_dirs) == 0:
#         logger.critical(
#             f"Folders of checkpoint detected but the following folder have bad structure: {bad_dirs}"
#         )
#         return "invalid"

#     return "multiple"
