from torch import nn
from .datasets import H5Dataset
from .lightning import LitSegmentation, SegmentationData
from ..models.unet import UNet
from ..config import Configuration
from typing import Any, Literal
import torch
import dill
import os


def load_checkpoint(
    ckpt_dir: None | str = None,
    model_path: str | None = None,
    config_path: str | None = None,
    indices_path: str | None = None,
) -> tuple[
    nn.Module,
    Configuration,
    dict[Literal["train_indices", "val_indices"], list[int]],
    H5Dataset,
    H5Dataset,
]:
    """
    Load the model, configuration, and its dataset from a checkpoint folder.

    Parameters
    ----------
    ckpt_dir : str, optional
        Folder containing "last.ckpt", "config.pkl", and "indices.pt" for
        the model weights, configuration object, and train-validation split
        indices respectively. If they are not named as those files, pass in
        `model_path`, `config_path`, and `indices_path` instead of this parameter.
    model_path : str, optional
        Path to the model checkpoint file. If not provided, defaults to
        "last.ckpt" in the `ckpt_dir`.
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
    model_path = model_path or os.path.join(ckpt_dir, "last.ckpt")
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
