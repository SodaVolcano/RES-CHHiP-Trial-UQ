from typing import Callable, Final, TypedDict

import torch
from pyparsing import C
from torch import nn, optim

Configuration = TypedDict(
    "Configuration",
    {
        "data_dir": str,
        "staging_dir": str,
        "staging_fname": str,
        "input_height": int,
        "input_width": int,
        "input_depth": int,
        "input_channel": int,
        "intensity_range": tuple[int, int],
        "output_channel": int,
        "val_split": float,
        "kernel_size": int,
        "n_convolutions_per_block": int,
        "use_batch_norm": bool,
        "batch_norm_decay": float,
        "batch_norm_epsilon": float,
        "activation": Callable[..., nn.Module],
        "dropout_rate": float,
        "n_kernels_init": int,
        "n_levels": int,
        "n_kernels_last": int,
        "final_layer_activation": Callable[..., nn.Module],
        "model_checkpoint_path": str,
        "n_epochs": int,
        "batch_size": int,
        "metrics": list[str],
        "initialiser": Callable[..., torch.Tensor],
        "optimiser": Callable[..., nn.Module],
        "optimiser_kwargs": dict[str, int | float | str],
        "loss": Callable[..., nn.Module],
        "lr_scheduler": type[optim.lr_scheduler.LRScheduler],
        "log_sink": str,
        "log_format": str,
        "log_level": str,
        "log_retention": str,
    },
)


def data_config(n_levels: int) -> dict[str, int | str | float | tuple[int, int]]:
    """
    Preset configuration for data

    Parameters
    ----------
    n_levels : int
        Number of levels in the U-Net, used to calculate input shape
        to ensure it's divisible by 2 * (n_levels - 1). The
        dimensions are divided that many times in the U-Net so setting
        it to a divisible number ensure no row/col is discarded.
    """
    return {
        # Directory containing folders of DICOM slices
        "data_dir": "/run/media/tin/Expansion/honours/dataset/originals/CHHiP/",
        # Directory to store h5 files (processed data)
        "staging_dir": "/run/media/tin/Expansion/honours/dataset/originals/CHHiP_patientScans/",
        "staging_fname": "dataset.h5",
        # Data are formatted as (height, width, depth, dimension)
        "input_height": (2**n_levels) * 12,
        "input_width": (2**n_levels) * 15,
        "input_depth": (2**n_levels) * 8,
        # For volume
        "input_channel": 1,
        "intensity_range": (0, 255),
        # Number of organs, mask only
        "output_channel": 3,
        "val_split": 0.2,  # percentage of data to use for validation
    }


def unet_config(n_levels: int) -> dict[str, int | float | str | type[nn.Module]]:
    """
    Preset configuration for U-Net model
    """

    return {
        # ------- ConvolutionBlock settings  --------
        "kernel_size": 3,
        "n_convolutions_per_block": 2,
        "activation": nn.LeakyReLU,
        "dropout_rate": 0.5,
        "use_batch_norm": True,
        # AKA momentum, how much of new batch's mean/variance are added to the running mean/variance
        "batch_norm_decay": 0.9,
        # Small value to avoid division by zero
        "batch_norm_epsilon": 1e-5,
        # ------- Encoder/Decoder settings -------
        # Number of kernels in the output of the first level of Encoder
        # doubles/halves at each level in Encoder/Decoder
        "n_kernels_init": 64,
        # Number of resolutions/blocks; height of U-Net
        "n_levels": n_levels,
        # Number of class to predict
        "n_kernels_last": 3,
        "final_layer_activation": nn.Softmax,
    }


def logger_config() -> dict[str, str]:
    """
    Preset configuration for logger
    """
    return {
        "log_sink": "./logs/out_{time}.log",
        "log_format": "{time:YYYY-MM-DD at HH:mm:ss} {level} {message}",
        "log_level": "DEBUG",
        "log_retention": "7 days",
    }


def training_config() -> dict[str, int | str | list[int | float | str] | type]:
    """
    Preset configuration for training
    """
    return {
        "model_checkpoint_path": "./checkpoints",
        # from nnU-Net settings
        "n_epochs": 1000,
        "n_batches_per_epoch": 250,
        "batch_size": 4,
        "metrics": ["accuracy"],
        "initialiser": nn.init.xavier_normal_,  # type: ignore
        "optimiser": optim.SGD,  # type: ignore
        "optimiser_kwargs": {"momentum": 0.99, "nesterov": True},
        # Learning rate scheduler, decrease learning rate at certain epochs
        "lr_scheduler": optim.lr_scheduler.PolynomialLR,
    }


def configuration() -> Configuration:
    """
    Preset configuration for U-Net model
    """
    n_levels: Final[int] = 5  # WARNING: used to calculate input shape

    return data_config(n_levels) | unet_config(n_levels) | training_config() | logger_config()  # type: ignore
