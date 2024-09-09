from typing import Callable, Final, TypedDict

import torch
from torch import nn, optim
from deepspeed.ops.adam import DeepSpeedCPUAdam

Configuration = TypedDict(
    "Configuration",
    {
        "data_dir": str,
        "staging_dir": str,
        "staging_fname": str,
        "train_fname": str,
        "test_fname": str,
        "input_height": int,
        "input_width": int,
        "input_depth": int,
        "input_channel": int,
        "patch_size": tuple[int, int, int],
        "patch_step": int,
        "foreground_oversample_ratio": float,
        "intensity_range": tuple[int, int],
        "output_channel": int,
        "val_split": float,
        "test_split": float,
        "kernel_size": int,
        "n_convolutions_per_block": int,
        "use_instance_norm": bool,
        "instance_norm_decay": float,
        "instance_norm_epsilon": float,
        "activation": Callable[..., nn.Module],
        "dropout_rate": float,
        "n_kernels_init": int,
        "n_kernels_max": int,
        "n_levels": int,
        "n_kernels_last": int,
        "final_layer_activation": Callable[..., nn.Module],
        "classification_threshold": float,
        "deep_supervision": bool,
        "model_checkpoint_path": str,
        "n_epochs": int,
        "n_batches_per_epoch": int,
        "n_batches_val": int,
        "batch_size": int,
        "batch_size_eval": int,
        "initialiser": Callable[..., torch.Tensor],
        "optimiser": Callable[..., nn.Module],
        "optimiser_kwargs": dict[str, int | float | str],
        "lr_scheduler": type[optim.lr_scheduler.LRScheduler],
        "log_sink": str,
        "log_format": str,
        "log_level": str,
        "log_retention": str,
    },
)


def data_config() -> dict[str, int | str | float | tuple[int, ...]]:
    """
    Preset configuration for data
    """
    return {
        # Directory containing folders of DICOM slices
        "data_dir": "./",
        # Directory to store h5 files (processed data)
        "staging_dir": "./",
        "staging_fname": "dataset.h5",  # for the whole dataset
        "train_fname": "train_preprocessed.h5",
        "test_fname": "test.h5",
        # Data are formatted as (height, width, depth, dimension)
        # For volume
        "input_channel": 1,
        # if patch size is bigger than image, image is padded
        # try to keep this divisible by (2 ** (n_level - 1))
        "patch_size": (256, 256, 64),
        # For sliding window patch samplers (used for validation and test set)
        "patch_step": 32,
        # % of sampled patches guaranteed to contain foreground
        "foreground_oversample_ratio": 1 / 3,
        "intensity_range": (0, 1),
        # Number of organs, mask only
        "output_channel": 3,
        "test_split": 0.2,  # percentage of total dataset for testing
        "val_split": 0.2,  # percentage of training data (after test split) to use for validation
    }


def unet_config() -> dict[str, int | float | str | type[nn.Module]]:
    """
    Preset configuration for U-Net model
    """

    return {
        # ------- ConvolutionBlock settings  --------
        "kernel_size": 3,
        "n_convolutions_per_block": 2,
        "activation": nn.LeakyReLU,
        "dropout_rate": 0.5,
        "use_instance_norm": True,  # batch norm don't work well with small batch size
        # AKA momentum, how much of new mean/variance are added to the running mean/variance
        "instance_norm_decay": 0.9,
        # Small value to avoid division by zero
        "instance_norm_epsilon": 1e-5,
        # ------- Encoder/Decoder settings -------
        # Number of kernels in the output of the first level of Encoder
        # doubles/halves at each level in Encoder/Decoder
        "n_kernels_init": 32,
        "n_kernels_max": 512,  # maximum allowed number of kernels for a level
        # Number of resolutions/blocks; height of U-Net
        "n_levels": 6,
        # Number of class to predict
        "n_kernels_last": 3,
        "final_layer_activation": nn.Sigmoid,
        "classification_threshold": 0.5,  # > threshold will be counted as 1
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
        "model_checkpoint_path": "./checkpoints/unet_2",
        # from nnU-Net settings
        "deep_supervision": True,
        "n_epochs": 750,
        "n_batches_per_epoch": 250,
        "n_batches_val": 50,  # Number of batches when using random sampler for validation set
        "batch_size": 2,
        "batch_size_eval": 4,  # batch size for both validation and test
        "initialiser": nn.init.kaiming_normal_,  # type: ignore
        # "optimiser": DeepSpeedCPUAdam,  # type: ignore
        # "optimiser_kwargs": {},
        "optimiser": optim.SGD,  # type: ignore
        "optimiser_kwargs": {"momentum": 0.99, "nesterov": True},
        # Learning rate scheduler, decrease learning rate at certain epochs
        # WARNING: interval is ignored in ensemble training because of manual optimisation
        "lr_scheduler": lambda optimiser: optim.lr_scheduler.PolynomialLR(
            optimiser, total_iters=750, power=0.9
        ),
    }


def configuration() -> Configuration:
    """
    Preset configuration for U-Net model
    """
    return data_config() | unet_config() | training_config() | logger_config()  # type: ignore
