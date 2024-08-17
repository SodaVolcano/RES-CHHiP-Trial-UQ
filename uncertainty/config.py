from typing import Callable, Final, TypedDict
from pyparsing import C
from torch import nn
from torch import optim
import torch

Configuration = TypedDict(
    "Configuration",
    {
        "data_dir": str,
        "input_height": int,
        "input_width": int,
        "input_depth": int,
        "input_channel": int,
        "output_channel": int,
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
        "final_layer_activation": Callable[..., nn.Module] | None,
        "model_checkpoint_path": str,
        "n_epochs": int,
        "batch_size": int,
        "metrics": list[str],
        "initializer": Callable[..., torch.Tensor],
        "optimizer": Callable[..., nn.Module],
        "loss": Callable[..., nn.Module],
        "lr_scheduler": type[optim.lr_scheduler.LRScheduler],
        "lr_schedule_percentages": list[float],
        "lr_schedule_values": list[float],
    },
)


def data_config(n_levels: int) -> dict[str, int | str]:
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
        "data_dir": "",
        # Data are formatted as (height, width, depth, dimension)
        "input_height": (2**n_levels) * 12,
        "input_width": (2**n_levels) * 15,
        "input_depth": (2**n_levels) * 8,
        "input_channel": 1,  # Volume
        "output_channel": 3,  # Mask; number of organs, mask only
    }


def unet_config(n_levels: int) -> dict[str, int | float | str | type[nn.Module]]:
    """
    Preset configuration for U-Net model
    """

    return {
        # ------- ConvolutionBlock settings  --------
        "kernel_size": 3,
        "n_convolutions_per_block": 2,
        "activation": nn.GELU,
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
        # Use sigmoid if using binary crossentropy, softmax if using categorical crossentropy
        # Set to None to disable
        "final_layer_activation": nn.Sigmoid,
    }


def training_config() -> dict[str, int | str | list[int | float | str] | type]:
    """
    Preset configuration for training
    """
    return {
        "model_checkpoint_path": "./checkpoints/model.pth",
        "n_epochs": 20,
        "n_batches_per_epoch": 100,
        "batch_size": 64,
        "metrics": ["accuracy"],
        "initializer": nn.init.xavier_normal_,  # type: ignore
        "optimizer": optim.Adam,  # type: ignore
        "loss": nn.BCELoss,
        # Learning rate scheduler, decrease learning rate at certain epochs
        "lr_scheduler": optim.lr_scheduler.StepLR,
        # Percentage of training where learning rate is decreased
        "lr_schedule_percentages": [0.2, 0.5, 0.8],
        # Gradually decrease learning rate, starting at first value
        "lr_schedule_values": [3e-4, 1e-4, 1e-5, 1e-6],
    }


def configuration() -> Configuration:
    """
    Preset configuration for U-Net model
    """
    n_levels: Final[int] = 5  # WARNING: used to calculate input shape

    return data_config(n_levels) | unet_config(n_levels) | training_config()  # type: ignore
