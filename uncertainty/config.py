from typing import Final, TypedDict
import keras


Configuration = TypedDict(
    "Configuration",
    {
        "data_dir": str,
        "input_height": int,
        "input_width": int,
        "input_depth": int,
        "input_channel": int,
        "kernel_size": tuple[int, int, int],
        "n_convolutions_per_block": int,
        "use_batch_norm": bool,
        "batch_norm_decay": float,
        "batch_norm_epsilon": float,
        "activation": str,
        "dropout_rate": float,
        "n_kernels_init": int,
        "n_levels": int,
        "n_kernels_last": int,
        "final_layer_activation": str,
        "n_kernels_per_block": list[int],
        "model_checkpoint_path": str,
        "n_epochs": int,
        "batch_size": int,
        "metrics": list[str],
        "initializer": str,
        "optimizer": type[keras.optimizers.Optimizer],
        "loss": type[keras.losses.Loss],
        "lr_scheduler": type[keras.optimizers.schedules.LearningRateSchedule],
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
        "input_height": (2**n_levels) * 10,
        "input_width": (2**n_levels) * 13,
        "input_depth": (2**n_levels) * 6,
        "input_channel": 3,  # Number of organs, mask only
    }


def unet_config(n_levels: int) -> dict[str, int | str | list[int]]:
    """
    Preset configuration for U-Net model
    """

    config = {
        # ------- ConvolutionBlock settings  --------
        "kernel_size": (3, 3, 3),
        "n_convolutions_per_block": 1,
        "activation": "gelu",
        "dropout_rate": 0.5,
        "use_batch_norm": False,
        "batch_norm_decay": 0.9,
        "batch_norm_epsilon": 1e-5,
        # ------- Encoder/Decoder settings -------
        # Number of kernels in first level of Encoder, doubles/halves at each level in Encoder/Decoder
        "n_kernels_init": 64,
        # Number of resolutions/blocks; height of U-Net
        "n_levels": n_levels,
        # Number of class to predict
        "n_kernels_last": 1,
        # Use sigmoid if using binary crossentropy, softmax if using categorical crossentropy
        # Note if using softmax, n_kernels_last should be equal to number of classes (e.g. 2 for binary segmentation)
        # If None, loss will use from_logits=True
        "final_layer_activation": "sigmoid",
        "n_kernels_per_block": [],  # Calculated in the next line
    }
    # Explicitly calculate number of kernels per block (for Encoder)
    config["n_kernels_per_block"] = [
        config["n_kernels_init"] * (2**block_level)
        for block_level in range(config["n_levels"])
    ]
    return config


def training_config() -> dict[str, int | str | list[int | float | str] | type]:
    """
    Preset configuration for training
    """
    return {
        "model_checkpoint_path": "./checkpoints/checkpoint.model.keras",
        "n_epochs": 1,
        "batch_size": 32,
        "metrics": ["accuracy"],
        "initializer": "he_normal",  # For kernel initialisation
        "optimizer": keras.optimizers.Adam,
        "loss": keras.losses.BinaryCrossentropy,
        # Learning rate scheduler, decrease learning rate at certain epochs
        "lr_scheduler": keras.optimizers.schedules.PiecewiseConstantDecay,
        # Percentage of training where learning rate is decreased
        "lr_schedule_percentages": [0.2, 0.5, 0.8],
        # Gradually decrease learning rate, starting at first value
        "lr_schedule_values": [3e-4, 1e-4, 1e-5, 1e-6],
    }


def configuration() -> Configuration:
    """
    Preset configuration for U-Net model
    """
    n_levels: Final[int] = 5  # used to calculate input shape

    return data_config(n_levels) | unet_config(n_levels) | training_config()  # type: ignore
