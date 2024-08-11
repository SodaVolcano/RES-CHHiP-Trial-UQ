"""
Type definitions for configuration dictionary.
"""

from typing import TypedDict
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
