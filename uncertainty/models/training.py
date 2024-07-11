"""
Utility functions for model training, from model construction, data loading, and training loop
"""

import numpy as np
from ..common.utils import curry
from ..common.constants import model_config

from typing import Callable, Iterable

from tensorflow.python.keras import Model
import tensorflow.keras as keras
import toolz as tz
import toolz.curried as curried


@curry
def construct_model(
    model_constructor: Callable[[dict], Model],
    batches_per_epoch: int,
    config: dict = model_config(),
    show_model_info: bool = True,
):
    """
    Initialise model, set learning schedule and loss function and plot model
    """
    model = model_constructor(config)

    # Model outputs are logits, loss must apply softmax before computing loss
    loss = config["loss"](from_logits=config["final_layer_activation"] is None)

    # Number of iterations (batches) to pass before decreasing learning rate
    boundaries = [
        int(config["n_epochs"] * percentage * batches_per_epoch)
        for percentage in config["lr_schedule_percentages"]
    ]
    lr_schedule = config["lr_scheduler"](boundaries, config["lr_schedule_values"])

    model.compile(
        loss=loss,
        optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
        metrics=config["metrics"],
    )

    if show_model_info:
        keras.utils.plot_model(model, to_file="unet.png")
        model.summary()

    return model


def preprocess_data(
    dataset: Iterable[tuple[np.ndarray, np.ndarray]], config: dict = model_config()
):
    """
    Preprocess data for training
    """
    return map()
    # standardise ROIs?
    # make sure label one-hot-encoding are in right order
    # resize images to input dimensions
    # ensure mask dimension match image dimensions
    # reshape to (height, width, channels)
    # clip first to HU range, then
    # normalise pixel values to [0, 1]
    # cast to float32
    pass
