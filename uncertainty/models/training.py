"""
Utility functions for model training, from model construction, data loading, and training loop
"""

from typing import Callable
from tensorflow.python.keras import Model
import tensorflow.python.keras as keras

from uncertainty.common.constants import (
    model_config,
)  # TODO: replace with tensorflow.keras


def construct_model(
    model_constructor: Callable[[dict], Model],
    batches_per_epoch: int,
    config: dict = model_config(),
):
    """ """
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

    model._name = model.__class__.__name__
    return model
