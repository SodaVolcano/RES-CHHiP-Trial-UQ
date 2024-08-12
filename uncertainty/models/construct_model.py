from typing import Callable

from ..config import configuration, Configuration
from ..utils.logging import logger_wraps

import keras
import tensorflow as tf
from toolz import curry


@logger_wraps(level="INFO")
@curry
def construct_model(
    model_fn: Callable[[tf.Tensor, Configuration], tf.Tensor],
    batches_per_epoch: int,
    config=configuration(),
    show_model_info: bool = True,
):
    """
    Initialise model, set learning schedule and loss function and visualise model

    Parameters
    ----------
    model_fn : Callable[[tf.Tensor, Configuration], tf.Tensor]
        Function that passes an input tensor through a model and returns the output tensor
    batches_per_epoch : int
        Number of batches per epoch
    config : Configuration (optional)
        Configuration object
    show_model_info : bool
        Whether to print model summary and output a diagram of the model. Default is True
    """
    input_ = keras.layers.Input(
        shape=(
            config["input_height"],
            config["input_width"],
            config["input_depth"],
            config["input_channel"],
        ),
        batch_size=config["batch_size"],
    )
    output, name = model_fn(input_, config)  # type: ignore
    model = keras.Model(input_, output, name=name)

    # Model outputs are logits, loss must apply softmax before computing loss
    loss = config["loss"](from_logits=config["final_layer_activation"] is None)  # type: ignore

    # Number of iterations (batches) to pass before decreasing learning rate
    boundaries = [
        int(config["n_epochs"] * percentage * batches_per_epoch)
        for percentage in config["lr_schedule_percentages"]
    ]
    lr_schedule = config["lr_scheduler"](boundaries, config["lr_schedule_values"])  # type: ignore

    model.compile(
        loss=loss,
        optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),  # type: ignore
        metrics=config["metrics"],
    )

    if show_model_info:
        keras.utils.plot_model(model, to_file="unet.png")
        model.summary()

    return model
