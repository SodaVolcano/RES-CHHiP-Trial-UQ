"""
Utility functions for model training, from model construction, data loading, and training loop
"""

from operator import methodcaller
from tests.context import PatientScan
from uncertainty.data.preprocessing import filter_roi, find_organ_roi, map_interval
from uncertainty.utils.parallel import pmap
from ..utils.logging import logger_wraps
from ..utils.wrappers import curry
from ..common.constants import model_config, HU_RANGE, ORGAN_MATCHES

from typing import Callable, Iterable

import numpy as np
from tensorflow.python.keras import Model
import tensorflow.keras as keras
import toolz as tz
import toolz.curried as curried
from scipy.ndimage import zoom
from fn import _


@logger_wraps(level="INFO")
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


@logger_wraps(level="INFO")
@curry
def preprocess_data(
    dataset: Iterable[tuple[np.ndarray, np.ndarray]], config: dict = model_config()
):
    """
    Preprocess data for training
    """

    def volume_pipeline(scan: PatientScan):
        return tz.pipe(
            scan.volume,
            lambda v: zoom(
                v,
                tuple(
                    to / from_
                    for to, from_ in zip(
                        (
                            config["input_height"],
                            config["input_width"],
                            config["input_dim"],
                        ),
                        v.shape,
                    )
                ),
            ),
            lambda v: np.clip(v, *HU_RANGE),
            map_interval(HU_RANGE, (0, 1)),
            _.astype(np.float32),
        )

    def mask_pipeline(scan: PatientScan):
        names = tz.pipe(
            scan.masks[""].get_organ_names(),
            filter_roi,
            lambda mask_names: [
                find_organ_roi(organ, mask_names) for organ in ORGAN_MATCHES
            ],
            curried.filter(_ is not None),
            list,
        )

        if len(names) != len(ORGAN_MATCHES):
            return None

        return tz.pipe(
            names,
            getattr(scan.masks[""], "as_array"),
            lambda m: zoom(
                m,
                tuple(
                    (
                        to / from_
                        for to, from_ in zip(
                            (
                                config["input_height"],
                                config["input_width"],
                                config["input_dim"],
                                m.shape[3],
                            ),
                            m.shape,
                        )
                    )
                ),
            ),
            _.astype(np.float32),
        )

    return pmap(curried.juxt(volume_pipeline, mask_pipeline), dataset)
