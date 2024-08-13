"""
Wrapper around a model that produces multiple outputs using test-time augmentation (TTA)
"""

from re import T
from uncertainty.utils.common import unpack_args
from ..config import Configuration, configuration
from ..utils.logging import logger_wraps
from ..utils.wrappers import curry

from typing import Callable

import tensorflow as tf
import keras
import numpy as np
from volumentations import Compose
import toolz as tz
from toolz import curried


class Augmentations(keras.layers.Layer):
    """
    Wrapper to handle symbolic tensors
    """

    def __init__(self, augmentations: Compose, n_augmentations: int, **kwargs):
        super(Augmentations, self).__init__(**kwargs)
        self.layer = augmentations
        self.n_augmentations = n_augmentations

    def transform(self, x: tf.Tensor) -> tf.Tensor:
        """
        Apply augmentations to a single input tensor
        """
        return tf.convert_to_tensor(self.layer(**{"image": x.numpy()})["image"])  # type: ignore

    def call(self, inputs, training=True):
        if training or tf.is_symbolic_tensor(inputs):
            return inputs
        return tz.pipe(
            inputs,
            lambda x: tf.repeat(x, self.n_augmentations, axis=0),
            lambda xs: tf.map_fn(
                self.transform,
                xs,
                dtype=xs.dtype,
            ),
        )


@logger_wraps(level="INFO")
@curry
def TTA(
    input_: tf.Tensor,
    model_fn: Callable[[tf.Tensor, Configuration], tuple[tf.Tensor, str]],
    augmentor: Compose,
    n_augmentations: int,
    config: Configuration = configuration(),
) -> tuple[list[tf.Tensor], str]:
    """
    Produce multiple outputs using test-time augmentation (TTA) using the provided model

    Parameters
    ----------
    model_fn : Callable[[tf.Tensor, Configuration], tuple[tf.Tensor, str]]
        Function that passes an input tensor through a model and returns the output tensor and model name
    """

    augmentations = Augmentations(augmentor, n_augmentations)
    single_input_pipe = lambda x: tz.pipe(
        x,
        lambda x: model_fn(x, config),
        unpack_args(lambda output, name: (output, f"{name}-TTA")),
    )

    multi_input_pipe = lambda x: tz.pipe(
        x,
        lambda x: tf.map_fn(lambda xx: model_fn(xx, config), x),
        lambda out_names: zip(*out_names),  # separate outputs and names
        tuple,
        unpack_args(
            lambda outputs, names: (outputs, f"{names[0]}-TTA"),
        ),
    )

    return tz.pipe(
        input_,
        augmentations,
        lambda x: multi_input_pipe(x) if isinstance(x, list) else single_input_pipe(x),
    )  # type: ignore
