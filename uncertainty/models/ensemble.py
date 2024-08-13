"""
An ensemble of models
"""

from itertools import repeat
from uncertainty.utils.common import unpack_args
from ..utils.wrappers import curry
from ..utils.logging import logger_wraps
from ..config import Configuration, configuration


from typing import Callable

import tensorflow as tf
import toolz as tz
from toolz import curried


@logger_wraps(level="INFO")
@curry
def DeepEnsemble(
    input_: tf.Tensor,
    ensemble_size: int,
    model_fn: Callable[[tf.Tensor, Configuration], tuple[tf.Tensor, str]],
    config: Configuration = configuration(),
) -> tuple[list[tf.Tensor], str]:
    """
    Pass input through an ensemble of models and return the ensemble output and name

    Parameters
    ----------
    ensemble_size : int
        The number of models in the ensemble
    model_fn : Callable[[tf.Tensor, Configuration], tuple[tf.Tensor, str]]
        Function that passes an input tensor through a model and returns the output tensor and model name
    """
    return tz.pipe(
        input_,
        lambda x: repeat(x, ensemble_size),
        curried.map(lambda x: model_fn(x, config)),
        lambda out_names: zip(*out_names),  # separate outputs and names
        tuple,
        unpack_args(
            lambda outputs, names: (outputs, f"{names[0]}DeepEnsemble{ensemble_size}"),
        ),
    )  # type: ignore
