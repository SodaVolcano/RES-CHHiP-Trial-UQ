from uncertainty.config import Configuration, configuration

from typing import Callable

import tensorflow as tf
import toolz as tz


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
    return [model_fn(input_, config)[0] for _ in range(ensemble_size)], "DeepEnsemble"
