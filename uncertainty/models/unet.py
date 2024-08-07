"""
Define a U-Net model using Keras functional API

Reference: https://github.com/christianversloot/machine-learning-articles/blob/main/how-to-build-a-u-net-for-image-segmentation-with-tensorflow-and-keras.md
"""

from typing import Sequence
import tensorflow as tf
from uncertainty.data.preprocessing import crop_nd
from ..utils.logging import logger_wraps
from ..common.constants import model_config
from ..utils.wrappers import curry
from ..utils.sequence import growby_accum
from ..utils.common import unpack_args

import toolz as tz
from toolz import curried
from tensorflow.keras import layers, Model
from tensorflow import keras


class CentreCrop3D(keras.layers.Layer):
    """
    Crop the input tensor to the new shape
    """

    def __init__(self, new_shape: tf.TensorShape) -> None:
        """new_shape is the shape of a single instance"""
        super(CentreCrop3D, self).__init__()
        self.out_shape = new_shape

    def call(self, inputs):
        new_shape = (inputs.shape[0],) + self.out_shape
        return crop_nd(inputs, new_shape)


@logger_wraps(level="INFO")
@curry
def ConvLayer(
    x: tf.Tensor,
    n_kernels: int,
    kernel_size: tuple[int, ...],
    initializer: str,
    use_batch_norm: bool,
    activation: str,
):
    """Convolution followed by (optionally) BN and activation"""
    return tz.pipe(
        x,
        layers.Conv3D(
            n_kernels,
            kernel_size,
            kernel_initializer=initializer,
            padding="same",  # Keep dimensions the same
            data_format="channels_last",
        ),
        layers.BatchNormalization() if use_batch_norm else tz.identity,
        layers.Activation(activation),
    )


@logger_wraps(level="INFO")
@curry
def ConvBlock(x: tf.Tensor, n_kernels: int, config: dict = model_config()):
    """Pass input through n convolution layers"""
    return tz.pipe(
        x,
        *[
            ConvLayer(
                n_kernels=n_kernels,
                kernel_size=config["kernel_size"],
                initializer=config["initializer"],
                use_batch_norm=config["use_batch_norm"],
                activation=config["activation"],
            )
            for _ in range(config["n_convolutions_per_block"])
        ]
    )


@logger_wraps(level="INFO")
@curry
def Encoder(x: tf.Tensor, config: dict = model_config()):
    """
    Pass input through encoder and return (output, skip_connections)
    """
    levels = [
        lambda x: tz.pipe(
            x,
            layers.MaxPool3D((2, 2, 2), strides=(2, 2, 2), data_format="channels_last"),
            ConvBlock(n_kernels=config["n_kernels_per_block"][level], config=config),
        )
        for level in range(1, config["n_levels"])  # Exclude first block
    ]

    return tz.pipe(
        x,
        ConvBlock(n_kernels=config["n_kernels_init"], config=config),
        # Repeatedly apply downsample and ConvBlock to input to get skip connection list
        growby_accum(fs=levels),
        list,
        # last element is from bottleneck so exclude from skip connections
        lambda skip_inputs: (skip_inputs[-1], skip_inputs[:-1]),
    )


@logger_wraps(level="INFO")
@curry
def DecoderLevel(
    x: tf.Tensor, skip: tf.Tensor, n_kernels: int, config: dict = model_config()
):
    """
    One level of decoder path: upsample, crop, concat with skip, and convolve the input
    """

    return tz.pipe(
        x,
        layers.Conv3DTranspose(
            n_kernels // 2,  # Halve dimensions because we are concatenating
            config["kernel_size"],
            strides=(2, 2, 2),
            kernel_initializer=config["initializer"],
            data_format="channels_last",
        ),
        # Crop skip to same size as x, x's last dim is half of skip's last dim
        CentreCrop3D(skip.shape[1:-1] + (skip.shape[-1] // 2,)),  # type: ignore
        lambda cropped_x: layers.Concatenate(axis=-1)([skip, cropped_x]),
        ConvBlock(n_kernels=n_kernels, config=config),
    )


@logger_wraps(level="INFO")
@curry
def Decoder(x: tf.Tensor, skips: Sequence[tf.Tensor], config: dict = model_config()):
    """
    Pass input through decoder consisting of upsampling and return output
    """
    # skips and config is in descending order, reverse to ascending
    levels = reversed(
        [
            DecoderLevel(skip=skip, n_kernels=n_kernels)
            # Ignore first block (bottleneck)
            for skip, n_kernels in (zip(skips, config["n_kernels_per_block"][1:]))
        ]
    )

    return tz.pipe(
        x,
        lambda x: tz.pipe(x, *levels),  # Run x through each decoder level
        layers.Conv3D(  # Final 1x1 convolution
            config["n_kernels_last"],
            kernel_size=(1, 1, 1),
            kernel_initializer=config["initializer"],
            padding="same",
            data_format="channels_last",
        ),
        (
            layers.Activation(config["final_layer_activation"])
            if config["final_layer_activation"]
            else tz.identity
        ),
    )


@logger_wraps(level="INFO")
@curry
def UNet(config: dict = model_config()) -> Model:
    """
    Construct a U-Net model
    """
    input_ = layers.Input(
        shape=(
            config["input_height"],
            config["input_width"],
            config["input_depth"],
            config["input_dim"],
        ),
        batch_size=config["batch_size"],
    )
    return tz.pipe(
        input_,
        Encoder(config=config),
        unpack_args(Decoder(config=config)),
        lambda output: Model(input_, output, name="U-Net"),
    )
