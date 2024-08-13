"""
Define a U-Net model using Keras functional API

Reference: https://github.com/christianversloot/machine-learning-articles/blob/main/how-to-build-a-u-net-for-image-segmentation-with-tensorflow-and-keras.md
"""

from typing import Sequence
import tensorflow as tf
from ..utils.logging import logger_wraps
from ..config import configuration, Configuration
from ..utils.wrappers import curry
from ..utils.sequence import growby_accum
from ..utils.common import conditional, unpack_args
from .layers import CentreCrop3D, MCDropout

import toolz as tz
from keras import layers


@logger_wraps(level="INFO")
@curry
def ConvLayer(
    x: tf.Tensor,
    n_kernels: int,
    kernel_size: tuple[int, ...],
    initializer: str,
    use_batch_norm: bool,
    activation: str,
    dropout_rate: float,
    mc_dropout: bool,
    bn_epsilon: float,
    bn_momentum: float,
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
        conditional(
            use_batch_norm,
            layers.BatchNormalization(epsilon=bn_epsilon, momentum=bn_momentum),
            tz.identity,
        ),
        layers.Activation(activation),
        conditional(
            mc_dropout,
            MCDropout(dropout_rate),
            layers.Dropout(dropout_rate),
        ),
    )


@logger_wraps(level="INFO")
@curry
def ConvBlock(
    x: tf.Tensor,
    n_kernels: int,
    mc_dropout: bool,
    config: Configuration,
):
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
                dropout_rate=config["dropout_rate"],
                mc_dropout=mc_dropout,
                bn_epsilon=config["batch_norm_epsilon"],
                bn_momentum=config["batch_norm_decay"],
            )
            for _ in range(config["n_convolutions_per_block"])
        ]
    )


@logger_wraps(level="INFO")
@curry
def Encoder(x: tf.Tensor, mc_dropout: bool, config: Configuration):
    """
    Pass input through encoder and return (output, skip_connections)
    """
    levels = [
        # need to bind n_kernels because all lambdas share same variable `level`
        lambda x, n_kernels=config["n_kernels_per_block"][level]: tz.pipe(
            x,
            layers.MaxPool3D((2, 2, 2), strides=(2, 2, 2), data_format="channels_last"),
            ConvBlock(n_kernels=n_kernels, mc_dropout=mc_dropout, config=config),
        )
        for level in range(1, config["n_levels"])  # Exclude first block
    ]

    return tz.pipe(
        x,
        ConvBlock(
            n_kernels=config["n_kernels_init"], mc_dropout=mc_dropout, config=config
        ),
        # Repeatedly apply downsample and ConvBlock to input to get skip connection list
        growby_accum(fs=levels),
        list,
        # last element is from bottleneck so exclude from skip connections
        lambda skip_inputs: (skip_inputs[-1], skip_inputs[:-1]),
    )


@logger_wraps(level="INFO")
@curry
def DecoderLevel(
    x: tf.Tensor,
    skip: tf.Tensor,
    n_kernels: int,
    mc_dropout: bool,
    config: Configuration,
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
        CentreCrop3D(skip.shape[1:]),  # type: ignore
        lambda cropped_x: layers.Concatenate(axis=-1)([skip, cropped_x]),
        ConvBlock(n_kernels=n_kernels, mc_dropout=mc_dropout, config=config),
    )


@logger_wraps(level="INFO")
@curry
def Decoder(
    x: tf.Tensor,
    skips: Sequence[tf.Tensor],
    mc_dropout: bool,
    config: Configuration,
):
    """
    Pass input through decoder consisting of upsampling and return output
    """
    # skips and config is in descending order, reverse to ascending
    levels = reversed(
        [
            DecoderLevel(
                skip=skip, mc_dropout=mc_dropout, n_kernels=n_kernels, config=config
            )
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
def _GenericUNet(
    input_: tf.Tensor, mc_dropout: bool, config: Configuration
) -> tf.Tensor:
    """
    Pass input to U-Net and return the output

    Parameters
    ----------
    mc_dropout : bool
        Create a MC Dropout U-Net, dropout is enabled during testing
    """

    return tz.pipe(
        input_,
        Encoder(mc_dropout=mc_dropout, config=config),
        unpack_args(Decoder(mc_dropout=mc_dropout, config=config)),
    )  # type: ignore


@logger_wraps(level="INFO")
def UNet(
    input_: tf.Tensor, config: Configuration = configuration()
) -> tuple[tf.Tensor, str]:
    """
    Pass input to U-Net and return the output and model name
    """
    return _GenericUNet(input_, mc_dropout=False, config=config), "UNet"


@logger_wraps(level="INFO")
def MCDropoutUNet(
    input_: tf.Tensor, config: Configuration = configuration()
) -> tuple[tf.Tensor, str]:
    """
    Pass input to MC Dropout U-Net and return the output and model name
    """
    return _GenericUNet(input_, mc_dropout=True, config=config), "MCDropoutUNet"
