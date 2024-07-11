from functools import reduce
from ..common.constants import model_config
from ..common.utils import curry, grow_seq, unpack_args

from typing import Callable
import toolz as tz
from toolz import curried
from tensorflow.python.keras import layers, Model  # TODO: replace with tensorflow.keras


def conv_layer(x, n_kernels, kernel_size, initializer, use_batch_norm, activation):
    """Convolution followed by (optionally) BN and activation"""
    return tz.pipe(
        x,
        layers.Conv2D(
            n_kernels,
            kernel_size,
            initializer,
            padding="same",  # Keep dimensions the same
        ),
        layers.BatchNormalization() if use_batch_norm else tz.identity,
        layers.Activation(activation),
    )


def conv_block(x, config: dict = model_config()):
    """Pass input through n convolution layers"""
    return tz.pipe(
        x,
        *[
            conv_layer(
                config["n_kernels"],
                config["kernel_size"],
                config["initializer"],
                config["use_batch_norm"],
                config["activation"],
            )
            for _ in range(config["n_convolutions_per_block"])
        ]
    )


def unet_encoder(x, config: dict = model_config()):
    """
    Pass input through encoder and return (output, skip_connections)
    """
    downsample = layers.MaxPool2D((2, 2), strides=(2, 2))
    return tz.pipe(
        x,
        # skip_inputs = [conv(x),], level 1 of encoder
        conv_block(config=config),
        # Repeatedly downsample and apply conv_block [conv(x), conv(down(conv(x))), ...]
        grow_seq(
            f=lambda x: conv_block(downsample(x), config=config),
            length=config["n_levels"] - 1,
        ),
        list,
        # last element is from bottleneck so exclude from skip connections
        lambda skip_inputs: (skip_inputs[-1], skip_inputs[:-1]),
    )


@curry
def decoder_level(x, skip, config: dict = model_config()):
    """
    One level of decoder path: upsample, crop, concat with skip, and convolve the input
    """

    return tz.pipe(
        x,
        layers.Conv2DTranspose(
            config["n_kernels"] // 2,  # Halve dimensions because we are concatenating
            config["kernel_size"],
            strides=(2, 2),
            kernel_initializer=config["initializer"],
        ),
        layers.CenterCrop(  # Crop to match size of skip connection
            height=skip.shape[1], width=skip.shape[2]
        ),
        lambda cropped_x: layers.Concatenate(axis=-1)(skip, cropped_x),
        conv_block(config=config),
    )


@curry
def unet_decoder(x, skips, config: dict = model_config()):
    """
    Pass input through decoder consisting of upsampling and return output
    """
    return tz.pipe(
        # skips is in descending order, reverse to ascending
        tz.concat([[x], reversed(skips)]),
        # Run x and corresponding skip connection through each decoder_level
        curried.reduce(function=decoder_level(config=config)),
        layers.Conv2D(
            config["n_kernels_last"],
            (1, 1),
            kernel_initializer=config["initializer"],
            padding="same",
        ),
        (
            layers.Activation(config["final_layer_activation"])
            if config["final_layer_activation"]
            else tz.identity
        ),
    )


def unet(config: dict = model_config()) -> Model:
    """
    Construct a U-Net model
    """
    input_ = (
        layers.Input(
            (config["input_height"], config["input_width"], config["input_dim"])
        ),
    )
    return tz.pipe(
        input_,
        unet_encoder(config=config),
        unpack_args(unet_decoder(config=config)),
        lambda output: Model(input_, output, name="U-Net"),
    )
