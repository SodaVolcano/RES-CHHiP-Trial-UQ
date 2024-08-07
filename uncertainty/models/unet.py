"""
Define a U-Net model using Keras functional API

Reference: https://github.com/christianversloot/machine-learning-articles/blob/main/how-to-build-a-u-net-for-image-segmentation-with-tensorflow-and-keras.md
"""

from uncertainty.data.preprocessing import crop_nd
from ..utils.logging import logger_wraps
from ..common.constants import model_config
from ..utils.wrappers import curry
from ..utils.sequence import growby_accum
from ..utils.common import unpack_args

import toolz as tz
from tensorflow.keras import layers, Model


@logger_wraps(level="INFO")
@curry
def conv_layer(
    x,
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
def conv_block(x, n_kernels, config: dict = model_config()):
    """Pass input through n convolution layers"""
    return tz.pipe(
        x,
        *[
            conv_layer(
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
def unet_encoder(x, config: dict = model_config()):
    """
    Pass input through encoder and return (output, skip_connections)
    """
    levels = [
        lambda x: tz.pipe(
            x,
            layers.MaxPool3D((2, 2, 2), strides=(2, 2, 2), data_format="channels_last"),
            conv_block(n_kernels=config["n_kernels_per_block"][level], config=config),
        )
        for level in range(1, config["n_levels"])  # Exclude first block
    ]

    return tz.pipe(
        x,
        conv_block(n_kernels=config["n_kernels_init"], config=config),
        # Repeatedly apply downsample and conv_block to input to get skip connection list
        growby_accum(fs=levels),
        list,
        # last element is from bottleneck so exclude from skip connections
        lambda skip_inputs: (skip_inputs[-1], skip_inputs[:-1]),
    )


@logger_wraps(level="INFO")
@curry
def decoder_level(x, skip, n_kernels: int, config: dict = model_config()):
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
        crop_nd(new_shape=skip.shape[1:-1]),  # Crop to match size of skip connection
        lambda cropped_x: layers.Concatenate(axis=-1)([skip, cropped_x]),
        conv_block(n_kernels=n_kernels, config=config),
    )


@logger_wraps(level="INFO")
@curry
def unet_decoder(x, skips, config: dict = model_config()):
    """
    Pass input through decoder consisting of upsampling and return output
    """
    # skips and config is in descending order, reverse to ascending
    levels = reversed(
        [
            decoder_level(skip=skip, n_kernels=n_kernels)
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
def unet(config: dict = model_config()) -> Model:
    """
    Construct a U-Net model
    """
    input_ = layers.Input(
        shape=(
            config["input_height"],
            config["input_width"],
            config["input_depth"],
            config["input_dim"],
        )
    )
    return tz.pipe(
        input_,
        unet_encoder(config=config),
        unpack_args(unet_decoder(config=config)),
        lambda output: Model(input_, output, name="U-Net"),
    )
