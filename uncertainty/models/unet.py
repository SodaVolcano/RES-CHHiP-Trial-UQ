"""
Define a U-Net model

References: 
    https://github.com/christianversloot/machine-learning-articles/blob/main/how-to-build-a-u-net-for-image-segmentation-with-tensorflow-and-keras.md
    https://github.com/milesial/Pytorch-UNet/tree/master
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable, List, Tuple

from ..config import Configuration
from ..utils.sequence import growby_accum
import toolz as tz


class ConvLayer(nn.Module):
    """
    Single convolution layer: Conv3D -> [BN] -> Activation -> Dropout
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        use_batch_norm: bool,
        activation: Callable[..., nn.Module],
        dropout_rate: float,
        bn_epsilon: float,
        bn_momentum: float,
    ):
        """
        Convolution layer: Conv3D -> [BN] -> Activation -> Dropout

        Parameters
        ----------
        in_channels : int
            Number of input channels
        out_channels : int
            Number of output channels
        kernel_size : int
            Kernel size for convolution
        use_batch_norm : bool
            Whether to use batch normalisation
        activation : Callable[..., nn.Module]
            Function returning an activation module, e.g. `nn.ReLU`
        dropout_rate : float
            Dropout rate
        bn_epsilon : float
            Epsilon for batch normalisation, small constant added to the variance
            to avoid division by zero
        bn_momentum : float
            Momentum for batch normalisation, controls update rate of running
            mean and variance during inference. Mean and variance are used to
            normalise inputs during inference.
        """
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv3d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=1,
                bias=False,
            ),
            (
                nn.BatchNorm3d(out_channels, eps=bn_epsilon, momentum=bn_momentum)
                if use_batch_norm
                else nn.Identity()
            ),
            activation(),
            nn.Dropout(dropout_rate, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class ConvBlock(nn.Module):
    """
    A group of convolution layers, each consisting of Conv3D -> [BN] -> Activation -> Dropout
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_convolutions: int,
        config: Configuration,
    ):
        """
        Multiple convolution layers in a single level of the U-Net

        Parameters
        ----------
        in_channels : int
            Number of input channels
        out_channels : int
            Number of output channels
        n_convolutions : int
            Number of convolution layers. A convolution layer consists of
            Conv3D -> [BN] -> Activation -> Dropout
        config : Configuration
            Dictionary containing configuration parameters
        """
        super().__init__()

        self.layers = nn.Sequential(
            *[
                ConvLayer(
                    in_channels=in_channels if i == 0 else out_channels,
                    out_channels=out_channels,
                    kernel_size=config["kernel_size"],
                    use_batch_norm=config["use_batch_norm"],
                    activation=config["activation"],
                    dropout_rate=config["dropout_rate"],
                    bn_epsilon=config["batch_norm_epsilon"],
                    bn_momentum=config["batch_norm_decay"],
                )
                for i in range(n_convolutions)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class DownConvBlock(nn.Module):
    """
    A downconvolution block, downsample -> conv block
    """

    def __init__(self, level: int, config: Configuration):
        """
        A single down convolution block, downsample -> conv block

        Parameters
        ----------
        level : int
            Level in the U-Net (0-indexed)
        """
        super().__init__()
        self.layer = nn.Sequential(
            nn.MaxPool3d(kernel_size=2, stride=(2, 2, 2)),
            ConvBlock(
                in_channels=config["n_kernels_init"] * 2 ** (level - 1),
                out_channels=config["n_kernels_init"] * 2**level,
                n_convolutions=config["n_convolutions_per_block"],
                config=config,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)


class Encoder(nn.Module):
    """
    U-Net Encoder, conv block -> downconv blocks... -> (output, skip_connections)
    """

    def __init__(
        self,
        config: Configuration,
    ):
        """
        Encoder branch of U-Net, returns (output, skip_connections)

        Parameters
        ----------
        config: Configuration
            Dictionary containing configuration parameters
        """
        super().__init__()

        self.levels = nn.ModuleList(
            [
                DownConvBlock(
                    level=level,
                    config=config,
                )
                for level in range(1, config["n_levels"])
            ]
        )

        self.init_block = ConvBlock(
            in_channels=config["input_channel"],
            out_channels=config["n_kernels_init"],
            n_convolutions=config["n_convolutions_per_block"],
            config=config,
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        return tz.pipe(
            x,
            self.init_block,
            growby_accum(fs=self.levels),
            list,
            # last element is from bottleneck so exclude from skip connections
            lambda skip_inputs: (skip_inputs[-1], skip_inputs[:-1]),
        )  # type: ignore


class UpConvBlock(nn.Module):
    """
    Upconvolution block, (input, skip_connections) -> upsample -> concat -> conv block
    """

    def __init__(self, level: int, config: Configuration) -> None:
        super().__init__()
        self.up = nn.ConvTranspose3d(
            in_channels=config["n_kernels_init"] * 2 ** (level + 1),
            out_channels=config["n_kernels_init"] * 2**level,
            kernel_size=config["kernel_size"],
            stride=(2, 2, 2),
        )
        self.conv = ConvBlock(
            # in_channel is doubled because skip connection is concatenated to input
            in_channels=config["n_kernels_init"] * 2 ** (level + 1),
            out_channels=config["n_kernels_init"] * 2**level,
            n_convolutions=config["n_convolutions_per_block"],
            config=config,
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # use last 3 dimensions in (N, C, D, H, W) format
        diff_dims = [skip.size(i) - x.size(i) for i in range(2, 5)]
        # left, right, top, bottom, front, back
        bounds = tz.interleave(
            [
                [dim // 2 for dim in diff_dims],
                [dim - dim // 2 for dim in diff_dims],
            ]
        )
        x = F.pad(x, list(bounds))
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class Decoder(nn.Module):
    """
    U-Net Decoder, upconv blocks... -> 1x1 conv -> (output)
    """

    def __init__(self, config: Configuration):
        super().__init__()
        self.levels = nn.ModuleList(
            [
                UpConvBlock(level=level, config=config)
                # is zero-indexed so max level is n_levels - 1, -1 again for bottleneck
                for level in range(config["n_levels"] - 2, -1, -1)
            ]
        )

        self.last_conv = nn.Sequential(
            nn.Conv3d(
                in_channels=config["n_kernels_init"],
                out_channels=config["n_kernels_last"],
                kernel_size=1,
                bias=False,
            ),
            (
                config["final_layer_activation"]()
                if config["final_layer_activation"]
                else nn.Identity()
            ),
        )

    def forward(self, x: torch.Tensor, skips: List[torch.Tensor]) -> torch.Tensor:
        for level, skip in zip(self.levels, reversed(skips)):
            x = level(x, skip)
        return self.last_conv(x)


class UNet(nn.Module):
    """
    A U-shaped architecture for image segmentation

    Consists of N resolution levels and divided into the Encoder
    and Decoder branches. The Encoder downsamples and convolves
    the input image in each level to extract features, while the
    Decoder upsamples and concatenates output from the previous
    level with the corresponding output from the Encoder of the
    same level and then apply convolutions. A final 1x1 convolution
    produces the final output.
    """

    def __init__(self, config: Configuration):
        super().__init__()
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded, skips = self.encoder(x)
        return self.decoder(encoded, skips)


class MCDropoutUNet(UNet):
    def eval(self):
        def activate_dropout(module):
            if isinstance(module, nn.Dropout):
                module.train(True)

        # Apply dropout during evaluation
        super().eval()
        self.apply(activate_dropout)
        return self
