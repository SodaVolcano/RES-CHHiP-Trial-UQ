"""
Define a U-Net model

References: 
    https://github.com/christianversloot/machine-learning-articles/blob/main/how-to-build-a-u-net-for-image-segmentation-with-tensorflow-and-keras.md
    https://github.com/milesial/Pytorch-UNet/tree/master
"""

from typing import Callable, List, Optional, Tuple, override

import toolz as tz
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import Configuration
from ..utils.sequence import growby_accum


def _calc_n_kernels(n_init: int, level: int, bound: int):
    return min(bound, n_init * 2**level)


class ConvLayer(nn.Module):
    """
    Convolution layer: Conv3D -> [InstanceNorm] -> Activation -> Dropout

    Parameters
    ----------
    in_channels : int
        Number of input channels
    out_channels : int
        Number of output channels
    kernel_size : int
        Kernel size for convolution
    use_instance_norm : bool
        Whether to use instance normalisation
    activation : Callable[..., nn.Module]
        Function returning an activation module, e.g. `nn.ReLU`
    dropout_rate : float
        Dropout rate
    inorm_epsilon : float
        Epsilon for instance normalisation, small constant added to the variance
        to avoid division by zero
    inorm_momentum : float
        Momentum for instance normalisation, controls update rate of running
        mean and variance during inference. Mean and variance are used to
        normalise inputs during inference.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        use_instance_norm: bool,
        activation: Callable[..., nn.Module],
        dropout_rate: float,
        inorm_epsilon: float,
        inorm_momentum: float,
    ):
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
                nn.InstanceNorm3d(
                    out_channels, eps=inorm_epsilon, momentum=inorm_momentum
                )
                if use_instance_norm
                else nn.Identity()
            ),
            activation(),
            nn.Dropout(dropout_rate, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class ConvBlock(nn.Module):
    """
    A group of convolution layers, each consisting of Conv3D -> [InstanceNorm] -> Activation -> Dropout
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
            Conv3D -> [InstanceNorm] -> Activation -> Dropout
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
                    use_instance_norm=config["use_instance_norm"],
                    activation=config["activation"],
                    dropout_rate=config["dropout_rate"],
                    inorm_epsilon=config["instance_norm_epsilon"],
                    inorm_momentum=config["instance_norm_decay"],
                )
                for i in range(n_convolutions)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class DownConvBlock(nn.Module):
    """
    A single down convolution block, downsample -> conv block

    Parameters
    ----------
    level : int
        Level in the U-Net (0-indexed)
    """

    def __init__(self, level: int, config: Configuration):
        super().__init__()
        self.layer = nn.Sequential(
            nn.MaxPool3d(kernel_size=2, stride=(2, 2, 2)),
            ConvBlock(
                in_channels=_calc_n_kernels(
                    config["n_kernels_init"], level - 1, config["n_kernels_max"]
                ),
                out_channels=_calc_n_kernels(
                    config["n_kernels_init"], level, config["n_kernels_max"]
                ),
                n_convolutions=config["n_convolutions_per_block"],
                config=config,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)


class Encoder(nn.Module):
    """
    U-Net Encoder, conv block -> downconv blocks... -> (output, skip_connections)

    Parameters
    ----------
    config: Configuration
        Dictionary containing configuration parameters
    """

    def __init__(
        self,
        config: Configuration,
    ):
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
            in_channels=_calc_n_kernels(
                config["n_kernels_init"], level + 1, config["n_kernels_max"]
            ),
            out_channels=_calc_n_kernels(
                config["n_kernels_init"], level, config["n_kernels_max"]
            ),
            kernel_size=config["kernel_size"],
            stride=(2, 2, 2),
        )
        self.conv = ConvBlock(
            # in_channel is doubled because skip connection is concatenated to input
            in_channels=_calc_n_kernels(
                config["n_kernels_init"], level, config["n_kernels_max"]
            )
            * 2,
            out_channels=_calc_n_kernels(
                config["n_kernels_init"], level, config["n_kernels_max"]
            ),
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
        # pad scans from END of list, so we need to reverse bounds!
        x = F.pad(x, list(reversed(list(bounds))))
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class Decoder(nn.Module):
    """
    U-Net Decoder, upconv blocks... -> 1x1 conv -> (output)

    Parameters
    ----------
    config : Configuration
        Dictionary containing configuration parameters
    deep_supervision : bool
        If True, a list of outputs from each level is returned. The outputs
        are from convolving the output of each level with a 1x1 convolution.
    """

    def __init__(self, config: Configuration, deep_supervision: bool = True):
        super().__init__()
        self.deep_supervision = deep_supervision
        self.levels = nn.ModuleList(
            [
                UpConvBlock(level=level, config=config)
                # is zero-indexed so max level is n_levels - 1, -1 again for bottleneck
                # if encoder have layer [3, 2, 1], this is [2, 1, 0]
                # i.e. encoder excludes first level (0), decoder ignores last level (3)
                for level in range(config["n_levels"] - 2, -1, -1)
            ]
        )
        if deep_supervision:
            self.last_conv = nn.ModuleList(
                [
                    (
                        nn.Conv3d(
                            in_channels=_calc_n_kernels(
                                config["n_kernels_init"], level, config["n_kernels_max"]
                            ),
                            out_channels=config["n_kernels_last"],
                            kernel_size=1,
                            bias=False,
                        )
                        if level != config["n_levels"] - 2
                        else nn.Identity()
                    )  # ignore lowest level from deep supervision
                    for level in range(config["n_levels"] - 2, -1, -1)
                ]
            )
        else:
            self.last_conv = nn.Conv3d(
                in_channels=config["n_kernels_init"],
                out_channels=config["n_kernels_last"],
                kernel_size=1,
                bias=False,
            )
        self.last_activation = config["final_layer_activation"]()

    def _forward_deep_supervision(self, x, skips, logits: bool) -> List[torch.Tensor]:
        """
        Forward pass with deep supervision, returns a list of outputs from each level
        """
        deep_outputs = []
        for i, (level, skip) in enumerate(zip(self.levels, reversed(skips))):
            x = level(x, skip)
            # ignore lowest level of decoder (2nd lowest level of U-Net)
            if i == 0:
                continue
            conved_x = self.last_conv[i](x)  # type: ignore
            if not logits:
                conved_x = self.last_activation(conved_x)
            deep_outputs.append(conved_x)

        return deep_outputs

    def forward(
        self,
        x: torch.Tensor,
        skips: List[torch.Tensor],
        logits: bool = False,
    ) -> torch.Tensor | List[torch.Tensor]:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input tensor
        skips : List[torch.Tensor]
            Skip connections from the encoder
        logits : bool
            Whether to return logits or not
        """
        if self.deep_supervision:
            return self._forward_deep_supervision(x, skips, logits)

        for level, skip in zip(self.levels, reversed(skips)):
            x = level(x, skip)
        x = self.last_conv(x)
        return self.last_activation(x) if not logits else x


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

    Parameters
    ----------
    config : Configuration
        Dictionary containing configuration parameters
    deep_supervision : bool
        If True, a list of outputs from each decoder level is returned.
        The outputs are from convolving the output of each level with
        a 1x1 convolution. **NOTE** that the last two level of the U-Net
        (bottleneck and last layer of decoder) is not returned.
    """

    def __init__(self, config: Configuration, deep_supervision: bool = True):
        super().__init__()
        self.encoder = Encoder(config)
        self.decoder = Decoder(config, deep_supervision=deep_supervision)
        self.deep_supervision = deep_supervision

    def forward(self, x: torch.Tensor, logits: bool = False) -> torch.Tensor:
        encoded, skips = self.encoder(x)
        return self.decoder(encoded, skips, logits)

    def last_activation(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder.last_activation(x)


class MCDropoutUNet(nn.Module):
    """
    Wrapper around U-Net to apply dropout during evaluation
    """

    @override
    def __init__(self, model: UNet):
        """
        Wrap an existing U-Net or create a new one using the provided configuration
        """
        super().__init__()
        self.model = model

    @override
    def forward(self, x: torch.Tensor, logits: bool = False) -> torch.Tensor:
        return self.model(x, logits)

    @override
    def eval(self):
        def activate_dropout(module):
            if isinstance(module, nn.Dropout):
                module.train(True)

        # Apply dropout during evaluation
        self.model.eval()
        self.model.apply(activate_dropout)
        return self
