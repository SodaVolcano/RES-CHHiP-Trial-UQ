"""
Modules used to build a U-Net model.

References: 
    https://github.com/christianversloot/machine-learning-articles/blob/main/how-to-build-a-u-net-for-image-segmentation-with-tensorflow-and-keras.md
    https://github.com/milesial/Pytorch-UNet/tree/master
"""

from typing import Callable, List, Tuple

import toolz as tz
import torch
import torch.nn as nn
import torch.nn.functional as F

from uncertainty.config import auto_match_config

from ..utils.sequence import growby_accum
from ..utils.wrappers import curry


def _calc_n_kernels(n_init: int, level: int, bound: int):
    return min(bound, n_init * 2**level)


@curry
def _concat_with_skip(x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
    # use last 3 dimensions, because we are in (N, C, D, H, W) format
    diff_dims = [skip.size(i) - x.size(i) for i in range(2, 5)]
    # (left, right), (top, bottom), (front, back)
    bounds = tz.interleave(
        [
            [dim // 2 for dim in diff_dims],
            [dim - dim // 2 for dim in diff_dims],
        ]
    )
    # pad scans from END of list, so we need to reverse bounds!
    x = F.pad(x, list(reversed(list(bounds))))
    return torch.cat([skip, x], dim=1)


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
    instance_norm_epsilon : float
        Epsilon for instance normalisation, small constant added to the variance
        to avoid division by zero
    instance_norm_momentum : float
        Momentum for instance normalisation, controls update rate of running
        mean and variance during inference. Mean and variance are used to
        normalise inputs during inference.
    """

    @auto_match_config(prefixes=["unet"])
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        use_instance_norm: bool,
        activation: Callable[..., nn.Module],
        dropout_rate: float,
        instance_norm_epsilon: float,
        instance_norm_momentum: float,
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
                    out_channels,
                    eps=instance_norm_epsilon,
                    momentum=instance_norm_momentum,
                )
                if use_instance_norm
                else nn.Identity()
            ),
            activation(),
            nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class ConvBlock(nn.Module):
    """
    A group of convolution layers, each consisting of Conv3D -> [InstanceNorm] -> Activation -> Dropout

    Parameters
    ----------
    level : int
        Resolution level of the U-Net that this block is on (zero-indexed)
    in_channels : int
        Number of input channels
    out_channels : int
        Number of output channels
    n_convolutions : int
        Number of convolution layers. A convolution layer consists of
        Conv3D -> [InstanceNorm] -> Activation -> Dropout
    n_levels : int
        Total number of resolution levels in the U-Net
    dropout_rate : float
        Dropout rate, only enabled at the last level (i.e. this parameter
        is ignored for all levels except the last)
    kernel_size : int
        Kernel size for convolution
    kwargs : dict
        Additional parameters to pass to ConvLayer
    """

    @auto_match_config(prefixes=["unet"])
    def __init__(
        self,
        level: int,
        in_channels: int,
        out_channels: int,
        n_convolutions: int,
        n_levels: int,
        dropout_rate: float,
        kernel_size: int,
        **kwargs: dict,
    ):
        super().__init__()
        # Only enable dropout if at last level
        if level < n_levels - 1:
            dropout_rate = 0.0

        self.conv_layers = nn.Sequential(
            *[
                ConvLayer(
                    in_channels=in_channels if i == 0 else out_channels,
                    out_channels=out_channels,
                    dropout_rate=dropout_rate,
                    kernel_size=kernel_size,
                    **kwargs,  # type: ignore
                )
                for i in range(n_convolutions)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv_layers(x)


class DownConvBlock(nn.Module):
    """
    A single down convolution block, downsample -> conv block

    Parameters
    ----------
    level : int
        Resolution level of the U-Net that this block is on (zero-indexed)
    n_kernels_init : int
        Initial number of kernels at the first level
    n_kernels_max : int
        Maximum number of kernels at any level
    n_convolutions_per_block : int
        Number of convolutions per block
    """

    @auto_match_config(prefixes=["unet"])
    def __init__(
        self,
        level: int,
        n_kernels_init: int,
        n_kernels_max: int,
        n_convolutions_per_block: int,
        **kwargs,
    ):
        super().__init__()
        self.conv_blocks = nn.Sequential(
            nn.MaxPool3d(kernel_size=2, stride=(2, 2, 2)),
            ConvBlock(
                level=level,
                in_channels=_calc_n_kernels(n_kernels_init, level - 1, n_kernels_max),
                out_channels=_calc_n_kernels(n_kernels_init, level, n_kernels_max),
                n_convolutions=n_convolutions_per_block,
                **kwargs,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv_blocks(x)


class Encoder(nn.Module):
    """
    U-Net Encoder, conv block -> downconv blocks... -> (output, skip_connections)

    Parameters
    ----------
    n_levels : int
        Total number of resolution levels in the U-Net
    input_channels : int
        Number of input channels in the input tensor
    n_kernels_init : int
        Initial number of kernels at the first level
    n_convolutions_per_block : int
        Number of convolutions per block
    kwargs : dict
        Additional parameters to pass to ConvBlock
    """

    @auto_match_config(prefixes=["unet"])
    def __init__(
        self,
        n_levels: int,
        input_channels: int,
        n_kernels_init: int,
        n_convolutions_per_block: int,
        **kwargs,
    ):
        super().__init__()

        self.levels = nn.ModuleList(
            [
                DownConvBlock(
                    level=level,
                    n_levels=n_levels,
                    n_kernels_init=n_kernels_init,
                    n_convolutions_per_block=n_convolutions_per_block,
                    **kwargs,
                )
                for level in range(1, n_levels)
            ]
        )

        self.init_block = ConvBlock(
            level=0,
            n_levels=n_levels,
            in_channels=input_channels,
            out_channels=n_kernels_init,
            n_convolutions=n_convolutions_per_block,
            **kwargs,
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

    Parameters
    ----------
    level : int
        Resolution level of the U-Net that this block is on (zero-indexed)
    n_kernels_init : int
        Initial number of kernels at the first level
    n_kernels_max : int
        Maximum number of kernels at any level
    kernel_size : int
        Kernel size for convolution
    n_convolutions_per_block : int
        Number of convolutions per block
    kwargs : dict
        Additional parameters to pass to ConvBlock
    """

    @auto_match_config(prefixes=["unet"])
    def __init__(
        self,
        level: int,
        n_kernels_init: int,
        n_kernels_max: int,
        kernel_size: int,
        n_convolutions_per_block: int,
        **kwargs,
    ) -> None:
        super().__init__()
        self.up = nn.ConvTranspose3d(
            in_channels=_calc_n_kernels(n_kernels_init, level + 1, n_kernels_max),
            out_channels=_calc_n_kernels(n_kernels_init, level, n_kernels_max),
            kernel_size=kernel_size,
            stride=(2, 2, 2),
        )
        self.conv = ConvBlock(
            level=level,
            # in_channel is doubled because skip connection is concatenated to input
            in_channels=_calc_n_kernels(n_kernels_init, level, n_kernels_max) * 2,
            out_channels=_calc_n_kernels(n_kernels_init, level, n_kernels_max),
            n_convolutions=n_convolutions_per_block,
            kernel_size=kernel_size,
            **kwargs,
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        return tz.pipe(x, self.up, _concat_with_skip(skip=skip), self.conv)


class Decoder(nn.Module):
    """
    U-Net Decoder, upconv blocks... -> 1x1 conv -> (output)

    Parameters
    ----------
    n_levels : int
        Total number of resolution levels in the U-Net
    n_kernels_init : int
        Initial number of kernels at the first level
    output_channels : int
        Number of output channels
    n_kernels_max : int
        Maximum number of kernels at any level
    final_layer_activation : Callable[..., nn.Module]
        Function returning an activation module, e.g. `nn.Softmax`
    deep_supervision : bool
        If True, a list of outputs from each level is returned. The outputs
        are from convolving the output of each level with a 1x1 convolution.
    """

    @auto_match_config(prefixes=["unet", "training"])
    def __init__(
        self,
        n_levels: int,
        n_kernels_init: int,
        output_channels: int,
        n_kernels_max: int,
        final_layer_activation: Callable[..., nn.Module],
        deep_supervision: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.deep_supervision = deep_supervision
        self.levels = nn.ModuleList(
            [
                UpConvBlock(
                    level=level,
                    n_levels=n_levels,
                    n_kernels_init=n_kernels_init,
                    n_kernels_max=n_kernels_max,
                    **kwargs,
                )
                # is zero-indexed so max level is n_levels - 1, -1 again for bottleneck
                # if encoder have layer [3, 2, 1], this is [2, 1, 0]
                # i.e. encoder excludes first level (0), decoder ignores last level (3)
                for level in range(n_levels - 2, -1, -1)
            ]
        )
        if deep_supervision:
            self.last_conv = nn.ModuleList(
                [
                    (
                        nn.Conv3d(
                            in_channels=_calc_n_kernels(
                                n_kernels_init, level, n_kernels_max
                            ),
                            out_channels=output_channels,
                            kernel_size=1,
                            bias=False,
                        )
                        if level != n_levels - 2
                        else nn.Identity()  # lowest 2 levels have no deep supervision, so no conv!
                    )
                    for level in range(n_levels - 2, -1, -1)
                ]
            )
        else:
            self.last_conv = nn.Conv3d(
                in_channels=n_kernels_init,
                out_channels=output_channels,
                kernel_size=1,
                bias=False,
            )
        self.last_activation = final_layer_activation()

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
        if self.deep_supervision and self.training:
            return self._forward_deep_supervision(x, skips, logits)

        for level, skip in zip(self.levels, reversed(skips)):
            x = level(x, skip)

        x = (
            self.last_conv[-1](x)
            if isinstance(self.last_conv, nn.ModuleList)
            else self.last_conv(x)
        )
        return self.last_activation(x) if not logits else x
