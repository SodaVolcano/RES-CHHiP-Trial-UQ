"""
ConfidNet, auxiliary network for predicting true class probability

Reference:
    https://github.com/valeoai/ConfidNet
    Addressing Failure Prediction by Learning Model Confidence, Corbiere et al.
"""

from typing import Callable

import toolz as tz
import torch
from torch import nn

from ..config import auto_match_config
from ..metrics import ConfidNetMSELoss
from .unet import UNet
from .unet_modules import _calc_n_kernels, _concat_with_skip


class UNetConfidNetEncoder(nn.Module):
    """
    Convert a U-Net into a ConfidNet encoder

    Outputs both the input to ConfidNet and output of the segmentation model.
    """

    def __init__(self, unet: UNet):
        super().__init__()
        self.unet = unet

    def _decoder_forward(
        self, x: torch.Tensor, skips: list[torch.Tensor], logits: bool
    ) -> torch.Tensor:
        """
        Modify decoder of U-Net to skip the last level
        """
        confidnet_in = x  # input to auxiliary network (confidnet)
        for i, (level, skip) in enumerate(
            zip(self.unet.decoder.levels, reversed(skips))
        ):
            x = level(x, skip)
            if i == len(skips) - 2:  # level 1 of unet
                confidnet_in = x
        x = (
            self.unet.decoder.last_conv[-1](x)
            if isinstance(self.unet.decoder.last_conv, nn.ModuleList)
            else self.unet.decoder.last_conv(x)
        )
        if not logits:
            x = self.last_activation(x)

        return tz.pipe(
            confidnet_in,
            # upconv and concat with skip but don't do final convolution
            self.unet.decoder.levels[-1].up if len(skips) > 1 else tz.identity,
            _concat_with_skip(skip=skips[0]),
            lambda confid_in: (confid_in, x),
        )  # type: ignore

    def forward(self, x: torch.Tensor, logits: bool) -> torch.Tensor:
        encoded, skip = self.unet.encoder(x)
        return self._decoder_forward(encoded, skip, logits)

    def last_activation(self, x: torch.Tensor) -> torch.Tensor:
        return self.unet.decoder.last_activation(x)


class UNetConfidNet(nn.Module):
    """
    An auxiliary network for predicting true class probability of a U-Net

    The architecture uses layers from a U-Net as the encoder and adds a series of
    convolutional layers to predict the confidence of the segmentation model.

    Parameters
    ----------
    unet : UNet
        U-Net model to use as the encoder.
    n_kernels_init : int
        Number of kernels in the first layer of the auxiliary network.
    n_kernels_max : int
        Maximum number of kernels in the auxiliary network.
    output_channels : int
        Number of output channels in the segmentation model.
    initialiser : Callable
        Initialiser to be used for the weights of the convolutional layers. A
        function from `torch.nn.init`.
    optimiser : Callable[..., torch.optim.Optimizer]
        Optimiser constructor to be used for training from `torch.optim`.
    optimiser_kwargs : dict
        Keyword arguments to be passed to the optimiser constructor.
    lr_scheduler : Callable[..., torch.optim.lr_scheduler._LRScheduler]
        Learning rate scheduler constructor to be used for training from
        `torch.optim.lr_scheduler`.
    lr_scheduler_kwargs : dict
        Keyword arguments to be passed to the lr_scheduler constructor.
    loss : Callable[[], nn.Module]
        Loss function to be used for training. Default is `ConfidNetMSELoss`.
    hidden_conv_dims : list[int]
        Number of kernels in each hidden convolutional layer. Length of the list
        determines the number of hidden layers.
    activation : Callable[[], nn.Module]
        Activation function to be used in the hidden layers. Default is `nn.LeakyReLU`.
    last_activation : Callable[[], nn.Module]
        Activation function to be used in the final layer. Default is `nn.Sigmoid`.
    """

    @auto_match_config(prefixes=["confidnet", "unet"])
    def __init__(
        self,
        unet: UNet,
        n_kernels_init: int,
        n_kernels_max: int,
        output_channels: int,
        initialiser: Callable,
        optimiser: Callable[..., torch.optim.Optimizer],
        optimiser_kwargs: dict,
        lr_scheduler: Callable[..., torch.optim.lr_scheduler._LRScheduler],
        lr_scheduler_kwargs: dict,
        loss: Callable[[], nn.Module] = ConfidNetMSELoss,
        hidden_conv_dims: list[int] = [128, 128, 64, 64],
        activation: Callable[[], nn.Module] = nn.LeakyReLU,
        last_activation: Callable[[], nn.Module] = nn.Sigmoid,
    ):
        super().__init__()

        input_dim = _calc_n_kernels(n_kernels_init, 1, n_kernels_max)
        dims = [input_dim] + hidden_conv_dims + [output_channels]

        self.activation = activation()
        self.last_activation = last_activation()
        self.conv_activations = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv3d(
                        in_dim,
                        out_dim,
                        kernel_size=3 if out_dim != output_channels else 1,
                        padding="same",
                    ),
                    (
                        self.activation
                        if out_dim != output_channels
                        else self.last_activation
                    ),
                )
                for in_dim, out_dim in zip(dims, dims[1:])
            ]
        )

        self.loss = loss()

        # Stored to be used by Lightning module
        self.optimiser = optimiser(self.parameters(), **optimiser_kwargs)
        self.lr_scheduler = lr_scheduler(self.optimiser, **lr_scheduler_kwargs)

        # Initialise weights before setting U-Net to avoid overwriting
        self.initialiser = initialiser
        self.apply(self._init_weights)

        self.unet = unet
        self.encoder = UNetConfidNetEncoder(unet)
        # Freeze U-Net parameters to only train the ConfidNet
        for param in self.unet.parameters():
            param.requires_grad = False

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Conv3d):
            self.initialiser(module.weight)

    def forward(self, x: torch.Tensor, logits: bool = False):
        """
        Return (confidence, segmentation), logits only apply to segmentation
        """
        encoder_out, seg_out = self.encoder(x, logits=logits)
        return tz.pipe(
            encoder_out,
            *self.conv_activations,  # pass through all convolutions and activations
            lambda confidence: (confidence, seg_out),
        )
