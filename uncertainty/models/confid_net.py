from typing import Callable, override
from torch import nn
import torch

from uncertainty.config import Configuration, configuration
import toolz as tz

from .unet import UNet, _concat_with_skip, _calc_n_kernels


class UNetConfidNetEncoder(nn.Module):
    """
    Convert a U-Net into a ConfidNet encoder by removing the last decoder level
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
        for level, skip in zip(self.unet.decoder.levels[:-1], reversed(skips)):  # type: ignore
            x = level(x, skip)
        return tz.pipe(
            x,
            self.unet.decoder.levels[-1].up,
            _concat_with_skip(skip=skips[0]),
            self.last_activation if logits else tz.identity,
        )

    def forward(self, x: torch.Tensor, logits: bool = False) -> torch.Tensor:
        encoded, skip = self.unet.encoder(x)
        return self._decoder_forward(encoded, skip, logits)

    def last_activation(self, x: torch.Tensor) -> torch.Tensor:
        return self.unet.decoder.last_activation(x)


class UNetConfidNet(nn.Module):
    def __init__(
        self,
        unet: UNet,
        config: Configuration,
        hidden_conv_dims: list[int] = [256, 128, 64, 64],
        activation: Callable[[], nn.Module] = configuration()["activation"],
        last_activation: Callable[[], nn.Module] = nn.Sigmoid,
    ):
        """
        n_convs: excludes final convolution
        conv_dims: dimension
        """
        super().__init__()
        self.unet = unet
        self.encoder = UNetConfidNetEncoder(unet)
        input_dim = _calc_n_kernels(
            config["n_kernels_init"], 1, config["n_kernels_max"]
        )
        output_dim = config["output_channel"]

        dims = [input_dim] + hidden_conv_dims + [output_dim]
        self.activation = activation()
        self.last_activation = last_activation()
        self.conv_activations = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv3d(in_dim, out_dim, kernel_size=1, padding="same"),
                    self.activation if out_dim != output_dim else self.last_activation,
                )
                for in_dim, out_dim in zip(dims, dims[1:])
            ]
        )

        # Freeze U-Net parameters to only train the ConfidNet
        for param in self.unet.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor):
        return tz.pipe(
            x,
            self.encoder,
            *self.conv_activations,  # pass through all convolutions and activations
        )
