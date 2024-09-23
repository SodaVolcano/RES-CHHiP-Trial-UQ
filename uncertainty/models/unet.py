"""
Define a U-Net model

References: 
    https://github.com/christianversloot/machine-learning-articles/blob/main/how-to-build-a-u-net-for-image-segmentation-with-tensorflow-and-keras.md
    https://github.com/milesial/Pytorch-UNet/tree/master
"""

from torch import nn
import torch
from .unet_modules import Encoder, Decoder
from ..config import Configuration


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
