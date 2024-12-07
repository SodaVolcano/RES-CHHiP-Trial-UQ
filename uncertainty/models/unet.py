"""
U-shaped architecture for image segmentation.

References: 
    https://github.com/christianversloot/machine-learning-articles/blob/main/how-to-build-a-u-net-for-image-segmentation-with-tensorflow-and-keras.md
    https://github.com/milesial/Pytorch-UNet/tree/master
"""

from typing import Callable

import torch
from torch import nn

from ..config import auto_match_config
from ..metrics.loss import DeepSupervisionLoss, DiceBCELoss
from .unet_modules import Decoder, Encoder


class UNet(nn.Module):
    """
    A U-shaped architecture for image segmentation

    **Warning**: not meant to be used directly, wrap in a Lightning module
    from `uncertainty.training.LightningModel`.

    Consists of N resolution levels and divided into the Encoder
    and Decoder branches. The Encoder downsamples and convolves
    the input image in each level to extract features, while the
    Decoder upsamples and concatenates output from the previous
    level with the corresponding output from the Encoder of the
    same level and then apply convolutions. A final 1x1 convolution
    produces the final output.

    If deep supervision is enabled, then for a U-Net with n levels, the
    loss is calculated for each level and summed as
        L = w1 * L1 + w2 * L2 + ... + wn * Ln
    Where the weights halve for each level and are normalised to sum to 1.
    Output from the two levels in the lowest resolution are not used.
    SEE https://arxiv.org/abs/1809.10486

    Parameters
    ----------
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
    deep_supervision : bool
        Whether to use deep supervision. If True, the loss is calculated
        for each level and summed with weights halving for each level.
    loss : Callable[[], nn.Module]
        Loss function to be used for training. Default is `DiceBCELoss`.
    """

    @auto_match_config(prefixes=["unet"])
    def __init__(
        self,
        initialiser: Callable,
        optimiser: Callable[..., torch.optim.Optimizer],
        optimiser_kwargs: dict,
        lr_scheduler: Callable[..., torch.optim.lr_scheduler._LRScheduler],
        lr_scheduler_kwargs: dict,
        deep_supervision: bool,
        loss: Callable[[], nn.Module] = DiceBCELoss,
        **kwargs,
    ):
        super().__init__()
        self.encoder = Encoder(**kwargs)
        self.decoder = Decoder(**kwargs)

        self.initialiser = initialiser
        self.apply(self._init_weights)

        self.deep_supervision = deep_supervision
        self.loss = loss() if not deep_supervision else DeepSupervisionLoss(loss())

        # Stored to be used by Lightning module
        self.optimiser = optimiser(self.parameters(), **optimiser_kwargs)
        self.lr_scheduler = lr_scheduler(self.optimiser, **lr_scheduler_kwargs)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Conv3d):
            self.initialiser(module.weight)

    def forward(self, x: torch.Tensor, logits: bool = False) -> torch.Tensor:
        encoded, skips = self.encoder(x)
        return self.decoder(encoded, skips, logits)

    def last_activation(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder.last_activation(x)
