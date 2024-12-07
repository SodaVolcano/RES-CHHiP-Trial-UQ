import torch
from torch import nn

from ..context import models

UNet = models.UNet
UNetConfidNetEncoder = models.confidnet.UNetConfidNetEncoder
UNetConfidNet = models.UNetConfidNet

config = {
    "unet__n_kernels_max": 256,
    "unet__n_kernels_init": 8,
    "unet__n_convolutions_per_block": 2,
    "unet__kernel_size": 3,
    "unet__use_instance_norm": True,
    "unet__activation": nn.ReLU,
    "unet__dropout_rate": 0.8,
    "unet__instance_norm_epsilon": 1e-5,
    "unet__instance_norm_momentum": 0.1,
    "unet__instance_norm_decay": 0.1,
    "unet__n_levels": 3,
    "unet__output_channels": 1,
    "unet__final_layer_activation": nn.Sigmoid,
    "unet__deep_supervision": True,
    "unet__optimiser": torch.optim.SGD,
    "unet__optimiser_kwargs": {"momentum": 0.9},
    "unet__lr_scheduler": torch.optim.lr_scheduler.PolynomialLR,
    "unet__lr_scheduler_kwargs": {"total_iters": 750},
    "unet__initialiser": torch.nn.init.kaiming_normal_,
}


class TestUNetConfidNetEncoder:
    # Initialize UNetConfidNetEncoder with a valid UNet instance
    def test_forward_pass(self, mocker):

        unet = UNet(**(config | {"unet__input_channels": 1}))
        encoder = UNetConfidNetEncoder(unet)
        test_tensor = torch.randn(2, 1, 64, 64, 64)

        result = encoder(test_tensor, logits=False)
        assert len(result) == 2
        assert result[0].shape == (2, config["unet__n_kernels_init"] * 2, 64, 64, 64)
        assert result[1].shape == (2, 1, 64, 64, 64)


class TestUNetConfidNet:
    # Initializes UNetConfidNet with default parameters and verifies structure
    def test_initializes_with_default_parameters(self, mocker):
        unet = UNet(**(config | {"unet__input_channels": 3}))
        model = UNetConfidNet(unet, **config)
        test_tensor = torch.randn(2, 3, 64, 64, 64)

        result = model(test_tensor)

        # all unet params are frozen
        assert isinstance(model.encoder, nn.Module)
        assert all(p.requires_grad for p in model.unet.parameters()) == False
        assert len(model.conv_activations) == 5  # 4 hidden + 1 output
        assert result[0].shape == (2, 1, 64, 64, 64)
        assert result[1].shape == (2, 1, 64, 64, 64)
