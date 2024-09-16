"""
ConfidNet, auxiliary network for predicting true class probability

Reference:
    https://github.com/valeoai/ConfidNet
    Addressing Failure Prediction by Learning Model Confidence, Corbiere et al.
"""

from .context import uncertainty
from uncertainty.models.confid_net import UNetConfidNetEncoder
import torch
from torch import nn

UNet = uncertainty.models.UNet
UNetConfidNetEncoder = uncertainty.models.confid_net.UNetConfidNetEncoder
UNetConfidNet = uncertainty.models.confid_net.UNetConfidNet


class TestUNetConfidNetEncoder:
    # Initialize UNetConfidNetEncoder with a valid UNet instance
    def test_forward_pass(self, mocker):
        config = {
            "n_kernels_max": 256,
            "n_kernels_init": 8,
            "n_convolutions_per_block": 2,
            "kernel_size": 3,
            "use_instance_norm": True,
            "activation": nn.ReLU,
            "dropout_rate": 0.8,
            "instance_norm_epsilon": 1e-5,
            "instance_norm_decay": 0.1,
            "n_levels": 2,
            "input_channel": 1,
            "n_kernels_last": 1,
            "final_layer_activation": nn.Sigmoid,
        }
        unet = UNet(config)  # type: ignore
        encoder = UNetConfidNetEncoder(unet)
        test_tensor = torch.randn(2, 1, 64, 64, 64)

        result = encoder(test_tensor)
        assert result.shape == (2, config["n_kernels_init"] * 2, 64, 64, 64)


class TestUNetConfidNet:
    # Initializes UNetConfidNet with default parameters and verifies structure
    def test_initializes_with_default_parameters(self, mocker):
        config = {
            "n_kernels_max": 256,
            "n_kernels_init": 8,
            "n_convolutions_per_block": 2,
            "kernel_size": 3,
            "use_instance_norm": True,
            "activation": nn.ReLU,
            "dropout_rate": 0.8,
            "instance_norm_epsilon": 1e-5,
            "instance_norm_decay": 0.1,
            "n_levels": 2,
            "input_channel": 3,
            "output_channel": 1,
            "n_kernels_last": 1,
            "final_layer_activation": nn.Sigmoid,
        }
        unet = UNet(config)  # type: ignore
        model = UNetConfidNet(unet, config)  # type: ignore
        test_tensor = torch.randn(2, 3, 64, 64, 64)

        result = model(test_tensor)

        # all unet params are frozen
        assert isinstance(model.encoder, nn.Module)
        assert all(p.requires_grad for p in model.unet.parameters()) == False
        assert len(model.conv_activations) == 5  # 4 hidden + 1 output
        assert result.shape == (2, 1, 64, 64, 64)
