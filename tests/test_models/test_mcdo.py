from torch import nn

from ..context import models
import torch

ConvLayer = models.unet_modules.ConvLayer
ConvBlock = models.unet_modules.ConvBlock
DownConvBlock = models.unet_modules.DownConvBlock
Encoder = models.unet.Encoder
Decoder = models.unet.Decoder
UpConvBlock = models.unet_modules.UpConvBlock
UNet = models.UNet
MCDropout = models.MCDropout

config = {
    "n_kernels_max": 256,
    "n_kernels_init": 4,
    "n_convolutions_per_block": 2,
    "kernel_size": 3,
    "use_instance_norm": True,
    "activation": nn.ReLU,
    "dropout_rate": 0.8,
    "instance_norm_epsilon": 1e-5,
    "instance_norm_decay": 0.1,
    "n_levels": 3,
    "input_channel": 1,
    "n_kernels_last": 1,
    "final_layer_activation": nn.Sigmoid,
}


class TestMCDropout:
    # Initialize MCDropoutUNet with a valid Configuration and ensure it wraps a new UNet instance
    def test_initialization_with_valid_config(self):
        unet_model = UNet(config, deep_supervision=False)  # type: ignore

        model = MCDropout(model=unet_model)  # type: ignore
        input_ = torch.randn(2, 1, 128, 128, 128)
        output = model(input_)

        assert model.model is unet_model
        assert output.shape == (2, 1, 128, 128, 128)

    # Test that passing the same input produces different outputs
    def test_same_input_different_output(self):
        unet = UNet(config=config, deep_supervision=False)  # type: ignore
        unet2 = UNet(config=config, deep_supervision=False)  # type: ignore
        unet2.eval()
        # Create an instance of MCDropoutUNet
        model = MCDropout(unet)  # type: ignore
        model.eval()

        # Prepare input tensor
        x = torch.randn(1, 1, 100, 100, 100)

        output_1 = model(x, logits=True)
        output_2 = model(x, logits=True)

        # check all Dropout layers are in train mode
        assert all(
            layer.training
            for layer in model.model.modules()
            if isinstance(layer, nn.Dropout)
        )
        # Check that the two outputs are different
        assert not torch.allclose(output_1, output_2)

        output_3 = unet2(x, logits=True)
        output_4 = unet2(x, logits=True)
        # Check that the two outputs are the same
        assert torch.allclose(output_3, output_4)
