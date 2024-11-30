import torch
from torch import nn

from ..context import models

ConvLayer = models.unet_modules.ConvLayer
ConvBlock = models.unet_modules.ConvBlock
DownConvBlock = models.unet_modules.DownConvBlock
Encoder = models.unet.Encoder
Decoder = models.unet.Decoder
UpConvBlock = models.unet_modules.UpConvBlock
UNet = models.UNet


class TestConvLayer:
    # Initializes ConvLayer with valid parameters
    def test_initializes_with_valid_parameters(self):
        in_channels = 3
        out_channels = 16

        conv_layer = ConvLayer(
            in_channels=3,
            out_channels=16,
            kernel_size=3,
            use_instance_norm=True,
            activation=nn.ReLU,
            dropout_rate=0.5,
            instance_norm_epsilon=1e-5,
            instance_norm_momentum=0.1,
        )
        input_ = torch.randn(1, in_channels, 64, 64, 64)
        output = conv_layer(input_)

        assert output.shape == (1, out_channels, 64, 64, 64)
        assert isinstance(conv_layer, ConvLayer)
        assert len(conv_layer.layers) == 4
        assert isinstance(conv_layer.layers[0], nn.Conv3d)
        assert isinstance(conv_layer.layers[1], nn.InstanceNorm3d)
        assert isinstance(conv_layer.layers[2], nn.ReLU)
        assert isinstance(conv_layer.layers[3], nn.Dropout)


class TestConvBlock:
    # Initializes ConvBlock with valid configuration and parameters
    def test_initializes_with_valid_configuration(self):
        config = {
            "unet__n_kernels_max": 256,
            "unet__kernel_size": 3,
            "unet__use_instance_norm": True,
            "unet__activation": nn.ReLU,
            "unet__dropout_rate": 0.5,
            "unet__instance_norm_epsilon": 1e-5,
            "unet__instance_norm_decay": 0.1,
            "unet__instance_norm_momentum": 0.1,
        }
        conv_block = ConvBlock(
            level=1,
            n_levels=3,
            in_channels=1,
            out_channels=64,
            n_convolutions=2,
            **config,
        )
        assert len(conv_block.conv_layers) == 2
        for layer in conv_block.conv_layers:
            assert isinstance(layer, ConvLayer)

    # Forward pass processes input tensor correctly through all convolution layers
    def test_forward_pass_processes_input_tensor_correctly(self, mocker):
        config = {
            "unet__kernel_size": 3,
            "unet__use_instance_norm": True,
            "unet__activation": nn.ReLU,
            "unet__dropout_rate": 0.5,
            "unet__instance_norm_epsilon": 1e-5,
            "unet__instance_norm_decay": 0.1,
            "unet__instance_norm_momentum": 0.1,
        }
        conv_block = ConvBlock(
            level=1,
            n_levels=3,
            in_channels=1,
            out_channels=64,
            n_convolutions=2,
            **config,
        )
        input_tensor = torch.randn(1, 1, 32, 32, 32)
        output = conv_block(input_tensor)
        assert output.size() == torch.Size([1, 64, 32, 32, 32])


class TestDownConvBlock:
    # Initializes DownConvBlock with valid level and configuration
    def test_initializes_with_valid_level_and_config(self):
        config = {
            "unet__n_kernels_init": 8,
            "unet__n_convolutions_per_block": 2,
            "unet__kernel_size": 3,
            "unet__use_instance_norm": True,
            "unet__activation": nn.ReLU,
            "unet__dropout_rate": 0.1,
            "unet__n_levels": 3,
            "unet__instance_norm_epsilon": 1e-5,
            "unet__instance_norm_decay": 0.1,
            "unet__instance_norm_momentum": 0.1,
            "unet__n_kernels_max": 256,
        }
        level = 1
        down_conv_block = DownConvBlock(level, **config)
        assert isinstance(down_conv_block, DownConvBlock)
        assert isinstance(down_conv_block.conv_blocks, nn.Sequential)

    # Handles typical 3D tensor inputs without errors
    def test_handles_typical_3d_inputs_without_errors(self):
        config = {
            "unet__n_kernels_max": 256,
            "unet__n_kernels_init": 8,
            "unet__n_convolutions_per_block": 1,
            "unet__kernel_size": 3,
            "unet__use_instance_norm": True,
            "unet__activation": nn.ReLU,
            "unet__n_levels": 3,
            "unet__dropout_rate": 0.1,
            "unet__instance_norm_epsilon": 1e-5,
            "unet__instance_norm_decay": 0.1,
            "unet__instance_norm_momentum": 0.1,
        }
        level = 1
        down_conv_block = DownConvBlock(level, **config)
        input_tensor = torch.randn(1, 8, 32, 32, 32)  # Example 3D tensor input
        output = down_conv_block(input_tensor)
        assert output.shape == torch.Size([1, 16, 16, 16, 16])  # Expected output shape


class TestEncoder:
    # Encoder initializes with valid configuration
    def test_encoder_initializes_with_valid_configuration(self):
        config = {
            "n_kernels_max": 256,
            "unet__kernel_size": 3,
            "unet__use_instance_norm": True,
            "unet__activation": nn.ReLU,
            "unet__dropout_rate": 0.5,
            "unet__instance_norm_decay": 0.9,
            "unet__instance_norm_epsilon": 1e-5,
            "unet__instance_norm_momentum": 0.1,
            "unet__n_kernels_init": 64,
            "unet__n_convolutions_per_block": 2,
            "unet__n_levels": 2,
            "unet__input_channels": 1,
            "unet__final_layer_activation": nn.Sigmoid,
        }

        encoder = Encoder(**config)
        assert isinstance(encoder, Encoder)

    # Encoder processes input tensor and returns expected output and skip connections
    def test_encoder_processes_input_and_returns_output(self, mocker):
        config = {
            "unet__n_kernels_max": 256,
            "unet__kernel_size": 3,
            "unet__use_instance_norm": True,
            "unet__activation": nn.ReLU,
            "unet__dropout_rate": 0.5,
            "unet__instance_norm_decay": 0.9,
            "unet__instance_norm_epsilon": 1e-5,
            "unet__instance_norm_momentum": 0.1,
            "unet__n_kernels_init": 8,
            "unet__n_convolutions_per_block": 1,
            "unet__n_levels": 3,
            "unet__input_channels": 1,
            "unet__final_layer_activation": nn.Sigmoid,
        }
        # Create Encoder instance
        encoder = Encoder(**config)

        # Prepare input tensor
        input_tensor = torch.randn(1, 1, 128, 128, 128)

        # Call forward method of the Encoder
        output, skip_connections = encoder.forward(input_tensor)

        # Assertions
        assert isinstance(output, torch.Tensor)
        assert isinstance(skip_connections, list)
        assert all(
            isinstance(skip_connection, torch.Tensor)
            for skip_connection in skip_connections
        )


class TestUpConvBlock:
    # Correct upsampling and concatenation with skip connections
    def test_correct_upsampling_and_concatenation(self):
        config = {
            "unet__n_kernels_max": 256,
            "unet__n_kernels_init": 16,
            "unet__kernel_size": 3,
            "unet__n_convolutions_per_block": 2,
            "unet__use_instance_norm": True,
            "unet__activation": nn.ReLU,
            "unet__n_levels": 3,
            "unet__dropout_rate": 0.1,
            "unet__instance_norm_epsilon": 1e-5,
            "unet__instance_norm_decay": 0.1,
            "unet__instance_norm_momentum": 0.1,
        }
        level = 1
        block = UpConvBlock(level, **config)
        # 64 because it's from previous layer
        x = torch.randn(1, 64, 8, 8, 8)
        skip = torch.randn(1, 32, 16, 16, 16)
        output = block(x, skip)
        assert output.shape == (1, 32, 16, 16, 16)


class TestDecoder:
    # Decoder initializes correctly with valid configuration
    def test_decoder_initializes_correctly(self):
        config = {
            "unet__n_kernels_max": 256,
            "unet__n_levels": 3,
            "unet__n_kernels_init": 16,
            "unet__output_channels": 1,
            "unet__kernel_size": 3,
            "unet__n_convolutions_per_block": 2,
            "unet__final_layer_activation": nn.ReLU,
            "unet__use_instance_norm": True,
            "unet__activation": nn.ReLU,
            "unet__dropout_rate": 0.1,
            "unet__instance_norm_epsilon": 1e-5,
            "unet__instance_norm_decay": 0.1,
            "unet__instance_norm_momentum": 0.1,
        }
        decoder = Decoder(**config, deep_supervision=False)
        # encoder don't have first level
        assert len(decoder.levels) == config["unet__n_levels"] - 1
        assert isinstance(decoder.last_conv, nn.Conv3d)
        assert isinstance(decoder.last_activation, nn.ReLU)

        assert isinstance(decoder.last_activation, nn.ReLU)

    # Forward pass without deep supervision returns correct tensor shape
    def test_forward_pass_without_deep_supervision_returns_correct_tensor_shape(self):
        config = {
            "unet__n_kernels_max": 256,
            "unet__n_levels": 3,
            "unet__n_kernels_init": 16,
            "unet__output_channels": 1,
            "unet__kernel_size": 3,
            "unet__n_convolutions_per_block": 2,
            "unet__final_layer_activation": nn.ReLU,
            "unet__use_instance_norm": True,
            "unet__activation": nn.ReLU,
            "unet__dropout_rate": 0.1,
            "unet__instance_norm_epsilon": 1e-5,
            "unet__instance_norm_decay": 0.1,
            "unet__instance_norm_momentum": 0.1,
        }
        decoder = Decoder(**config, deep_supervision=False)
        x = torch.randn(1, 64, 32, 32, 32)  # from last layer
        skips = [torch.randn(1, 16, 32, 32, 32), torch.randn(1, 32, 16, 16, 16)]
        output = decoder(x, skips, logits=False)
        assert output.shape == (1, config["unet__output_channels"], 32, 32, 32)

    # Forward pass with deep supervision returns list of correct tensor shapes
    def test_forward_deep_supervision_tensor_shapes(self):
        config = {
            "unet__n_kernels_max": 256,
            "unet__n_levels": 3,
            "unet__n_kernels_init": 16,
            "unet__output_channels": 1,
            "unet__kernel_size": 3,
            "unet__n_convolutions_per_block": 2,
            "unet__final_layer_activation": nn.ReLU,
            "unet__use_instance_norm": True,
            "unet__activation": nn.ReLU,
            "unet__dropout_rate": 0.1,
            "unet__instance_norm_epsilon": 1e-5,
            "unet__instance_norm_decay": 0.1,
            "unet__instance_norm_momentum": 0.1,
        }
        decoder = Decoder(**config, deep_supervision=True)
        x = torch.randn(1, 64, 32, 32, 32)
        skips = [torch.randn(1, 16, 32, 32, 32), torch.randn(1, 32, 16, 16, 16)]
        outputs = decoder(x, skips, logits=False)
        assert len(outputs) == config["unet__n_levels"] - 2
        expected_shapes = [(1, 1, 32, 32, 32)]
        for output, shape in zip(outputs, expected_shapes):
            assert output.shape == torch.Size(shape)


class TestUNet:
    # Initializes UNet with valid configuration and processes input tensor correctly
    def test_unet_initialization_and_forward_pass(self, mocker):
        config = {
            "unet__n_kernels_max": 256,
            "unet__n_kernels_init": 4,
            "unet__kernel_size": 3,
            "unet__n_convolutions_per_block": 2,
            "unet__use_instance_norm": True,
            "unet__activation": nn.ReLU,
            "unet__dropout_rate": 0.1,
            "unet__instance_norm_epsilon": 1e-5,
            "unet__instance_norm_momentum": 0.1,
            "unet__instance_norm_decay": 0.1,
            "unet__n_levels": 4,
            "unet__input_channels": 1,
            "unet__output_channels": 1,
            "unet__final_layer_activation": nn.Sigmoid,
            "unet__input_height": 128,
            "unet__input_width": 128,
            "unet__input_depth": 128,
        }

        # Initialize UNet
        model = UNet(**config)

        # Create a dummy input tensor
        input_tensor = torch.randn(
            1,
            config["unet__input_channels"],
            config["unet__input_height"],
            config["unet__input_width"],
            config["unet__input_depth"],
        )

        # Forward pass
        outputs = model(input_tensor)

        # Assertions
        assert outputs is not None
        assert isinstance(outputs, list)
        for output, expected in zip(outputs, [64, 128]):
            assert output.shape == (1, 1, *(expected,) * 3)

    # Deep supervision mode in forward pass returns list of outputs from each level
    def test_deep_supervision_mode_returns_list_of_outputs(self, mocker):
        config = {
            "unet__n_kernels_max": 256,
            "unet__n_kernels_init": 8,
            "unet__n_convolutions_per_block": 2,
            "unet__kernel_size": 3,
            "unet__use_instance_norm": True,
            "unet__activation": nn.ReLU,
            "unet__dropout_rate": 0.1,
            "unet__instance_norm_epsilon": 1e-5,
            "unet__instance_norm_momentum": 0.1,
            "unet__instance_norm_decay": 0.1,
            "unet__n_levels": 4,
            "unet__input_channels": 1,
            "unet__output_channels": 1,
            "unet__final_layer_activation": nn.Sigmoid,
            "unet__input_height": 128,
            "unet__input_width": 128,
            "unet__input_depth": 128,
        }

        # Initialize UNet
        model = UNet(**config, deep_supervision=True)

        # Create a dummy input tensor
        input_tensor = torch.randn(
            1,
            config["unet__input_channels"],
            config["unet__input_height"],
            config["unet__input_width"],
            config["unet__input_depth"],
        )

        # Forward pass with deep supervision
        output = model(input_tensor)

        # Assertions
        assert output is not None
        assert isinstance(output, list)
        assert all(isinstance(out, torch.Tensor) for out in output)

    def test_prime_input_shape(self, mocker):
        config = {
            "unet__n_kernels_max": 256,
            "unet__n_kernels_init": 8,
            "unet__n_convolutions_per_block": 2,
            "unet__kernel_size": 3,
            "unet__use_instance_norm": True,
            "unet__activation": nn.ReLU,
            "unet__dropout_rate": 0.1,
            "unet__instance_norm_epsilon": 1e-5,
            "unet__instance_norm_momentum": 0.1,
            "unet__instance_norm_decay": 0.1,
            "unet__n_levels": 4,
            "unet__input_channels": 1,
            "unet__output_channels": 1,
            "unet__final_layer_activation": nn.Sigmoid,
            "unet__input_height": 97,
            "unet__input_width": 89,
            "unet__input_depth": 79,
        }

        model = UNet(**config, deep_supervision=False)
        input_ = torch.randn(1, 1, 97, 89, 79)
        output = model(input_)
        assert output.shape == (1, 1, 97, 89, 79)
