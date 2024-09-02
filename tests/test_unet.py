from torch import nn

from .context import uncertainty
import torch

ConvLayer = uncertainty.models.unet.ConvLayer
ConvBlock = uncertainty.models.unet.ConvBlock
DownConvBlock = uncertainty.models.unet.DownConvBlock
Encoder = uncertainty.models.unet.Encoder
Decoder = uncertainty.models.unet.Decoder
UpConvBlock = uncertainty.models.unet.UpConvBlock
UNet = uncertainty.models.unet.UNet
MCDropoutUNet = uncertainty.models.unet.MCDropoutUNet


class TestConvLayer:
    # Initializes ConvLayer with valid parameters
    def test_initializes_with_valid_parameters(self):
        in_channels = 3
        out_channels = 16
        kernel_size = 3
        use_instance_norm = True
        activation = nn.ReLU
        dropout_rate = 0.5
        inorm_epsilon = 1e-5
        inorm_momentum = 0.1

        conv_layer = ConvLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            use_instance_norm=use_instance_norm,
            activation=activation,
            dropout_rate=dropout_rate,
            inorm_epsilon=inorm_epsilon,
            inorm_momentum=inorm_momentum,
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
            "n_kernels_max": 256,
            "kernel_size": 3,
            "use_instance_norm": True,
            "activation": nn.ReLU,
            "dropout_rate": 0.5,
            "instance_norm_epsilon": 1e-5,
            "instance_norm_decay": 0.1,
        }
        conv_block = ConvBlock(
            in_channels=1,
            out_channels=64,
            n_convolutions=2,
            config=config,  # type: ignore
        )
        assert len(conv_block.layers) == 2
        for layer in conv_block.layers:
            assert isinstance(layer, ConvLayer)

    # Forward pass processes input tensor correctly through all convolution layers
    def test_forward_pass_processes_input_tensor_correctly(self, mocker):
        config = {
            "kernel_size": 3,
            "use_instance_norm": True,
            "activation": nn.ReLU,
            "dropout_rate": 0.5,
            "instance_norm_epsilon": 1e-5,
            "instance_norm_decay": 0.1,
        }
        conv_block = ConvBlock(
            in_channels=1,
            out_channels=64,
            n_convolutions=2,
            config=config,  # type: ignore
        )
        input_tensor = torch.randn(1, 1, 32, 32, 32)
        output = conv_block(input_tensor)
        assert output.size() == torch.Size([1, 64, 32, 32, 32])


class TestDownConvBlock:
    # Initializes DownConvBlock with valid level and configuration
    def test_initializes_with_valid_level_and_config(self):
        config = {
            "n_kernels_init": 8,
            "n_convolutions_per_block": 2,
            "kernel_size": 3,
            "use_instance_norm": True,
            "activation": nn.ReLU,
            "dropout_rate": 0.1,
            "instance_norm_epsilon": 1e-5,
            "instance_norm_decay": 0.1,
            "n_kernels_max": 256,
        }
        level = 1
        down_conv_block = DownConvBlock(level, config)  # type: ignore
        assert isinstance(down_conv_block, DownConvBlock)
        assert isinstance(down_conv_block.layer, nn.Sequential)

    # Handles typical 3D tensor inputs without errors
    def test_handles_typical_3d_inputs_without_errors(self):
        config = {
            "n_kernels_max": 256,
            "n_kernels_init": 8,
            "n_convolutions_per_block": 1,
            "kernel_size": 3,
            "use_instance_norm": True,
            "activation": nn.ReLU,
            "dropout_rate": 0.1,
            "instance_norm_epsilon": 1e-5,
            "instance_norm_decay": 0.1,
        }
        level = 1
        down_conv_block = DownConvBlock(level, config)  # type: ignore
        input_tensor = torch.randn(1, 8, 32, 32, 32)  # Example 3D tensor input
        output = down_conv_block(input_tensor)
        assert output.shape == torch.Size([1, 16, 16, 16, 16])  # Expected output shape


class TestEncoder:
    # Encoder initializes with valid configuration
    def test_encoder_initializes_with_valid_configuration(self):
        config = {
            "n_kernels_max": 256,
            "kernel_size": 3,
            "use_instance_norm": True,
            "activation": nn.ReLU,
            "dropout_rate": 0.5,
            "instance_norm_decay": 0.9,
            "instance_norm_epsilon": 1e-5,
            "n_kernels_init": 64,
            "n_convolutions_per_block": 2,
            "n_levels": 2,
            "input_channel": 1,
            "final_layer_activation": nn.Sigmoid,
        }

        encoder = Encoder(config)  # type: ignore
        assert isinstance(encoder, Encoder)

    # Encoder processes input tensor and returns expected output and skip connections
    def test_encoder_processes_input_and_returns_output(self, mocker):
        config = {
            "n_kernels_max": 256,
            "kernel_size": 3,
            "use_instance_norm": True,
            "activation": nn.ReLU,
            "dropout_rate": 0.5,
            "instance_norm_decay": 0.9,
            "instance_norm_epsilon": 1e-5,
            "n_kernels_init": 8,
            "n_convolutions_per_block": 1,
            "n_levels": 3,
            "input_channel": 1,
            "final_layer_activation": nn.Sigmoid,
        }
        # Create Encoder instance
        encoder = Encoder(config)  # type: ignore

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
            "n_kernels_max": 256,
            "n_kernels_init": 16,
            "kernel_size": 3,
            "n_convolutions_per_block": 2,
            "use_instance_norm": True,
            "activation": nn.ReLU,
            "dropout_rate": 0.1,
            "instance_norm_epsilon": 1e-5,
            "instance_norm_decay": 0.1,
        }
        level = 1
        block = UpConvBlock(level, config)  # type: ignore
        # 64 because it's from previous layer
        x = torch.randn(1, 64, 8, 8, 8)
        skip = torch.randn(1, 32, 16, 16, 16)
        output = block(x, skip)
        assert output.shape == (1, 32, 16, 16, 16)


class TestDecoder:
    # Decoder initializes correctly with valid configuration
    def test_decoder_initializes_correctly(self):
        config = {
            "n_kernels_max": 256,
            "n_levels": 3,
            "n_kernels_init": 16,
            "n_kernels_last": 1,
            "kernel_size": 3,
            "n_convolutions_per_block": 2,
            "final_layer_activation": nn.ReLU,
            "use_instance_norm": True,
            "activation": nn.ReLU,
            "dropout_rate": 0.1,
            "instance_norm_epsilon": 1e-5,
            "instance_norm_decay": 0.1,
        }
        decoder = Decoder(config, deep_supervision=False)  # type: ignore
        # encoder don't have first level
        assert len(decoder.levels) == config["n_levels"] - 1
        assert isinstance(decoder.last_conv, nn.Conv3d)
        assert isinstance(decoder.last_activation, nn.ReLU)

        assert isinstance(decoder.last_activation, nn.ReLU)

    # Forward pass without deep supervision returns correct tensor shape
    def test_forward_pass_without_deep_supervision_returns_correct_tensor_shape(self):
        config = {
            "n_kernels_max": 256,
            "n_levels": 3,
            "n_kernels_init": 16,
            "n_kernels_last": 1,
            "kernel_size": 3,
            "n_convolutions_per_block": 2,
            "final_layer_activation": nn.ReLU,
            "use_instance_norm": True,
            "activation": nn.ReLU,
            "dropout_rate": 0.1,
            "instance_norm_epsilon": 1e-5,
            "instance_norm_decay": 0.1,
        }
        decoder = Decoder(config, deep_supervision=False)  # type: ignore
        x = torch.randn(1, 64, 32, 32, 32)  # from last layer
        skips = [torch.randn(1, 16, 32, 32, 32), torch.randn(1, 32, 16, 16, 16)]
        output = decoder(x, skips, logits=False)
        assert output.shape == (1, config["n_kernels_last"], 32, 32, 32)

    # Forward pass with deep supervision returns list of correct tensor shapes
    def test_forward_deep_supervision_tensor_shapes(self):
        config = {
            "n_kernels_max": 256,
            "n_levels": 3,
            "n_kernels_init": 16,
            "n_kernels_last": 1,
            "kernel_size": 3,
            "n_convolutions_per_block": 2,
            "final_layer_activation": nn.ReLU,
            "use_instance_norm": True,
            "activation": nn.ReLU,
            "dropout_rate": 0.1,
            "instance_norm_epsilon": 1e-5,
            "instance_norm_decay": 0.1,
        }
        decoder = Decoder(config, deep_supervision=True)  # type: ignore
        x = torch.randn(1, 64, 32, 32, 32)
        skips = [torch.randn(1, 16, 32, 32, 32), torch.randn(1, 32, 16, 16, 16)]
        outputs = decoder(x, skips, logits=False)
        assert len(outputs) == config["n_levels"] - 2
        expected_shapes = [(1, 1, 32, 32, 32)]
        for output, shape in zip(outputs, expected_shapes):
            assert output.shape == torch.Size(shape)


class TestUNet:
    # Initializes UNet with valid configuration and processes input tensor correctly
    def test_unet_initialization_and_forward_pass(self, mocker):
        config = {
            "n_kernels_max": 256,
            "n_kernels_init": 4,
            "kernel_size": 3,
            "n_convolutions_per_block": 2,
            "use_instance_norm": True,
            "activation": nn.ReLU,
            "dropout_rate": 0.1,
            "instance_norm_epsilon": 1e-5,
            "instance_norm_decay": 0.1,
            "n_levels": 4,
            "input_channel": 1,
            "n_kernels_last": 1,
            "final_layer_activation": nn.Sigmoid,
            "input_height": 128,
            "input_width": 128,
            "input_depth": 128,
        }

        # Initialize UNet
        model = UNet(config)  # type: ignore

        # Create a dummy input tensor
        input_tensor = torch.randn(
            1,
            config["input_channel"],
            config["input_height"],
            config["input_width"],
            config["input_depth"],
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
            "n_kernels_max": 256,
            "n_kernels_init": 8,
            "n_convolutions_per_block": 2,
            "kernel_size": 3,
            "use_instance_norm": True,
            "activation": nn.ReLU,
            "dropout_rate": 0.1,
            "instance_norm_epsilon": 1e-5,
            "instance_norm_decay": 0.1,
            "n_levels": 4,
            "input_channel": 1,
            "n_kernels_last": 1,
            "final_layer_activation": nn.Sigmoid,
            "input_height": 128,
            "input_width": 128,
            "input_depth": 128,
        }

        # Initialize UNet
        model = UNet(config, deep_supervision=True)  # type: ignore

        # Create a dummy input tensor
        input_tensor = torch.randn(
            1,
            config["input_channel"],
            config["input_height"],
            config["input_width"],
            config["input_depth"],
        )

        # Forward pass with deep supervision
        output = model(input_tensor)

        # Assertions
        assert output is not None
        assert isinstance(output, list)
        assert all(isinstance(out, torch.Tensor) for out in output)

    def test_prime_input_shape(self, mocker):
        config = {
            "n_kernels_max": 256,
            "n_kernels_init": 8,
            "n_convolutions_per_block": 2,
            "kernel_size": 3,
            "use_instance_norm": True,
            "activation": nn.ReLU,
            "dropout_rate": 0.1,
            "instance_norm_epsilon": 1e-5,
            "instance_norm_decay": 0.1,
            "n_levels": 4,
            "input_channel": 1,
            "n_kernels_last": 1,
            "final_layer_activation": nn.Sigmoid,
            "input_height": 97,
            "input_width": 89,
            "input_depth": 79,
        }

        model = UNet(config, deep_supervision=False)  # type: ignore
        input_ = torch.randn(1, 1, 97, 89, 79)
        output = model(input_)
        assert output.shape == (1, 1, 97, 89, 79)


class TestMCDropoutUNet:
    # Initialize MCDropoutUNet with a valid Configuration and ensure it wraps a new UNet instance
    def test_initialization_with_valid_config(self):
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

        unet_model = UNet(config, deep_supervision=False)  # type: ignore

        model = MCDropoutUNet(model=unet_model)  # type: ignore
        input_ = torch.randn(1, 1, 128, 128, 128)
        output = model(input_)

        assert model.model is unet_model
        assert output.shape == (1, 1, 128, 128, 128)

    # Test that passing the same input produces different outputs
    def test_same_input_different_output(self):
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
            "n_kernels_last": 1,
            "final_layer_activation": nn.Sigmoid,
        }
        unet = UNet(config=config, deep_supervision=False)  # type: ignore
        unet2 = UNet(config=config, deep_supervision=False)  # type: ignore
        unet2.eval()
        # Create an instance of MCDropoutUNet
        model = MCDropoutUNet(unet)  # type: ignore
        model.eval()

        # Prepare input tensor
        x = torch.randn(1, 3, 100, 100, 100)

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
