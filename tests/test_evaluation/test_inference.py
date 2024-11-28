import numpy as np
import toolz as tz
import torch
from skimage.util import view_as_windows

from ..context import evaluation

_calc_pad_amount = evaluation.inference._calc_pad_amount
_pad_image = evaluation.inference._pad_image
_spline_window_1d = evaluation.inference._spline_window_1d
_spline_window_3d = evaluation.inference._spline_window_3d
_unpad_image = evaluation.inference._unpad_image
_reconstruct_image = evaluation.inference._reconstruct_image
sliding_inference = evaluation.sliding_inference


class Test_CalcPadAmount:

    # Calculate correct padding for standard patch sizes and subdivisions (e.g. 64,64,64 and 2,2,2)
    def test_standard_patch_sizes(self):
        patch_sizes = (64, 64, 64)
        subdivisions = (1, 2, 3)

        result = _calc_pad_amount(patch_sizes, subdivisions)

        assert result == [(0, 0), (32, 32), (43, 43)]


class Test_PadImage:

    # Pad 3D image with reflect mode using calculated padding amounts
    def test_pad_3d_image_reflect_mode(self):
        # Create sample 3D image with channels
        image = np.random.rand(2, 4, 6, 8)  # (C,D,H,W)
        patch_sizes = (2, 4, 4)
        subdivisions = (2, 2, 2)

        # Calculate expected padding
        pad_amounts = _calc_pad_amount(patch_sizes, subdivisions)
        expected_shape = (
            2,  # channels unchanged
            4 + sum(pad_amounts[0]),  # depth
            6 + sum(pad_amounts[1]),  # height
            8 + sum(pad_amounts[2]),  # width
        )

        # Pad image
        padded = _pad_image(image, patch_sizes, subdivisions)

        # Check output shape and type
        assert isinstance(padded, np.ndarray)
        assert padded.shape == expected_shape

        # Check original image is preserved in center
        assert np.array_equal(
            padded[
                :,
                pad_amounts[0][0] : pad_amounts[0][0] + 4,
                pad_amounts[1][0] : pad_amounts[1][0] + 6,
                pad_amounts[2][0] : pad_amounts[2][0] + 8,
            ],
            image,
        )


class Test_SplineWindow1d:

    # Window size of 8 returns correct spline window array shape and values
    def test_window_size_8_returns_correct_output(self):
        # Test with window size 8 and default power=2
        window_size = 8
        result = _spline_window_1d(window_size)

        # Check shape
        assert result.shape == (window_size,)

        # Expected values for window size 8 with power 2
        expected = np.array(
            [0.03125, 0.28125, 0.71875, 0.96875, 0.96875, 0.71875, 0.28125, 0.03125]
        )

        # Check values match expected within tolerance
        assert np.allclose(result, expected, rtol=1e-5)

    # Window size of 10 returns correct spline window array shape and values
    def test_window_size_10_power_3(self):
        # Test with window size 8
        window_size = 10
        result = _spline_window_1d(window_size, 3)

        assert result.shape == (window_size,)

        # Expected values for window size 8 with power 2
        expected = np.array(
            [0.004, 0.108, 0.5, 0.892, 0.996, 0.996, 0.892, 0.5, 0.108, 0.004]
        )

        # Check values match expected within tolerance
        assert np.allclose(result, expected, rtol=1e-5)

    # Window size of 1 should handle minimal case
    def test_window_size_1_minimal_case(self):
        # Test with minimal window size of 1
        window_size = 1
        result = _spline_window_1d(window_size, 3)

        # Check shape
        assert result.shape == (window_size,)

        # For window size 1, expect single value array
        assert len(result) == 1

        assert np.allclose(result, 4.0, rtol=1e-5)


class Test_SplineWindow3d:

    def test_window_size_334_power_2(self):
        window_size = (3, 3, 4)
        result = _spline_window_3d(window_size, 2)

        expected = np.array(
            [
                [
                    [0.03125, 0.21875, 0.21875, 0.03125],
                    [0.125, 0.875, 0.875, 0.125],
                    [0.03125, 0.21875, 0.21875, 0.03125],
                ],
                [
                    [0.125, 0.875, 0.875, 0.125],
                    [0.5, 3.5, 3.5, 0.5],
                    [0.125, 0.875, 0.875, 0.125],
                ],
                [
                    [0.03125, 0.21875, 0.21875, 0.03125],
                    [0.125, 0.875, 0.875, 0.125],
                    [0.03125, 0.21875, 0.21875, 0.03125],
                ],
            ]
        )

        assert np.allclose(result, expected, rtol=1e-5)


class Test_UnpadImage:

    # Correctly removes padding from all sides of a 4D image array (C,D,H,W)
    def test_removes_padding_correctly(self):
        # Create test input array with shape (C,D,H,W)
        input_array = np.random.rand(2, 10, 12, 8)
        patch_sizes = (4, 6, 4)
        subdivisions = (2, 2, 2)

        padded = _pad_image(input_array, patch_sizes, subdivisions)

        # Call function
        output = _unpad_image(padded, patch_sizes, subdivisions)

        # Check output shape matches expected
        assert output.shape == input_array.shape

        # Check values match original array with padding removed
        np.testing.assert_array_equal(output, input_array)


class Test_ReconstructImage:

    # Correctly reconstructs 3D image from patches using provided window and stride
    def test_reconstruct_3d_image(self):
        # Set up test data
        patch_size = (10, 7, 5)
        subdivisions = (2, 2, 2)

        img = (np.random.rand(1, 20, 15, 10) > 0.5).astype(float)
        window = _spline_window_3d(patch_size, 2)
        img_padded = _pad_image(img, patch_size, subdivisions)
        stride = tuple(
            p_size // subdiv for p_size, subdiv in zip(patch_size, subdivisions)
        )
        patches_ = view_as_windows(
            img_padded, (img_padded.shape[0], *patch_size), (1,) + stride  # type: ignore
        )
        patch_indices = [idx for idx in np.ndindex(patches_.shape[:-4])]
        patches = [patches_[idx] for idx in patch_indices]

        result = _reconstruct_image(img_padded.shape, zip(patch_indices, patches), window, stride)  # type: ignore
        result = _unpad_image(result, patch_size, subdivisions)
        result = (result > 0.5).astype(float)

        # Assert
        assert result.shape == img.shape
        np.testing.assert_array_equal(result, img)


class TestSlidingInference:

    # Model correctly processes input tensor and returns expected output tensor shape
    def test_output_matches_input(self):
        input_shape = (1, 16, 32, 32)
        x = torch.rand(input_shape)

        # Run inference
        output = sliding_inference(
            model=tz.identity,
            x=x,
            patch_size=(8, 8, 8),
            batch_size=4,
            output_channels=1,
            prog_bar=False,
        )

        # Assert output shape matches expected
        assert output.shape == x.shape
        assert np.allclose(output, x, atol=1e-3)

    # Model correctly processes input tensor and returns expected output tensor shape
    def test_output_shape_matches_input(self):
        # Setup test data
        input_channels = 1
        output_channels = 3
        input_shape = (16, 32, 32)
        x = torch.randn(input_channels, *input_shape)
        patch_size = (8, 8, 8)
        batch_size = 4

        # Mock model that returns tensor with expected output channels
        def mock_model(x):
            batch_size = x.shape[0]
            return torch.randn(batch_size, output_channels, *patch_size)

        # Run inference
        output = sliding_inference(
            model=mock_model,
            x=x,
            patch_size=patch_size,
            batch_size=batch_size,
            output_channels=output_channels,
            prog_bar=False,
        )

        # Assert output shape matches expected
        expected_shape = (output_channels, *input_shape)
        assert output.shape == expected_shape
