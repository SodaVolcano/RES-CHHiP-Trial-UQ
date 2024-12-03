from datetime import date

import numpy as np

from ..context import data

# Import aliases
map_interval = data.map_interval
make_isotropic = data.make_isotropic
ensure_min_size = data.ensure_min_size
_bounding_box3d = data.processing._bounding_box3d
preprocess_dataset = data.processing.preprocess_dataset
preprocess_volume = data.processing.preprocess_volume
preprocess_mask = data.processing.preprocess_mask
preprocess_patient_scan = data.processing.preprocess_patient_scan


class TestEnsureMinSize:

    # Input volume and mask smaller than min_size are padded to match min_size
    def test_padding_to_min_size(self):
        # Create small volume and mask
        volume = np.random.rand(
            1,
            2,  # resize to 4
            10,  # no change (bigger than 5)
            6,  # no change (same as 6)
        )  # C,H,W,D
        mask = np.random.randint(0, 2, (3, 2, 10, 6))
        min_size = (4, 5, 6)  # H,W,D

        # Call function
        result_vol, result_mask = ensure_min_size((volume, mask), min_size)

        # Check output shapes
        assert result_vol.shape == (1, 4, 10, 6)
        assert result_mask.shape == (3, 4, 10, 6)

        # Check original content is preserved
        np.testing.assert_array_equal(result_vol[:, 1:3, :, :], volume)
        np.testing.assert_array_equal(result_mask[:, 1:3, :, :], mask)


class TestMapInterval:
    # correctly maps values from one range to another
    def test_correctly_maps_values_from_one_range_to_another(self):
        from_range = (0, 10)
        to_range = (0, 100)
        array = np.array([0, 5, 10])
        expected = np.array([0.0, 50.0, 100.0])

        result = map_interval(from_range, to_range, array)
        np.testing.assert_array_almost_equal(result, expected)

    # handles empty arrays
    def test_handles_empty_arrays(self):
        from_range = (0, 10)
        to_range = (0, 100)
        array = np.array([])
        expected = np.array([])

        result = map_interval(from_range, to_range, array)
        np.testing.assert_array_equal(result, expected)

    # handles negative integer ranges
    def test_handles_negative_integer_ranges(self):
        from_range = (-10, 10)
        to_range = (0, 100)
        array = np.array([-10, 0, 10])
        expected = np.array([0.0, 50.0, 100.0])

        result = map_interval(from_range, to_range, array)
        np.testing.assert_array_almost_equal(result, expected)

    # handles floating point ranges
    def test_handles_floating_point_ranges(self):
        from_range = (0.0, 1.0)
        to_range = (0.0, 100.0)
        array = np.array([0.0, 0.5, 1.0])
        expected = np.array([0.0, 50.0, 100.0])

        result = map_interval(from_range, to_range, array)
        np.testing.assert_array_almost_equal(result, expected)

    # processes multi-dimensional arrays correctly
    def test_processes_multi_dimensional_arrays_correctly(self):
        from_range = (0, 10)
        to_range = (0, 100)
        array = np.array([[0, 5, 10], [1, 6, 11]])
        expected = np.array([[0.0, 50.0, 100.0], [10.0, 60.0, 110.0]])

        result = map_interval(from_range, to_range, array)
        np.testing.assert_array_almost_equal(result, expected)

    # handles arrays with a single element
    def test_handles_single_element_array(self):
        from_range = (0, 10)
        to_range = (0, 100)
        array = np.array([5])
        expected = np.array([50.0])

        result = map_interval(from_range, to_range, array)
        np.testing.assert_array_almost_equal(result, expected)

    # handles arrays with all elements the same
    def test_handles_arrays_with_all_elements_the_same(self):
        from_range = (0, 10)
        to_range = (0, 100)
        array = np.array([5, 5, 5])
        expected = np.array([50.0, 50.0, 50.0])

        result = map_interval(from_range, to_range, array)
        np.testing.assert_array_almost_equal(result, expected)


class TestMakeIsotropic:
    # Interpolates a 2D array with given spacings to an isotropic grid with 1 unit spacing
    def test_interpolates_2d_array_to_isotropic_grid(self, mocker):
        import numpy as np

        # Define input array and spacings
        array = np.array([[1.0, 2.0], [3.0, 4.0]])
        spacings = [2, 2]

        # Expected output after interpolation

        expected_output = np.array(
            [
                [1.0, 1.5, 2.0, 0.0],
                [2.0, 2.5, 3.0, 0.0],
                [3.0, 3.5, 4.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ]
        )

        # Call the function
        result = make_isotropic(array, spacings)

        # Assert the result is as expected
        np.testing.assert_array_almost_equal(result, expected_output)
        assert result.dtype == array.dtype

    def test_interpolates_integer_array(self, mocker):
        import numpy as np

        # Define input array and spacings
        array = np.array([[1, 2], [3, 4]])
        spacings = [2, 2]

        # Expected output after interpolation

        expected_output = np.array(
            [
                [1, 1, 2, 0.0],
                [2, 2, 3, 0.0],
                [3, 3, 4, 0.0],
                [0, 0, 0, 0],
            ]
        )

        # Call the function
        result = make_isotropic(array, spacings)

        # Assert the result is as expected
        np.testing.assert_array_almost_equal(result, expected_output)

        assert result.dtype == array.dtype


class Test_BoundingBox3d:

    # Function correctly computes bounding box coordinates for a 3D binary array with clear boundaries
    def test_bounding_box_computation(self):
        # Create a 3D binary array with clear boundaries
        img = np.zeros((10, 10, 10))
        img[2:5, 3:7, 4:8] = 1

        # Call the function
        rmin, rmax, cmin, cmax, zmin, zmax = _bounding_box3d(img)

        # Assert the correct bounding box coordinates
        assert rmin == 2
        assert rmax == 4
        assert cmin == 3
        assert cmax == 6
        assert zmin == 4
        assert zmax == 7

    def test_bounding_box_oval_volume(self):
        def generate_binary_3d_oval_with_bbox(
            volume_shape=(100, 100, 100), center=(50, 50, 50), radii=(30, 20, 10)
        ):
            z, y, x = np.indices(volume_shape)
            # Normalized distances for the ellipsoid
            normalized_distances = (
                ((x - center[0]) / radii[0]) ** 2
                + ((y - center[1]) / radii[1]) ** 2
                + ((z - center[2]) / radii[2]) ** 2
            )
            binary_volume = (normalized_distances <= 1).astype(np.uint8)
            # Compute bounding box: find the min/max indices where binary volume is 1
            non_zero_indices = np.argwhere(binary_volume)
            min_z, min_y, min_x = non_zero_indices.min(axis=0)
            max_z, max_y, max_x = non_zero_indices.max(axis=0)

            bounding_box = ((min_z, min_y, min_x), (max_z, max_y, max_x))

            return binary_volume, bounding_box

        # Generate binary 3D oval and bounding box
        volume_shape = (100, 100, 100)
        center = (50, 50, 50)
        radii = (30, 20, 10)

        binary_oval, bbox = generate_binary_3d_oval_with_bbox(
            volume_shape, center, radii
        )
        rmin, cmin, zmin = bbox[0]
        rmax, cmax, zmax = bbox[1]

        # Call the function
        rmin_, rmax_, cmin_, cmax_, zmin_, zmax_ = _bounding_box3d(binary_oval)

        # Assert the correct bounding box coordinates
        assert rmin == rmin_
        assert rmax == rmax_
        assert cmin == cmin_
        assert cmax == cmax_
        assert zmin == zmin_
        assert zmax == zmax_


class TestPreprocessVolume:

    # Volume is correctly preprocessed with linear interpolation and default spacings
    def test_preprocess_volume_with_linear_interpolation(self):
        volume = np.random.rand(10, 10, 10)
        spacings = (2.0, 3.0, 1.5)
        interpolation = "linear"

        result = preprocess_volume(volume, spacings, interpolation)

        assert result.shape == (1, 20, 30, 15)
        assert result.dtype == volume.dtype
        assert np.isclose(result.mean(), 0, atol=1e-6)
        assert np.isclose(result.std(), 1, atol=1e-6)


class TestPreprocessMask:

    # Successfully preprocesses mask dictionary with all required organs present
    def test_preprocess_mask_with_all_organs(self):
        mask_dict = {
            "prostate": np.ones((10, 10, 10)),
            "bladder": np.ones((10, 10, 10)),
            "rectum": np.ones((10, 10, 10)),
        }
        spacings = (1.0, 1.5, 1.0)
        organ_ordering = ["prostate", "bladder", "rectum"]

        # Call function
        result = preprocess_mask(mask_dict, spacings, organ_ordering)

        # Verify result
        assert result is not None
        assert isinstance(result, np.ndarray)
        assert result.shape == (3, 10, 15, 10)  # (C,H,W,D) where C is num organs


# Generated by Qodo Gen

import pytest


class TestPreprocessPatientScan:

    # Successfully preprocesses scan with all required organs present
    def test_successful_preprocessing_with_all_organs(self):
        # Prepare test data
        volume = np.random.rand(100, 100, 100)
        masks = {
            "prostate": np.random.randint(0, 2, (100, 100, 100)),
            "bladder": np.random.randint(0, 2, (100, 100, 100)),
            "rectum": np.random.randint(0, 2, (100, 100, 100)),
            "eye": np.random.randint(0, 2, (100, 100, 100)),
            "banana": np.random.randint(0, 2, (100, 100, 100)),
        }

        scan = {
            "patient_id": 1,
            "volume": volume,
            "dimension_original": (100, 100, 100),
            "spacings": (1.0, 1.5, 1.0),
            "modality": "CT",
            "manufacturer": "test",
            "scanner": "test",
            "study_date": date(2020, 1, 1),
            "masks": masks,
        }

        min_size = (100, 100, 120)

        # Call function
        result = preprocess_patient_scan(scan, min_size)

        # Verify result
        assert isinstance(result, dict)
        assert result["patient_id"] == 1
        assert result["volume"].shape == (1, 100, 143, 120)
        assert result["masks"].shape == (3, 100, 143, 120)
        assert all(
            dim >= min_dim for dim, min_dim in zip(result["volume"].shape[1:], min_size)
        )
