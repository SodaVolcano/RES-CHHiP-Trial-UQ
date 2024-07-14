import numpy as np

from uncertainty.data.preprocessing import make_isotropic

from .context import uncertainty

# Import aliases
Mask = uncertainty.data.datatypes.Mask
PatientScan = uncertainty.data.datatypes.PatientScan
load_patient_scan = uncertainty.data.nifti.load_patient_scan
map_interval = uncertainty.data.preprocessing.map_interval
_isotropic_grid = uncertainty.data.preprocessing._isotropic_grid
_get_spaced_coords = uncertainty.data.preprocessing._get_spaced_coords
make_isotropic = uncertainty.data.preprocessing.make_isotropic


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


class Test_IsotropicGrid:

    # Generates isotropic grid for 1D coordinate arrays
    def test_generates_isotropic_grid_for_1d_coordinate_arrays(self):
        coords = (np.array([0, 2, 4]),)
        expected_output = (np.array([0, 1, 2, 3, 4]),)

        result = _isotropic_grid(coords)
        assert all(np.array_equal(r, e) for r, e in zip(result, expected_output))

    # Generates isotropic grid for 1D coordinate arrays
    def test_generates_isotropic_grid_for_1d_coordinate_arrays_float(self):
        coords = (np.array([0.3, 1.0, 1.7, 2.4, 3.1, 3.8, 4.5, 5.2]),)
        expected_output = (np.array([1, 2, 3, 4, 5]),)

        result = _isotropic_grid(coords)
        assert all(np.array_equal(r, e) for r, e in zip(result, expected_output))

    # Empty coordinate arrays
    def test_empty_coordinate_arrays(self):
        coords = (np.array([]),)
        expected_output = (np.array([]),)

        result = _isotropic_grid(coords)
        assert all(np.array_equal(r, e) for r, e in zip(result, expected_output))

    # Handles negative coordinate values correctly
    def test_handles_negative_coordinates_correctly(self):
        coords = (np.array([-4, -2, 0]),)
        expected_output = (np.array([-4, -3, -2, -1, 0]),)

        result = _isotropic_grid(coords)
        assert all(np.array_equal(r, e) for r, e in zip(result, expected_output))

    # Generates isotropic grid for 3D coordinate arrays
    def test_generates_isotropic_grid_for_3d_coordinate_arrays(self):
        coords = (np.array([0, 2, 4]), np.array([1, 3, 5]), np.array([10, 20, 30]))
        expected_output = tuple(
            np.meshgrid(
                np.arange(0, 5, dtype=float),
                np.arange(1, 6, dtype=float),
                np.arange(10, 31, dtype=float),
                indexing="ij",
            )
        )

        result = _isotropic_grid(coords)
        assert all(np.array_equal(r, e) for r, e in zip(result, expected_output))

    # Coordinate arrays with a single element
    def test_generates_isotropic_grid_for_single_element_array(self):
        coords = (np.array([5]),)
        expected_output = (np.array([5]),)

        result = _isotropic_grid(coords)
        assert all(np.array_equal(r, e) for r, e in zip(result, expected_output))

    # Coordinate arrays with repeating values
    def test_coordinate_arrays_with_repeating_values(self):
        coords = (np.array([0, 1, 1, 2, 2, 3]),)
        expected_output = (np.array([0, 1, 2, 3]),)

        result = _isotropic_grid(coords)
        assert all(np.array_equal(r, e) for r, e in zip(result, expected_output))

    # Handles mixed positive and negative coordinate values
    def test_handles_mixed_positive_and_negative_coordinates(self):
        coords = (np.array([-3, -1, 0, 2, 4]), np.array([-2, 0, 1, 3]))
        expected_output = tuple(
            np.meshgrid(
                np.arange(-3, 5, dtype=float),
                np.arange(-2, 4, dtype=float),
                indexing="ij",
            )
        )

        result = _isotropic_grid(coords)
        assert all(np.array_equal(r, e) for r, e in zip(result, expected_output))


class Test_GetSpacedCoords:

    # Returns correct list of coordinates when given valid spacing and length
    def test_returns_correct_coords_with_valid_spacing_and_length(self):
        spacing = 2
        length = 5
        expected_result = [0, 2, 4, 6, 8]
        result = _get_spaced_coords(spacing, length)
        assert result == expected_result

    # Handles zero spacing correctly
    def test_handles_zero_spacing_correctly(self):
        spacing = 0
        length = 5
        expected_result = [0, 0, 0, 0, 0]
        result = _get_spaced_coords(spacing, length)
        assert result == expected_result


class TestMakeIsotropic:

    # Interpolates a 2D array with given spacings to an isotropic grid with 1 unit spacing
    def test_interpolates_2d_array_to_isotropic_grid(self, mocker):
        import numpy as np

        # Define input array and spacings
        array = np.array([[1, 2], [3, 4]])
        spacings = [2, 2]

        # Expected output after interpolation
        expected_output = np.array([[1, 1.5, 2], [2, 2.5, 3], [3, 3.5, 4]])

        # Call the function
        result = make_isotropic(spacings, array)

        # Assert the result is as expected
        np.testing.assert_array_almost_equal(result, expected_output)
