import numpy as np

from uncertainty.data.preprocessing import make_isotropic

from .context import uncertainty

# Import aliases
Mask = uncertainty.data.mask.Mask
PatientScan = uncertainty.data.patient_scan.PatientScan
load_patient_scan = uncertainty.data.nifti.load_patient_scan
map_interval = uncertainty.data.preprocessing.map_interval
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


class TestMakeIsotropic:

    # Interpolates a 2D array with given spacings to an isotropic grid with 1 unit spacing
    def test_interpolates_2d_array_to_isotropic_grid(self, mocker):
        import numpy as np

        # Define input array and spacings
        array = np.array([[1, 2], [3, 4]])
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
        result = make_isotropic(spacings, array)

        # Assert the result is as expected
        np.testing.assert_array_almost_equal(result, expected_output)
