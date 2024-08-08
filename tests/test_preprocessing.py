import cv2
import numpy as np

from uncertainty.data.preprocessing import enlarge_array, make_isotropic

from .context import uncertainty

# Import aliases
Mask = uncertainty.data.mask.Mask
PatientScan = uncertainty.data.patient_scan.PatientScan
load_patient_scan = uncertainty.data.nifti.load_patient_scan
map_interval = uncertainty.data.preprocessing.map_interval
make_isotropic = uncertainty.data.preprocessing.make_isotropic
center_box_slice = uncertainty.data.preprocessing.center_box_slice
enlarge_array = uncertainty.data.preprocessing.enlarge_array
shift_center = uncertainty.data.preprocessing.shift_center


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


class TestCenterBoxSlice:

    # Returns correct slices for a centered box within a larger background
    def test_centered_box_within_larger_background(self):
        background_shape = (10, 10)
        box_shape = (4, 4)
        expected_slices = (slice(3, 7), slice(3, 7))
        result = center_box_slice(background_shape, box_shape)
        assert result == expected_slices

    # Box shape larger than background shape
    def test_box_larger_than_background(self):
        background_shape = (4, 4)
        box_shape = (10, 10)
        expected_slices = (slice(-3, 7), slice(-3, 7))
        result = center_box_slice(background_shape, box_shape)
        assert result == expected_slices

    def test_box_odd_shape(self):
        result = center_box_slice((15, 15, 9), (5, 5, 3))
        expected = (slice(5, 10), slice(5, 10), slice(3, 6))
        assert result == expected


class TestEnlargeArray:

    # Enlarge array with scale 2 and fill "min"
    def test_enlarge_array_scale_2_fill_min(self):
        array = np.array([[1, 2], [3, 4]])
        result = enlarge_array(array, 2, "min")
        expected = np.array([[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]])
        expected[1:3, 1:3] = array
        assert np.array_equal(result, expected)

    # Enlarge array with scale 3 and fill "max"
    def test_enlarge_array_scale_3_fill_max(self):
        array = np.array([[1, 2], [3, 4]])
        result = enlarge_array(array, 3, "max")
        expected = np.array(
            [
                [4, 4, 4, 4, 4, 4],
                [4, 4, 4, 4, 4, 4],
                [4, 4, 1, 2, 4, 4],
                [4, 4, 3, 4, 4, 4],
                [4, 4, 4, 4, 4, 4],
                [4, 4, 4, 4, 4, 4],
            ]
        )
        assert np.array_equal(result, expected)

    # Enlarge array with scale 3 and fill 10
    def test_enlarge_array_scale_3_fill_10(self):
        array = np.array([[1, 2], [3, 4]])
        result = enlarge_array(array, 3, 10)
        expected = np.array(
            [
                [10, 10, 10, 10, 10, 10],
                [10, 10, 10, 10, 10, 10],
                [10, 10, 1, 2, 10, 10],
                [10, 10, 3, 4, 10, 10],
                [10, 10, 10, 10, 10, 10],
                [10, 10, 10, 10, 10, 10],
            ]
        )
        assert np.array_equal(result, expected)

    # Enlarge array with scale 1 and fill as an integer
    def test_enlarge_array_scale_1_fill_integer(self):
        expected = np.array([[1, 2], [3, 4]])
        result = enlarge_array(expected, 1, 0)
        assert np.array_equal(result, expected)

    def test_enlarge_array_scale_3_odd_depth(self):
        array = np.zeros((5, 5, 3))
        expected = np.zeros((15, 15, 9))
        result = enlarge_array(array, 3, "min")
        assert np.array_equal(result, expected)


class TestShiftCenter:

    # Shifts a point to the center of a 2D array correctly
    def test_shift_point_to_center_2d_array(self):

        axes = (6, 2)  # Width and height of the ellipse
        angle = 150.0  # Angle of rotation of the ellipse
        start_angle = 0.0  # Starting angle of the elliptic arc
        end_angle = 360.0  # Ending angle of the elliptic arc
        color = (255, 255, 255)  # Color (white)

        # Create a 20x20 array filled with zeros (black background)
        expected = np.zeros((20, 20), dtype=np.uint8)
        cv2.ellipse(expected, (10, 10), axes, angle, start_angle, end_angle, color, -1)
        img = np.zeros((20, 20), dtype=np.uint8)
        cv2.ellipse(img, (13, 3), axes, angle, start_angle, end_angle, color, -1)

        result = shift_center(img, np.mean(np.argwhere(img), axis=0))
        np.testing.assert_array_equal(result, expected)

    # Shifts a point to the center of a 3D array correctly
    def test_shift_point_to_center_3d_array(self):
        def create_3d_sphere(shape: tuple, center: tuple, radius: int) -> np.ndarray:
            z, y, x = np.ogrid[: shape[0], : shape[1], : shape[2]]
            cz, cy, cx = center
            return (z - cz) ** 2 + (y - cy) ** 2 + (x - cx) ** 2 <= radius**2

        # Parameters
        shape = (20, 20, 20)
        radius = 3

        img = create_3d_sphere(shape, (15, 7, 12), radius)
        expected = create_3d_sphere(shape, (10, 10, 10), radius)
        result = shift_center(img, np.mean(np.argwhere(img), axis=0))
        assert np.array_equal(result, expected)
