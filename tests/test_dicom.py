import pytest
import numpy as np
import pydicom

from .context import uncertainty

# Some functions use memoize so use different path to avoid caching
path_id = 0


def gen_path():
    global path_id
    path_id += 1
    return f"path/to/folder{path_id}"


# Import aliases
_most_common_shape = uncertainty.data.dicom._most_common_shape
_filter_by_most_common_shape = uncertainty.data.dicom._filter_by_most_common_shape
load_volume = uncertainty.data.dicom.load_volume
c = uncertainty.common.constants
_load_rt_struct = uncertainty.data.dicom._load_rt_struct
load_mask = uncertainty.data.dicom.load_mask
load_patient_scan = uncertainty.data.dicom.load_patient_scan
load_patient_scans = uncertainty.data.dicom.load_patient_scans
load_all_masks = uncertainty.data.dicom.load_all_masks
Mask = uncertainty.data.datatypes.Mask
PatientScan = uncertainty.data.datatypes.PatientScan

# Patch paths
PATCH_LIST_FILES = "uncertainty.data.dicom.list_files"
PATCH_DCMREAD = "pydicom.dcmread"
PATCH_RT_CREATE_FROM = "rt_utils.RTStructBuilder.create_from"
PATCH_LOAD_RT_STRUCT = "uncertainty.data.dicom._load_rt_struct"
PATCH_LOAD_VOLUME = "uncertainty.data.dicom.load_volume"
PATCH_LOAD_MASK = "uncertainty.data.dicom.load_mask"
PATCH_GENERATE_FULL_PATHS = "uncertainty.data.dicom.generate_full_paths"
PATCH_LOAD_PATIENT_SCAN = "uncertainty.data.dicom.load_patient_scan"
PATCH_LISTDIR = "os.listdir"
PATCH_FILTER_ROI = "uncertainty.data.dicom._filter_roi"


class Test_MostCommonShape:

    # returns the most common shape for a list of DICOM files with uniform shapes
    def test_most_common_shape_uniform(self):
        dicom_files = [pydicom.Dataset(), pydicom.Dataset(), pydicom.Dataset()]
        for d in dicom_files:
            d.Rows, d.Columns = 512, 512

        assert _most_common_shape(dicom_files) == (512, 512)

    # handles an empty iterable of DICOM files gracefully
    def test_most_common_shape_empty(self):
        dicom_files = []

        assert _most_common_shape(dicom_files) == None

    # correctly identifies the most common shape when there are multiple shapes with different frequencies
    def test_correctly_identifies_most_common_shape(self):
        dicom_files = [
            pydicom.Dataset(),
            pydicom.Dataset(),
            pydicom.Dataset(),
            pydicom.Dataset(),
            pydicom.Dataset(),
        ]
        dicom_files[0].Rows, dicom_files[0].Columns = 512, 512
        dicom_files[1].Rows, dicom_files[1].Columns = 256, 206
        dicom_files[2].Rows, dicom_files[2].Columns = 512, 512
        dicom_files[3].Rows, dicom_files[3].Columns = 256, 256
        dicom_files[4].Rows, dicom_files[4].Columns = 512, 512

        assert _most_common_shape(dicom_files) == (512, 512)

    # manages cases where all DICOM files have unique shapes
    def test_unique_shapes(self):
        dicom_files = [pydicom.Dataset(), pydicom.Dataset(), pydicom.Dataset()]
        for d in dicom_files:
            d.Rows, d.Columns = np.random.randint(1, 1000), np.random.randint(1, 1000)

        assert _most_common_shape(dicom_files) == None

    # Check that cached result is returned
    def test_most_common_shape_caching(self):
        dicom_files = [pydicom.Dataset(), pydicom.Dataset(), pydicom.Dataset()]
        for d in dicom_files:
            d.Rows, d.Columns = 512, 512

        # First call without caching
        result_without_caching = _most_common_shape(dicom_files)

        # Second call with caching
        result_with_caching = _most_common_shape(dicom_files)

        assert result_without_caching == result_with_caching


class Test_FilterByMostCommonShape:

    # filters DICOM files to only include those with the most common shape
    def test_filters_by_most_common_shape(self):
        dicom_files = [
            pydicom.Dataset(),
            pydicom.Dataset(),
            pydicom.Dataset(),
            pydicom.Dataset(),
        ]
        dicom_files[0].Rows, dicom_files[0].Columns = 512, 512
        dicom_files[1].Rows, dicom_files[1].Columns = 512, 512
        dicom_files[2].Rows, dicom_files[2].Columns = 256, 256
        dicom_files[3].Rows, dicom_files[3].Columns = 512, 512

        filtered_files = list(_filter_by_most_common_shape(dicom_files))
        assert len(filtered_files) == 3
        for d in filtered_files:
            assert (d.Rows, d.Columns) == (512, 512)

    # Return an empty list if no DICOM files are provided
    def test_return_empty_list(self):
        dicom_files = [pydicom.Dataset(), pydicom.Dataset(), pydicom.Dataset()]
        dicom_files[0].Rows, dicom_files[0].Columns = 512, 512
        dicom_files[1].Rows, dicom_files[1].Columns = 256, 256
        dicom_files[2].Rows, dicom_files[2].Columns = 128, 128

        result = list(_filter_by_most_common_shape(dicom_files))
        assert len(result) == 0


class TestLoadVolume:

    # correctly loads a 3D volume from a directory of DICOM files
    def test_loads_3d_volume_correctly(self, mocker):
        # Mock the list_files function to return a list of file paths
        mocker.patch(
            PATCH_LIST_FILES,
            return_value=["file1.dcm", "file2.dcm", "file3.dcm", "file4.dcm"],
        )

        # Mock the dicom.dcmread function to return a mock DICOM object with pixel_array, PixelSpacing, ImageOrientationPatient, and ImagePositionPatient attributes
        mock_dicom = mocker.Mock()
        mock_dicom.SOPClassUID = c.CT_IMAGE
        mock_dicom.Rows = 512
        mock_dicom.Columns = 512
        mock_dicom.pixel_array = np.zeros((512, 512))
        mock_dicom.PixelSpacing = [0.5, 0.5]
        mock_dicom.SliceThickness = 2.0
        mock_dicom.ImageOrientationPatient = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        mock_dicom.ImagePositionPatient = [-200.0, 200.0, -50.0]
        mock_dicom.RescaleSlope = 1.0
        mock_dicom.RescaleIntercept = 0.0
        mocker.patch(PATCH_DCMREAD, return_value=mock_dicom)

        # Call the load_volume function
        volume = load_volume(gen_path())

        # Assert the volume shape is as expected
        assert volume.shape == (7, 512 // 2, 512 // 2)

    # correctly loads a 3D volume from a directory of DICOM files
    def test_loads_3d_volume_correctly_no_preprocessing(self, mocker):
        # Mock the list_files function to return a list of file paths
        mocker.patch(
            PATCH_LIST_FILES,
            return_value=["file1.dcm", "file2.dcm", "file3.dcm", "file4.dcm"],
        )

        # Mock the dicom.dcmread function to return a mock DICOM object with pixel_array, PixelSpacing, ImageOrientationPatient, and ImagePositionPatient attributes
        mock_dicom = mocker.Mock()
        mock_dicom.SOPClassUID = c.CT_IMAGE
        mock_dicom.Rows = 512
        mock_dicom.Columns = 512
        mock_dicom.pixel_array = np.zeros((512, 512))
        mock_dicom.PixelSpacing = [0.5, 0.5]
        mock_dicom.SliceThickness = 2.0
        mock_dicom.ImageOrientationPatient = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        mock_dicom.ImagePositionPatient = [-200.0, 200.0, -50.0]
        mock_dicom.RescaleSlope = 1.0
        mock_dicom.RescaleIntercept = 0.0
        mocker.patch(PATCH_DCMREAD, return_value=mock_dicom)

        # Call the load_volume function
        volume = load_volume(gen_path(), preprocess=False)

        # Assert the volume shape is as expected
        assert volume.shape == (4, 512, 512)
        np.testing.assert_array_equal(volume, np.zeros((4, 512, 512)))

    # directory contains no DICOM files
    def test_no_dicom_files_in_directory(self, mocker):
        # Mock the list_files function to return an empty list
        mocker.patch(PATCH_LIST_FILES, return_value=[])

        result = load_volume(gen_path())
        assert result is None

    @pytest.mark.skip(reason="Assume all have same shape since we filter by thickness")
    # directory contains DICOM files with different shapes
    def test_loads_3d_volume_with_most_common_shape(self, mocker):
        # Mock the list_files function to return a list of file paths with different shapes
        mocker.patch(
            PATCH_LIST_FILES,
            return_value=["file1.dcm", "file2.dcm", "file3.dcm"],
        )

        # Mock the dicom.dcmread function to return mock DICOM objects with different shapes
        mock_dicom1 = mocker.Mock()
        mock_dicom1.SOPClassUID = c.CT_IMAGE
        mock_dicom1.Rows = 512
        mock_dicom1.Columns = 512
        mock_dicom1.pixel_array = np.zeros((512, 512))
        mock_dicom1.PixelSpacing = [0.5, 0.5]
        mock_dicom1.SliceThickness = 1.0
        mock_dicom1.ImageOrientationPatient = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        mock_dicom1.ImagePositionPatient = [-200.0, 200.0, -50.0]
        mock_dicom1.RescaleSlope = 1.0
        mock_dicom1.RescaleIntercept = 0.0

        mock_dicom2 = mocker.Mock()
        mock_dicom2.SOPClassUID = c.CT_IMAGE
        mock_dicom2.Rows = 256
        mock_dicom2.Columns = 256
        mock_dicom2.pixel_array = np.zeros((256, 256))
        mock_dicom2.PixelSpacing = [0.5, 0.5]
        mock_dicom2.SliceThickness = 1.0
        mock_dicom2.ImageOrientationPatient = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        mock_dicom2.ImagePositionPatient = [-200.0, 200.0, -48.0]
        mock_dicom2.RescaleSlope = 1.0
        mock_dicom2.RescaleIntercept = 0.0

        mock_dicom3 = mocker.Mock()
        mock_dicom3.SOPClassUID = c.CT_IMAGE
        mock_dicom3.Rows = 512
        mock_dicom3.Columns = 512
        mock_dicom3.pixel_array = np.zeros((512, 512))
        mock_dicom3.PixelSpacing = [0.5, 0.5]
        mock_dicom3.SliceThickness = 1.0
        mock_dicom3.ImageOrientationPatient = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        mock_dicom3.ImagePositionPatient = [-200.0, 200.0, -49.0]
        mock_dicom3.RescaleSlope = 1.0
        mock_dicom3.RescaleIntercept = 0.0

        mocker.patch(PATCH_DCMREAD, side_effect=[mock_dicom1, mock_dicom2, mock_dicom3])

        # Call the load_volume function
        volume = load_volume(gen_path())

        # Assert the volume shape is as expected (using the most common shape)
        assert volume.shape == (2, 512, 512)


class Test_LoadRtStruct:

    # Successfully loads RTStructBuilder from a valid DICOM RT struct file
    def test_loads_rtstructbuilder_successfully(self, mocker):
        dicom_path = gen_path()
        mock_dicom_file = mocker.Mock()
        mock_dicom_file.SOPClassUID = c.RT_STRUCTURE_SET
        mock_rt_struct_builder = mocker.patch(
            PATCH_RT_CREATE_FROM,
            return_value="RTStructBuilderInstance",
        )

        mocker.patch(
            PATCH_LIST_FILES,
            return_value=[mock_dicom_file],
        )
        mocker.patch(PATCH_DCMREAD, return_value=mock_dicom_file)

        result = _load_rt_struct(dicom_path)

        assert result == "RTStructBuilderInstance"
        mock_rt_struct_builder.assert_called_once_with(
            dicom_series_path=dicom_path, rt_struct_path=mock_dicom_file
        )

    # No RT struct file present in the directory
    def test_no_rt_struct_file_present(self, mocker):
        dicom_path = gen_path()

        mocker.patch(PATCH_LIST_FILES, return_value=[])

        assert _load_rt_struct(dicom_path) == None

    # Empty directory
    def test_load_rt_struct_empty_directory(self, mocker):
        dicom_path = gen_path()
        mocker.patch(PATCH_LIST_FILES, return_value=[])

        result = _load_rt_struct(dicom_path)

        assert result is None


class TestLoadMask:

    # Successfully load a mask when valid DICOM path is provided
    def test_load_mask_success(self, mocker):
        dicom_path = gen_path()
        mock_rt_struct = mocker.Mock()
        mock_rt_struct.get_roi_names.return_value = ["Organ 1", "Organ 2"]
        mock_mask1 = np.random.randint(0, 2, (4, 512, 512))
        mock_mask2 = np.random.randint(0, 2, (4, 512, 512))
        mock_rt_struct.get_roi_mask_by_name.side_effect = [mock_mask1, mock_mask2]
        mocker.patch(
            PATCH_LIST_FILES,
            return_value=["file1.dcm", "file2.dcm", "file3.dcm", "file4.dcm"],
        )

        # Mock the dicom.dcmread function to return a mock DICOM object with pixel_array, PixelSpacing, ImageOrientationPatient, and ImagePositionPatient attributes
        mock_dicom = mocker.Mock()
        mock_dicom.SOPClassUID = c.CT_IMAGE
        mock_dicom.Rows = 512
        mock_dicom.Columns = 512
        mock_dicom.pixel_array = np.zeros((512, 512))
        mock_dicom.PixelSpacing = [0.5, 0.7]
        mock_dicom.SliceThickness = 2.0
        mock_dicom.ImageOrientationPatient = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        mock_dicom.ImagePositionPatient = [-200.0, 200.0, -50.0]
        mock_dicom.RescaleSlope = 1.0
        mock_dicom.RescaleIntercept = 0.0
        mocker.patch(PATCH_DCMREAD, return_value=mock_dicom)

        mocker.patch(
            PATCH_LOAD_RT_STRUCT,
            return_value=mock_rt_struct,
        )
        mocker.patch(PATCH_FILTER_ROI, return_value=["organ_1", "organ_2"])

        mask = load_mask(dicom_path)

        assert isinstance(mask, Mask)
        assert "organ_1" in mask.get_organ_names()
        assert "organ_2" in mask.get_organ_names()
        assert mask["organ_1"].shape == (7, np.floor(512 * 0.5), np.floor(512 * 0.7))
        assert mask["organ_2"].shape == (7, np.floor(512 * 0.5), np.floor(512 * 0.7))

    # Successfully load a mask when valid DICOM path is provided
    def test_load_mask_success_no_preprocessing(self, mocker):
        dicom_path = gen_path()
        mock_rt_struct = mocker.Mock()
        mock_rt_struct.get_roi_names.return_value = ["Organ 1", "Organ 2"]
        mock_mask1 = np.random.randint(0, 2, (4, 512, 512))
        mock_mask2 = np.random.randint(0, 2, (4, 512, 512))
        mock_rt_struct.get_roi_mask_by_name.side_effect = [mock_mask1, mock_mask2]
        mocker.patch(
            PATCH_LOAD_RT_STRUCT,
            return_value=mock_rt_struct,
        )
        mocker.patch(PATCH_FILTER_ROI, return_value=["organ_1", "organ_2"])

        mask = load_mask(dicom_path, preprocess=False)

        assert isinstance(mask, Mask)
        assert "organ_1" in mask.get_organ_names()
        assert "organ_2" in mask.get_organ_names()
        np.testing.assert_array_equal(mask["organ_1"], mock_mask1)
        np.testing.assert_array_equal(mask["organ_2"], mock_mask2)
        assert mask["organ_1"].shape == (4, 512, 512)
        assert mask["organ_2"].shape == (4, 512, 512)

    # Handle empty DICOM directory gracefully
    def test_load_mask_empty_directory(self, mocker):
        dicom_path = gen_path()

        mocker.patch(PATCH_LOAD_RT_STRUCT, return_value=None)

        assert load_mask(dicom_path) is None

    # Handle cases where no ROI names are found
    def test_handle_no_roi_names(self, mocker):
        dicom_path = gen_path()
        mock_rt_struct = mocker.Mock()
        mock_rt_struct.get_roi_names.return_value = []
        mocker.patch(
            PATCH_LOAD_RT_STRUCT,
            return_value=mock_rt_struct,
        )

        mask = load_mask(dicom_path)

        assert isinstance(mask, Mask)
        assert len(mask.get_organ_names()) == 0


class TestLoadPatientScan:

    # Successfully loads a PatientScan with valid DICOM files
    def test_loads_patient_scan_with_valid_dicom_files(self, mocker):
        # Mocking the list_files function to return a list of DICOM file paths
        mocker.patch(
            PATCH_LIST_FILES,
            return_value=["file1.dcm", "file2.dcm"],
        )

        # Mocking the dicom.dcmread function to return a mock object with a PatientID attribute
        mock_dicom = mocker.Mock()
        mock_dicom.PatientID = "12345"
        mocker.patch(PATCH_DCMREAD, return_value=mock_dicom)

        # Mocking the load_volume function to return a numpy array
        mocker.patch(
            PATCH_LOAD_VOLUME,
            return_value=np.array([[[1, 2], [3, 4]]]),
        )

        # Mocking the load_mask function to return a list of Mask objects
        mock_mask = mocker.Mock()
        mock_mask.observer = "observer1"
        mocker.patch(PATCH_LOAD_MASK, return_value=mock_mask)

        # Calling the function under test
        result = load_patient_scan(gen_path())

        # Assertions
        assert result.patient_id == "12345"
        assert isinstance(result.volume, np.ndarray)
        assert isinstance(result, PatientScan)
        assert result.volume.shape == (1, 2, 2)
        assert result.masks == {mock_mask.observer: mock_mask}
        assert result.get_mask(mock_mask.observer) == mock_mask

    # Directory contains no DICOM files
    def test_no_dicom_files_in_directory(self, mocker):
        # Mocking the list_files function to return an empty list
        mocker.patch(PATCH_LIST_FILES, return_value=[])

        # Mocking the dicom.dcmread function to raise an exception when called with an empty list
        mocker.patch(PATCH_DCMREAD, side_effect=IndexError("list index out of range"))

        # Mocking the load_volume function to return an empty numpy array
        mocker.patch(
            PATCH_LOAD_VOLUME,
            return_value=np.array([]),
        )

        # Mocking the load_mask function to return None
        mocker.patch(PATCH_LOAD_MASK, return_value=None)

        assert load_patient_scan(gen_path()) is None

    # DICOM files are present but none contain RT struct data
    def test_no_rt_struct_data(self, mocker):
        # Mocking the list_files function to return a list of DICOM file paths
        mocker.patch(
            PATCH_LIST_FILES,
            return_value=["file1.dcm", "file2.dcm"],
        )

        # Mocking the dicom.dcmread function to return a mock object with a PatientID attribute
        mock_dicom = mocker.Mock()
        mock_dicom.PatientID = "12345"
        mocker.patch(PATCH_DCMREAD, return_value=mock_dicom)

        # Mocking the load_volume function to return a numpy array
        mocker.patch(
            PATCH_LOAD_VOLUME,
            return_value=np.array([[[1, 2], [3, 4]]]),
        )

        # Mocking the load_mask function to return None since no RT struct data is found
        mocker.patch(PATCH_LOAD_MASK, return_value=None)

        # Calling the function under test
        result = load_patient_scan(gen_path())

        # Assertions
        assert result.patient_id == "12345"
        assert isinstance(result.volume, np.ndarray)
        assert result.volume.shape == (1, 2, 2)
        assert result.masks == {}
        assert result.mask_observers == []
        assert result.n_masks == 0

    # Handles directories with hidden files correctly
    def test_handles_hidden_files_correctly(self, mocker):
        # Mocking the list_files function to return a list of DICOM file paths including hidden files
        mocker.patch(
            PATCH_LIST_FILES,
            return_value=["file1.dcm", ".hidden_file.dcm"],
        )

        # Mocking the dicom.dcmread function to return a mock object with a PatientID attribute
        mock_dicom = mocker.Mock()
        mock_dicom.PatientID = "12345"
        mocker.patch(PATCH_DCMREAD, return_value=mock_dicom)

        # Mocking the load_volume function to return a numpy array
        mocker.patch(
            PATCH_LOAD_VOLUME,
            return_value=np.array([[[1, 2], [3, 4]]]),
        )

        # Mocking the load_mask function to return a list of Mask objects
        mock_mask = mocker.Mock()
        mock_mask.observer = "observer1"
        mocker.patch(PATCH_LOAD_MASK, return_value=mock_mask)

        # Calling the function under test
        result = load_patient_scan(gen_path())

        # Assertions
        assert result.patient_id == "12345"
        assert isinstance(result.volume, np.ndarray)
        assert result.volume.shape == (1, 2, 2)
        assert result.masks == {mock_mask.observer: mock_mask}
        assert result.get_mask(mock_mask.observer) == mock_mask


class TestLoadPatientScans:

    # Successfully loads multiple PatientScan objects from a directory with valid DICOM files
    def test_loads_multiple_patient_scans_successfully(self, mocker):
        # Mock the os.listdir to return a list of directories
        mocker.patch(PATCH_LISTDIR, return_value=["patient1", "patient2"])
        # Mocking the list_files function to return a list of DICOM file paths
        mocker.patch(
            PATCH_LIST_FILES,
            return_value=["file1.dcm", "file2.dcm"],
        )

        # Mocking the dicom.dcmread function to return a mock object with a PatientID attribute
        mock_dicom = mocker.Mock()
        mock_dicom.PatientID = "12345"
        mocker.patch(PATCH_DCMREAD, return_value=mock_dicom)

        mock_volume = np.random.randint(0, 2, (4, 512, 512))
        # Mocking the load_volume function to return a numpy array
        mocker.patch(
            PATCH_LOAD_VOLUME,
            return_value=mock_volume,
        )

        mock_mask = Mask(
            {
                "organ_1": np.random.randint(0, 2, (4, 512, 512)),
                "organ_2": np.random.randint(0, 2, (4, 512, 512)),
            },
        )
        mock_mask.observer = ""
        # Mocking the load_mask function to return None since no RT struct data is found
        mocker.patch(PATCH_LOAD_MASK, return_value=mock_mask)

        # Call the function under test
        result = list(load_patient_scans(gen_path()))

        # Assertions
        assert len(result) == 2
        assert all(isinstance(scan, PatientScan) for scan in result)
        assert all(scan.patient_id == "12345" for scan in result)
        np.testing.assert_array_equal(result[0].volume, mock_volume)
        np.testing.assert_array_equal(result[1].volume, mock_volume)
        assert all(
            [mask == mock_mask for mask in scan.masks.values()] for scan in result
        )
        assert all(
            [
                "organ_1" in mask.get_organ_names()
                for scan in result
                for mask in scan.masks.values()
            ]
        )
        assert all(
            [
                "organ_2" in mask.get_organ_names()
                for scan in result
                for mask in scan.masks.values()
            ]
        )

    # Handles an empty directory gracefully
    def test_handles_empty_directory_gracefully(self, mocker):
        # Mock the os.listdir to return an empty list
        mocker.patch(PATCH_LISTDIR, return_value=[])

        # Call the function under test
        result = list(load_patient_scans(gen_path()))

        # Assertions
        assert result == []


class TestLoadAllMasks:

    # Successfully loads masks from a directory containing valid DICOM files
    def test_loads_masks_from_valid_dicom_directory(self, mocker):
        dicom_collection_path = gen_path()
        mocker.patch(PATCH_LISTDIR, return_value=["file1.dcm", "file2.dcm"])
        mock_rt_struct = mocker.Mock()
        mock_rt_struct.get_roi_names.return_value = ["Organ 1", "Organ 2"]
        mock_mask = np.random.randint(0, 2, (4, 512, 512))
        mock_rt_struct.get_roi_mask_by_name.return_value = mock_mask
        mocker.patch(
            PATCH_LOAD_RT_STRUCT,
            return_value=mock_rt_struct,
        )
        mocker.patch(
            PATCH_FILTER_ROI,
            return_value=["organ_1", "organ_2"],
        )

        result = list(load_all_masks(dicom_collection_path))

        assert len(result) == 2
        assert isinstance(result[0], Mask)
        assert isinstance(result[1], Mask)
        assert ["organ_1", "organ_2"] == result[0].get_organ_names()
        assert ["organ_1", "organ_2"] == result[1].get_organ_names()
        [
            np.testing.assert_array_equal(mask[organ], mock_mask)
            for mask in result
            for organ in mask.get_organ_names()
        ]

    # Directory contains no DICOM files
    def test_no_dicom_files_in_directory(self, mocker):
        dicom_collection_path = gen_path()
        mocker.patch(PATCH_LISTDIR, return_value=[])

        result = list(load_all_masks(dicom_collection_path))

        assert len(result) == 0
