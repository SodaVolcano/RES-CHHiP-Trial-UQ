from unittest import mock

import numpy as np
import pydicom
from toolz import curried

from .context import gen_path, uncertainty

load_volume = uncertainty.data.dicom.load_volume
c = uncertainty.constants
_load_rt_struct = uncertainty.data.dicom._load_rt_struct
load_mask = uncertainty.data.dicom.load_mask
load_patient_scan = uncertainty.data.dicom.load_patient_scan
load_patient_scans = uncertainty.data.dicom.load_patient_scans
load_all_masks = uncertainty.data.dicom.load_all_masks
Mask = uncertainty.data.mask.Mask
PatientScan = uncertainty.data.patient_scan.PatientScan
get_organ_mask = uncertainty.data.mask.get_organ_mask
get_organ_names = uncertainty.data.mask.get_organ_names

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
PATCH_GET_DICOM_SLICES = "uncertainty.data.dicom._get_dicom_slices"
PATCH_PMAP = "uncertainty.data.dicom.pmap"


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
        mock_dicom.pixel_array = np.zeros((512, 412))
        mock_dicom.PixelSpacing = [0.5, 0.5]
        mock_dicom.SliceThickness = 2.0
        mock_dicom.ImageOrientationPatient = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        mock_dicom.ImagePositionPatient = [-200.0, 200.0, -50.0]
        mock_dicom.RescaleSlope = 1.0
        mock_dicom.RescaleIntercept = 0.0
        mocker.patch(PATCH_DCMREAD, return_value=mock_dicom)

        with mock.patch(PATCH_PMAP, curried.map):
            # Call the load_volume function
            volume = load_volume(gen_path())

        # Assert the volume shape is as expected
        assert volume.shape == (512 // 2, 412 // 2, 8)

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
        mock_dicom.pixel_array = np.zeros((512, 412))
        mock_dicom.PixelSpacing = [0.5, 0.5]
        mock_dicom.SliceThickness = 2.0
        mock_dicom.ImageOrientationPatient = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        mock_dicom.ImagePositionPatient = [-200.0, 200.0, -50.0]
        mock_dicom.RescaleSlope = 1.0
        mock_dicom.RescaleIntercept = 0.0
        mocker.patch(PATCH_DCMREAD, return_value=mock_dicom)

        with mock.patch(PATCH_PMAP, curried.map):
            # Call the load_volume function
            volume = load_volume(gen_path(), preprocess=False)

        # Assert the volume shape is as expected
        assert volume.shape == (512, 412, 4)
        np.testing.assert_array_equal(volume, np.zeros((512, 412, 4)))

    # directory contains no DICOM files
    def test_no_dicom_files_in_directory(self, mocker):
        # Mock the list_files function to return an empty list
        mocker.patch(PATCH_LIST_FILES, return_value=[])

        result = load_volume(gen_path())
        assert result is None


class Test_LoadRtStruct:
    # Successfully loads RTStructBuilder from a valid DICOM RT struct file
    def test_loads_rtstructbuilder_successfully(self, mocker):
        dicom_path = gen_path()
        mock_dicom_file = pydicom.Dataset()
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
        mock_mask1 = np.random.randint(0, 2, (512, 412, 4))
        mock_mask2 = np.random.randint(0, 2, (512, 412, 4))
        mock_rt_struct.get_roi_mask_by_name.side_effect = [mock_mask1, mock_mask2]
        mocker.patch(
            PATCH_LIST_FILES,
            return_value=["file1.dcm", "file2.dcm", "file3.dcm", "file4.dcm"],
        )

        # Mock the dicom.dcmread function to return a mock DICOM object with pixel_array, PixelSpacing, ImageOrientationPatient, and ImagePositionPatient attributes
        mock_dicom = pydicom.Dataset()
        mock_dicom.SOPClassUID = c.CT_IMAGE
        mock_dicom.Rows = 412
        mock_dicom.Columns = 512
        mock_dicom._pixel_array = np.zeros((512, 412))
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

        with mock.patch(PATCH_PMAP, curried.map):
            mask = load_mask(dicom_path)

        assert isinstance(mask, Mask)
        assert "organ_1" in get_organ_names(mask)
        assert "organ_2" in get_organ_names(mask)
        assert mask["organ_1"].shape == (
            int(512 * 0.5),
            int(412 * 0.7),
            8,
        )
        assert mask["organ_2"].shape == (
            int(512 * 0.5),
            int(412 * 0.7),
            8,
        )

    # Successfully load a mask when valid DICOM path is provided
    def test_load_mask_success_no_preprocessing(self, mocker):
        dicom_path = gen_path()
        mock_rt_struct = mocker.Mock()
        mock_rt_struct.get_roi_names.return_value = ["Organ 1", "Organ 2"]
        mock_mask1 = np.random.randint(0, 2, (512, 400, 4))
        mock_mask2 = np.random.randint(0, 2, (512, 400, 4))
        mock_rt_struct.get_roi_mask_by_name.side_effect = [mock_mask1, mock_mask2]
        mocker.patch(
            PATCH_LOAD_RT_STRUCT,
            return_value=mock_rt_struct,
        )

        mask = load_mask(dicom_path, preprocess=False)

        assert isinstance(mask, Mask)
        assert "organ_1" in get_organ_names(mask)
        assert "organ_2" in get_organ_names(mask)
        mock_mask1 = np.flip(mock_mask1, axis=1)
        mock_mask1 = np.flip(mock_mask1, axis=2)
        mock_mask2 = np.flip(mock_mask2, axis=1)
        mock_mask2 = np.flip(mock_mask2, axis=2)
        np.testing.assert_array_equal(mask["organ_1"], mock_mask1)
        np.testing.assert_array_equal(mask["organ_2"], mock_mask2)
        assert mask["organ_1"].shape == (512, 400, 4)
        assert mask["organ_2"].shape == (512, 400, 4)

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
        assert len(get_organ_names(mask)) == 0


class TestLoadPatientScan:
    # Successfully loads a PatientScan with valid DICOM files
    def test_loads_patient_scan_with_valid_dicom_files(self, mocker):
        # Mocking the list_files function to return a list of DICOM file paths
        mocker.patch(
            PATCH_LIST_FILES,
            return_value=["file1.dcm", "file2.dcm"],
        )

        # Mocking the dicom.dcmread function to return a mock object with a PatientID attribute
        mock_dicom = pydicom.Dataset()
        mock_dicom.PatientID = "12345"
        mocker.patch(PATCH_DCMREAD, return_value=mock_dicom)

        # Mocking the load_volume function to return a numpy array
        mocker.patch(
            PATCH_LOAD_VOLUME,
            return_value=np.moveaxis(np.array([[[1, 2], [3, 4]]]), 0, -1),
        )

        # Mocking the load_mask function to return a list of Mask objects
        mock_mask = Mask({}, "observer1")
        mocker.patch(PATCH_LOAD_MASK, return_value=mock_mask)

        with mock.patch(PATCH_PMAP, curried.map):
            # Calling the function under test
            result = load_patient_scan(gen_path())

        # Assertions
        assert result.patient_id == "12345"
        assert isinstance(result.volume, np.ndarray)
        assert isinstance(result, PatientScan)
        assert result.volume.shape == (2, 2, 1)
        assert result.masks == {mock_mask.observer: mock_mask}
        assert get_organ_mask(result, mock_mask.observer) == mock_mask

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
        mock_dicom = pydicom.Dataset()
        mock_dicom.PatientID = "12345"
        mocker.patch(PATCH_DCMREAD, return_value=mock_dicom)

        # Mocking the load_volume function to return a numpy array
        mocker.patch(
            PATCH_LOAD_VOLUME,
            return_value=np.moveaxis(np.array([[[1, 2], [3, 4]]]), 0, -1),
        )

        with mock.patch(PATCH_PMAP, curried.map):
            # Mocking the load_mask function to return None since no RT struct data is found
            mocker.patch(PATCH_LOAD_MASK, return_value=None)

        with mock.patch(PATCH_PMAP, curried.map):
            # Calling the function under test
            result = load_patient_scan(gen_path())

        # Assertions
        assert result.patient_id == "12345"
        assert isinstance(result.volume, np.ndarray)
        assert result.volume.shape == (2, 2, 1)
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
        mock_dicom = pydicom.Dataset()
        mock_dicom.PatientID = "12345"
        mocker.patch(PATCH_DCMREAD, return_value=mock_dicom)

        # Mocking the load_volume function to return a numpy array
        mocker.patch(
            PATCH_LOAD_VOLUME,
            return_value=np.moveaxis(np.array([[[1, 2], [3, 4]]]), 0, -1),
        )

        # Mocking the load_mask function to return a list of Mask objects
        mock_mask = Mask({}, "observer1")
        mocker.patch(PATCH_LOAD_MASK, return_value=mock_mask)

        with mock.patch(PATCH_PMAP, curried.map):
            # Calling the function under test
            result = load_patient_scan(gen_path())

        # Assertions
        assert result.patient_id == "12345"
        assert isinstance(result.volume, np.ndarray)
        assert result.volume.shape == (2, 2, 1)
        assert result.masks == {mock_mask.observer: mock_mask}
        assert get_organ_mask(result, mock_mask.observer) == mock_mask


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
        mock_dicom = pydicom.Dataset()
        mock_dicom.PatientID = "12345"
        mocker.patch(PATCH_DCMREAD, return_value=mock_dicom)

        mock_volume = np.random.randint(0, 2, (512, 512, 4))
        # Mocking the load_volume function to return a numpy array
        mocker.patch(
            PATCH_LOAD_VOLUME,
            return_value=mock_volume,
        )

        mock_mask = Mask(
            {
                "organ_1": np.random.randint(0, 2, (512, 512, 4)),
                "organ_2": np.random.randint(0, 2, (512, 512, 4)),
            },
            "",
        )
        # Mocking the load_mask function to return None since no RT struct data is found
        mocker.patch(PATCH_LOAD_MASK, return_value=mock_mask)

        with mock.patch(PATCH_PMAP, curried.map):
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
                "organ_1" in get_organ_names(mask)
                for scan in result
                for mask in scan.masks.values()
            ]
        )
        assert all(
            [
                "organ_2" in get_organ_names(mask)
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
        mocker.patch(PATCH_LISTDIR, return_value=["patient_1", "patient_2"])
        mock_rt_struct = mocker.Mock()
        mock_rt_struct.get_roi_names.return_value = ["Organ 1", "Organ 2"]
        mock_mask = np.random.randint(0, 2, (512, 512, 4))
        mock_dicom = pydicom.Dataset()
        mock_dicom.SOPClassUID = c.CT_IMAGE
        mock_dicom.Rows = 512
        mock_dicom.Columns = 512
        mock_dicom._pixel_array = np.zeros((512, 512))
        mock_dicom.PixelSpacing = [0.5, 0.5]
        mock_dicom.SliceThickness = 2.0
        mock_dicom.ImageOrientationPatient = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        mock_dicom.ImagePositionPatient = [-200.0, 200.0, -50.0]
        mock_dicom.RescaleSlope = 1.0
        mock_dicom.RescaleIntercept = 0.0
        mocker.patch(PATCH_DCMREAD, return_value=mock_dicom)

        mock_rt_struct.get_roi_mask_by_name.return_value = mock_mask
        mocker.patch(
            PATCH_LOAD_RT_STRUCT,
            return_value=mock_rt_struct,
        )
        mocker.patch(PATCH_GET_DICOM_SLICES, return_value=[mock_dicom, mock_dicom])

        with mock.patch(PATCH_PMAP, curried.map):
            result = list(load_all_masks(dicom_collection_path))

        from uncertainty.data.preprocessing import make_isotropic

        mock_mask = np.flip(mock_mask, axis=1)
        mock_mask = np.flip(mock_mask, axis=2)
        mock_mask_interp = make_isotropic((0.5, 0.5, 2.0), mock_mask, method="nearest")

        assert len(result) == 2
        assert isinstance(result[0], Mask)
        assert isinstance(result[1], Mask)
        assert get_organ_names(result[0]) == ["organ_1", "organ_2"]
        assert get_organ_names(result[1]) == ["organ_1", "organ_2"]
        [
            np.testing.assert_array_equal(mask[organ], mock_mask_interp)
            for mask in result
            for organ in get_organ_names(mask)
        ]

    # Directory contains no DICOM files
    def test_no_dicom_files_in_directory(self, mocker):
        dicom_collection_path = gen_path()
        mocker.patch(PATCH_LISTDIR, return_value=[])

        result = list(load_all_masks(dicom_collection_path))

        assert len(result) == 0
