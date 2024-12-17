from unittest import mock

import numpy as np
import pydicom

from ..context import constants, data, gen_path

load_volume = data.load_volume
c = constants
_load_rt_structs = data.dicom._load_rt_structs
load_mask = data.load_mask
load_patient_scan = data.load_patient_scan
load_all_patient_scans = data.load_all_patient_scans
load_all_masks = data.dicom.load_all_masks
MaskDict = data.MaskDict
PatientScan = data.PatientScan
load_roi_names = data.load_roi_names

# Patch paths
PATCH_LIST_FILES = "uncertainty.data.dicom.list_files"
PATCH_DCMREAD = "pydicom.dcmread"
PATCH_RT_CREATE_FROM = "rt_utils.RTStructBuilder.create_from"
PATCH_LOAD_RT_STRUCTS = "uncertainty.data.dicom._load_rt_structs"
PATCH_LOAD_VOLUME = "uncertainty.data.dicom.load_volume"
PATCH_LOAD_MASK = "uncertainty.data.dicom.load_mask"
PATCH_GENERATE_FULL_PATHS = "uncertainty.data.dicom.generate_full_paths"
PATCH_LOAD_PATIENT_SCAN = "uncertainty.data.dicom.load_patient_scan"
PATCH_LISTDIR = "os.listdir"
PATCH_GET_DICOM_SLICES = "uncertainty.data.dicom._get_dicom_slices"

MOCK_DICOM = mock.Mock()
MOCK_DICOM.PatientID = "12345"
MOCK_DICOM.Modality = "CT"
MOCK_DICOM.Manufacturer = "manufacturer"
MOCK_DICOM.scanner = "scanner"
MOCK_DICOM.StudyDate = "20171208"
MOCK_DICOM.ManufacturerModelName = "model"
MOCK_DICOM.SOPClassUID = c.CT_IMAGE
MOCK_DICOM.Rows = 512
MOCK_DICOM.Columns = 512
MOCK_DICOM.pixel_array = np.zeros((512, 412))  # type: ignore
MOCK_DICOM.PixelSpacing = [0.5, 0.5]
MOCK_DICOM.SliceThickness = 2.0
MOCK_DICOM.ImageOrientationPatient = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
MOCK_DICOM.ImagePositionPatient = [-200.0, 200.0, -50.0]
MOCK_DICOM.RescaleSlope = 1.0
MOCK_DICOM.RescaleIntercept = 0.0


class TestLoadVolume:
    # correctly loads a 3D volume from a directory of DICOM files
    def test_loads_3d_volume_correctly(self, mocker):
        # Mock the list_files function to return a list of file paths
        mocker.patch(
            PATCH_LIST_FILES,
            return_value=["file1.dcm", "file2.dcm", "file3.dcm", "file4.dcm"],
        )

        # Mock the dicom.dcmread function to return a mock DICOM object with pixel_array, PixelSpacing, ImageOrientationPatient, and ImagePositionPatient attributes

        mocker.patch(PATCH_DCMREAD, return_value=MOCK_DICOM)

        volume = load_volume(gen_path())

        # Assert the volume shape is as expected
        assert volume.shape == (512, 412, 4)

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
            return_value=["file.dcm"],
        )
        mocker.patch(PATCH_DCMREAD, return_value=mock_dicom_file)

        result = list(_load_rt_structs(dicom_path))[0]

        assert result == "RTStructBuilderInstance"
        mock_rt_struct_builder.assert_called_once_with(
            dicom_series_path=dicom_path, rt_struct_path="file.dcm"
        )

    # No RT struct file present in the directory
    def test_no_rt_struct_file_present(self, mocker):
        dicom_path = gen_path()

        mocker.patch(PATCH_LIST_FILES, return_value=[])

        assert list(_load_rt_structs(dicom_path)) == []

    # Empty directory
    def test_load_rt_struct_empty_directory(self, mocker):
        dicom_path = gen_path()
        mocker.patch(PATCH_LIST_FILES, return_value=[])

        result = list(_load_rt_structs(dicom_path))

        assert result == []


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

        mocker.patch(PATCH_DCMREAD, return_value=MOCK_DICOM)

        mocker.patch(
            PATCH_LOAD_RT_STRUCTS,
            return_value=[mock_rt_struct],
        )

        mask = load_mask(dicom_path)

        assert "organ_1" in mask.keys()
        assert "organ_2" in mask.keys()
        assert mask["organ_1"].shape == mock_mask1.shape
        assert mask["organ_2"].shape == mock_mask2.shape

    # Handle empty DICOM directory gracefully
    def test_load_mask_empty_directory(self, mocker):
        dicom_path = gen_path()

        mocker.patch(PATCH_LOAD_RT_STRUCTS, return_value=[])

        assert load_mask(dicom_path) is None

    # Handle cases where no ROI names are found
    def test_handle_no_roi_names(self, mocker):
        dicom_path = gen_path()
        mock_rt_struct = mocker.Mock()
        mock_rt_struct.get_roi_names.return_value = []
        mocker.patch(
            PATCH_LOAD_RT_STRUCTS,
            return_value=[mock_rt_struct],
        )

        mask = load_mask(dicom_path)

        assert len(mask.keys()) == 0


class TestLoadPatientScan:
    # Successfully loads a PatientScan with valid DICOM files
    def test_loads_patient_scan_with_valid_dicom_files(self, mocker):
        # Mocking the list_files function to return a list of DICOM file paths
        mocker.patch(
            PATCH_LIST_FILES,
            return_value=["file1.dcm", "file2.dcm"],
        )

        mocker.patch(PATCH_DCMREAD, return_value=MOCK_DICOM)

        # Mocking the load_volume function to return a numpy array
        mocker.patch(
            PATCH_LOAD_VOLUME,
            return_value=np.moveaxis(np.array([[[1, 2], [3, 4]]]), 0, -1),
        )

        # Mocking the load_mask function to return a list of Mask objects
        mock_mask = {"a": np.ndarray((30, 30))}
        mocker.patch(PATCH_LOAD_MASK, return_value=mock_mask)

        result = load_patient_scan(gen_path())

        # Assertions
        assert result["patient_id"] == "12345"
        assert isinstance(result["volume"], np.ndarray)
        assert result["volume"].shape == (2, 2, 1)
        assert result["masks"] == mock_mask

    def test_loads_patient_scan_no_masks(self, mocker):
        # Mocking the list_files function to return a list of DICOM file paths
        mocker.patch(
            PATCH_LIST_FILES,
            return_value=["file1.dcm", "file2.dcm"],
        )

        mocker.patch(PATCH_DCMREAD, return_value=MOCK_DICOM)

        # Mocking the load_volume function to return a numpy array
        mocker.patch(
            PATCH_LOAD_VOLUME,
            return_value=np.moveaxis(np.array([[[1, 2], [3, 4]]]), 0, -1),
        )

        # Mocking the load_mask function to return a list of Mask objects
        mock_mask = {}
        mocker.patch(PATCH_LOAD_MASK, return_value=mock_mask)

        result = load_patient_scan(gen_path())

        assert result is None

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

        mocker.patch(PATCH_DCMREAD, return_value=MOCK_DICOM)

        # Mocking the load_volume function to return a numpy array
        mocker.patch(
            PATCH_LOAD_VOLUME,
            return_value=np.moveaxis(np.array([[[1, 2], [3, 4]]]), 0, -1),
        )

        mocker.patch(PATCH_LOAD_MASK, return_value=None)
        result = load_patient_scan(gen_path())

        # Assertions
        assert result is None


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

        mocker.patch(PATCH_DCMREAD, return_value=MOCK_DICOM)

        mock_volume = np.random.randint(0, 2, (512, 512, 4))
        # Mocking the load_volume function to return a numpy array
        mocker.patch(
            PATCH_LOAD_VOLUME,
            return_value=mock_volume,
        )

        mock_mask = {
            "organ_1": np.random.randint(0, 2, (512, 512, 4)),
            "organ_2": np.random.randint(0, 2, (512, 512, 4)),
        }
        # Mocking the load_mask function to return None since no RT struct data is found
        mocker.patch(PATCH_LOAD_MASK, return_value=mock_mask)

        # Call the function under test
        result = list(load_all_patient_scans(gen_path()))

        # Assertions
        assert len(result) == 2
        assert all(scan["patient_id"] == "12345" for scan in result)
        np.testing.assert_array_equal(result[0]["volume"], mock_volume)
        np.testing.assert_array_equal(result[1]["volume"], mock_volume)
        assert all(
            [mask == mock_mask for mask in scan["masks"].values()] for scan in result
        )
        assert all(["organ_1" in scan["masks"].keys() for scan in result])
        assert all(["organ_2" in scan["masks"].keys() for scan in result])

    # Handles an empty directory gracefully
    def test_handles_empty_directory_gracefully(self, mocker):
        # Mock the os.listdir to return an empty list
        mocker.patch(PATCH_LISTDIR, return_value=[])

        # Call the function under test
        result = list(load_all_patient_scans(gen_path()))

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

        mocker.patch(PATCH_DCMREAD, return_value=MOCK_DICOM)

        mock_rt_struct.get_roi_mask_by_name.return_value = mock_mask
        mocker.patch(
            PATCH_LOAD_RT_STRUCTS,
            return_value=[mock_rt_struct],
        )
        mocker.patch(PATCH_GET_DICOM_SLICES, return_value=[MOCK_DICOM, MOCK_DICOM])

        result = list(load_all_masks(dicom_collection_path))

        mock_mask = np.flip(mock_mask, axis=1)
        mock_mask = np.flip(mock_mask, axis=2)

        assert len(result) == 2
        assert list(result[0].keys()) == ["organ_1", "organ_2"]
        assert list(result[1].keys()) == ["organ_1", "organ_2"]
        [
            np.testing.assert_array_equal(mask[organ], mock_mask)
            for mask in result
            for organ in mask.keys()
        ]

    # Directory contains no DICOM files
    def test_no_dicom_files_in_directory(self, mocker):
        dicom_collection_path = gen_path()
        mocker.patch(PATCH_LISTDIR, return_value=[])

        result = list(load_all_masks(dicom_collection_path))

        assert len(result) == 0


class TestLoadRoiNames:

    # Returns correct ROI names from single RT struct file
    def test_single_rt_struct_file(self, mocker, tmp_path):
        # Create mock RT struct with ROI names
        rt_struct = mocker.Mock()
        rt_struct.get_roi_names.return_value = ["ROI1", "ROI2", "ROI3"]

        mocker.patch(PATCH_LOAD_RT_STRUCTS, return_value=[rt_struct])
        mocker.patch(PATCH_GENERATE_FULL_PATHS, return_value=lambda _: [tmp_path])

        # Call function
        result = next(load_roi_names(str(tmp_path)))

        # Verify results
        assert result == ["ROI1", "ROI2", "ROI3"]

    # test multiple RT struct files
    def test_multiple_rt_struct_files(self, tmp_path, mocker):
        expected = [
            ["ROI1", "ROI2", "ROI2", "ROI3"],
            ["ROI4", "ROI5", "ROI6", "ROI1"],
            ["ROI7", "ROI8", "ROI2", "ROI1"],
        ]
        rt_struct = mocker.Mock()
        rt_struct.get_roi_names.return_value = expected[0]
        rt_struct2 = mocker.Mock()
        rt_struct2.get_roi_names.return_value = expected[1]
        rt_struct3 = mocker.Mock()
        rt_struct3.get_roi_names.return_value = expected[2]

        # Mock _load_rt_structs to return single RT struct
        mocker.patch(
            PATCH_LOAD_RT_STRUCTS,
            side_effect=[[rt_struct], [rt_struct2], [rt_struct3]],
        )
        mocker.patch(
            PATCH_GENERATE_FULL_PATHS,
            return_value=lambda _: [tmp_path, tmp_path, tmp_path],
        )

        # Call function
        result = list(load_roi_names(str(tmp_path)))

        # Verify results
        assert result == expected
