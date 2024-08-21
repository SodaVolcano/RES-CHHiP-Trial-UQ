import numpy as np

from tests.test_dicom import PATCH_LIST_FILES
from uncertainty.data.mask import get_organ_names

from .context import gen_path, uncertainty

# Import aliases
Mask = uncertainty.data.mask.Mask
PatientScan = uncertainty.data.patient_scan.PatientScan
load_patient_scan = uncertainty.data.nifti.load_patient_scan
load_patient_scans = uncertainty.data.nifti.load_patient_scans
load_mask = uncertainty.data.nifti.load_mask

# Patch aliases
PATCH_LOAD_VOLUME = "uncertainty.data.nifti.load_volume"
PATCH_LOAD_MASK_MULTIPLE_OBSERVERS = (
    "uncertainty.data.nifti.load_mask_multiple_observers"
)
PATCH_LOAD_VOLUME = "uncertainty.data.nifti.load_volume"
PATCH_RESOLVE_PATH_PLACEHOLDERS = "uncertainty.data.nifti.resolve_path_placeholders"
PATCH_PLACEHOLDER_MATCHES = "uncertainty.data.nifti.placeholder_matches"
PATCH_LOAD_PATIENT_SCAN = "uncertainty.data.nifti.load_patient_scan"
PATCH_NIBABEL_LOAD = "nibabel.load"
PATCH_LIST_FILES = "uncertainty.utils.path.list_files"


class TestLoadPatientScan:

    # Correctly loads volume and mask for a valid patient ID
    def test_correctly_loads_volume_and_mask_for_valid_patient_id(self, mocker):
        # Arrange
        base_path = gen_path()
        volume_path_pattern = base_path + "/{patient_id}.nii"
        mask_path_pattern = base_path + "/{patient_id}/{organ}/{observer}.nii"
        patient_id = "12345"
        mock_volume = np.zeros((10, 10, 10))

        mock_mask = [Mask(np.zeros((10, 10, 10)))]

        mocker.patch(
            PATCH_LOAD_VOLUME,
            return_value=mock_volume,
        )
        mocker.patch(
            PATCH_LOAD_MASK_MULTIPLE_OBSERVERS,
            return_value=mock_mask,
        )

        # Act
        patient_scan = load_patient_scan(
            volume_path_pattern, mask_path_pattern, patient_id
        )

        # Assert
        assert patient_scan.patient_id == patient_id
        assert np.array_equal(patient_scan.volume, mock_volume)
        assert patient_scan.masks[0] == mock_mask

    # Validates the structure and data types of the returned PatientScan object
    def test_correctly_loads_volume_and_mask_for_valid_patient_id(self, mocker):
        base_path = gen_path()
        volume_path_pattern = base_path + "/{patient_id}.nii"
        mask_path_pattern = base_path + "/{patient_id}/{organ}/{observer}.nii"
        patient_id = "12345"
        mock_volume = np.zeros((10, 10, 10))

        mock_mask = [Mask(np.zeros((10, 10, 10)))]

        mocker.patch(
            PATCH_LOAD_VOLUME,
            return_value=mock_volume,
        )
        mocker.patch(
            PATCH_LOAD_MASK_MULTIPLE_OBSERVERS,
            return_value=mock_mask,
        )

        patient_scan = load_patient_scan(
            volume_path_pattern, mask_path_pattern, patient_id
        )

        assert patient_scan.patient_id == patient_id
        assert np.array_equal(patient_scan.volume, mock_volume)
        assert patient_scan.masks[mock_mask[0].observer] == mock_mask[0]


class TestLoadPatientScans:

    # Correctly loads a PatientScan when given valid volume and mask path patterns
    def test_correctly_loads_patient_scan_with_valid_paths(self, mocker):
        base_path = gen_path()
        # Mock dependencies
        mocker.patch(
            PATCH_LIST_FILES,
            return_value=[
                f"{base_path}/1_CT.nii.gz",
                f"{base_path}/1_CT_bladder_Alice.nii.gz",
                f"{base_path}/2_CT.nii.gz",
                f"{base_path}/2_CT_bladder_Alice.nii.gz",
            ],
        )

        mock_volume = np.zeros((10, 10, 10))
        mock_mask = [Mask({"bladder": np.zeros((10, 10, 10))}, "Alice")]

        mocker.patch(
            PATCH_LOAD_VOLUME,
            return_value=mock_volume,
        )
        mocker.patch(
            PATCH_LOAD_MASK_MULTIPLE_OBSERVERS,
            return_value=mock_mask,
        )

        # Initialize and invoke the load_patient_scans function
        patient_scans = load_patient_scans(
            volume_path_pattern=base_path + "/{patient_id}_CT.nii.gz",
            mask_path_pattern=base_path + "/{patient_id}_CT_{organ}_{observer}.nii.gz",
        )

        # Iterate over the generator to load patient scans
        result = list(patient_scans)

        # Assertions
        assert len(result) == 2
        assert result[0].patient_id == "1"
        assert result[1].patient_id == "2"
        assert all(np.all(result[i].volume == mock_volume) for i in range(2))
        assert all(
            mask_name == "Alice" for i in range(2) for mask_name in result[i].masks
        )
        assert all(
            [
                np.all(mask == mock_mask[0])
                for i in range(2)
                for mask in result[i].masks.values()
            ]
        )
        assert all(
            get_organ_names(mask) == ["bladder"]
            for i in range(2)
            for mask in result[i].masks.values()
        )

    # Handles missing volume files gracefully
    def test_handles_missing_volume_files_gracefully(self, mocker):
        base_path = gen_path()
        # Mock dependencies
        mocker.patch(
            PATCH_LIST_FILES,
            return_value=[
                f"{base_path}/a_awd__dwCT.nii.gz",
                f"{base_path}/wad_wadCdwTad_wbdlwaadder_Alice.nii.gz",
            ],
        )

        # Initialize and invoke the load_patient_scans function
        patient_scans = load_patient_scans(
            volume_path_pattern=base_path + "/{patient_id}_CT.nii.gz",
            mask_path_pattern=base_path + "/{patient_id}_CT_{organ}_{observer}.nii.gz",
        )

        # Iterate over the generator to load patient scans
        result = list(patient_scans)

        # Assertions
        assert len(result) == 0


# Generated by CodiumAI

import pytest


class TestLoadMask:

    # Correctly loads mask when given valid mask_path_pattern with organ placeholders
    def test_loads_mask_with_valid_path_pattern(self, mocker):
        base_path = gen_path()
        mask_path_pattern = base_path + "/1_CT_{organ}_Alice.nii.gz"
        observer = "Alice"
        mocker.patch(
            PATCH_LIST_FILES,
            return_value=[
                f"{base_path}/1_CT.nii.gz",
                f"{base_path}/1_CT_bladder_{observer}.nii.gz",
                f"{base_path}/1_CT_eyes_{observer}.nii.gz",
                f"{base_path}/1_CT_heart_{observer}.nii.gz",
            ],
        )

        mock_nib_load = mocker.patch(PATCH_NIBABEL_LOAD)
        mock_nib_load.return_value.get_fdata.return_value = np.array([1, 2, 3])

        mask = load_mask(mask_path_pattern, observer)

        assert all(
            organ in get_organ_names(mask) for organ in ["bladder", "eyes", "heart"]
        )
        assert mask.observer == observer

    # Raises assertion error when mask_path_pattern does not contain {organ} placeholder
    def test_raises_assertion_error_without_organ_placeholder(self):
        base_path = gen_path()
        mask_path_pattern = f"{base_path}/mask.nii.gz"
        observer = "Alice"

        with pytest.raises(
            AssertionError, match="mask_path_pattern must contain {organ}"
        ):
            load_mask(mask_path_pattern, observer)

    # Handles mask_path_pattern with no matching files
    def test_handles_no_matching_files(self, mocker):
        base_path = gen_path()
        mask_path_pattern = base_path + "/{organ}_mask.nii.gz"
        observer = "Alice"

        mocker.patch(
            PATCH_LIST_FILES,
            return_value=[
                f"{base_path}/1_CT.nii.gz",
                f"{base_path}/1_CT_bladder_{observer}.nii.gz",
                f"{base_path}/1_CT_eyes_{observer}.nii.gz",
                f"{base_path}/1_CT_heart_{observer}.nii.gz",
            ],
        )

        mask = load_mask(mask_path_pattern, observer)

        assert get_organ_names(mask) == []
        assert mask.observer == observer
