import tempfile
import numpy as np
from datetime import date
import h5py
import os

from ..context import data

save_scans_to_h5 = data.save_scans_to_h5
load_scans_from_h5 = data.load_scans_from_h5

dataset = [
    {
        "patient_id": 5,
        "volume": np.zeros((10, 10, 10)),
        "dimension_original": (10, 10, 10),
        "spacings": (1.0, 1.0, 1.0),
        "modality": "CT",
        "manufacturer": "GE",
        "scanner": "Optima",
        "study_date": date(2021, 1, 1),
        "masks": {
            "organ1": np.ones((10, 10, 10)),
            "organ2": np.zeros((10, 10, 10)),
        },
    },
    {
        "patient_id": 6,
        "volume": np.ones((10, 10, 10)),
        "dimension_original": (10, 10, 10),
        "spacings": (1.0, 1.0, 1.0),
        "modality": "CT",
        "manufacturer": "GE",
        "scanner": "Optima",
        "study_date": date(2021, 1, 1),
        "masks": {
            "organ1": np.zeros((10, 10, 10)),
            "organ2": np.ones((10, 10, 10)),
        },
    },
]


class TestSaveScansToH5:

    # Successfully saves single PatientScan with all fields to H5 file
    def test_save_patient_scans(self):
        with tempfile.NamedTemporaryFile() as tmp:
            test_path = tmp.name
            # Save scan
            save_scans_to_h5(dataset, test_path)

            # Verify file exists and contents
            assert os.path.exists(test_path)

            with h5py.File(test_path, "r") as f:
                for i in range(2):
                    patient = f[str(i + 5)]
                    assert patient["patient_id"][()] == dataset[i]["patient_id"]  # type: ignore
                    assert np.array_equal(patient["volume"][()], dataset[i]["volume"])  # type: ignore
                    assert patient["modality"][()].decode() == "CT"  # type: ignore
                    assert np.array_equal(patient["masks/organ1"][()], dataset[i]["masks"]["organ1"])  # type: ignore


class TestLoadScansFromH5:

    # Successfully saves single PatientScan with all fields to H5 file
    def test_load_patient_scans(self):
        with tempfile.NamedTemporaryFile() as tmp:
            test_path = tmp.name
            # Save scan
            save_scans_to_h5(dataset, test_path)
            loaded = list(load_scans_from_h5(test_path))
            assert len(loaded) == 2
            for i, data in enumerate(loaded):
                assert data["patient_id"] == dataset[i]["patient_id"]
                assert np.array_equal(data["volume"], dataset[i]["volume"])
                assert data["modality"] == "CT"
                assert data["masks"]["organ1"].shape == (10, 10, 10)
                assert np.array_equal(
                    data["masks"]["organ1"], dataset[i]["masks"]["organ1"]
                )
                assert np.array_equal(
                    data["masks"]["organ2"], dataset[i]["masks"]["organ2"]
                )
                assert data["study_date"] == date(2021, 1, 1)
                assert data["dimension_original"] == (10, 10, 10)
                assert data["spacings"] == (1.0, 1.0, 1.0)
                assert data["manufacturer"] == "GE"
                assert data["scanner"] == "Optima"
