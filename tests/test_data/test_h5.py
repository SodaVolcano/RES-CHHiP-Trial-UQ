import os
import tempfile
from datetime import date

import h5py
import numpy as np
import torch

from ..context import data

save_scans_to_h5 = data.save_scans_to_h5
load_scans_from_h5 = data.load_scans_from_h5
_create_group = data.h5._create_group
save_prediction_to_h5 = data.save_prediction_to_h5
save_predictions_to_h5 = data.save_predictions_to_h5

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


class TestCreateGroup:
    def test_create_group_with_different_datatypes(self, tmp_path):
        test_path = tmp_path / "test.h5"
        dataset2 = {
            "patient_id": 5,
            "volume": np.zeros((10, 10, 10)),
            "spacings": (1.0, 1.0, 1.0),
            "modality": "CT",
            "study_date": date(2021, 1, 1),
            "masks": {
                "organ1": np.ones((10, 10, 10)),
                "organ2": np.zeros((10, 10, 10)),
                "list": [
                    np.array([1, 2, 3]),
                    np.array([1.0, 2.0, 3.0]),
                    torch.tensor([1, 2, 3]),
                ],
            },
            "list": [1, 2, 3],
            "tensor": torch.tensor([1, 2, 3]),
            "list2": [
                np.array([1, 2, 3]),
                np.array([1.0, 2.0, 3.0]),
                torch.tensor([1, 2, 3]),
            ],
        }

        with h5py.File(test_path, "w") as f:
            _create_group(dataset2, "test_group", f)
            assert "test_group" in f
            assert f["test_group"]["patient_id"][()] == 5  # type: ignore
            assert np.array_equal(f["test_group"]["volume"][()], np.zeros((10, 10, 10)))  # type: ignore
            assert f["test_group"]["spacings"][()].tolist() == [1.0, 1.0, 1.0]  # type: ignore
            assert f["test_group"]["modality"][()].decode() == "CT"  # type: ignore
            assert f["test_group"]["study_date"][()].decode() == "2021-01-01"  # type: ignore
            assert np.array_equal(f["test_group"]["masks/organ1"][()], np.ones((10, 10, 10)))  # type: ignore
            assert np.array_equal(f["test_group"]["masks/organ2"][()], np.zeros((10, 10, 10)))  # type: ignore
            assert np.array_equal(f["test_group"]["masks/list"][()], np.array([[1.0, 2.0, 3.0] for _ in range(3)]))  # type: ignore
            assert np.array_equal(f["test_group"]["list"][()], np.array([1, 2, 3]))  # type: ignore
            assert np.array_equal(f["test_group"]["tensor"][()], np.array([1, 2, 3]))  # type: ignore
            assert np.array_equal(f["test_group"]["list2"][()], np.array([[1.0, 2.0, 3.0] for _ in range(3)]))  # type: ignore


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


class TestSavePredictionToH5:
    def test_save_prediction_to_h5(self, tmp_path):
        test_path = tmp_path / "test.h5"
        x = torch.ones((1, 10, 10, 10))
        y = torch.zeros((3, 10, 10, 10))
        y_pred = torch.ones((3, 10, 10, 10))
        save_prediction_to_h5(test_path, "test", x, y, y_pred)

        with h5py.File(test_path, "r") as f:
            assert "test" in f
            assert np.array_equal(f["test"]["x"][()], x)  # type: ignore
            assert np.array_equal(f["test"]["y"][()], y)  # type: ignore
            assert np.array_equal(f["test"]["y_pred"][()], y_pred)  # type: ignore

    def test_dont_save_x(self, tmp_path):
        test_path = tmp_path / "test.h5"
        x = None
        y = torch.zeros((3, 10, 10, 10))
        y_pred = torch.ones((3, 10, 10, 10))
        save_prediction_to_h5(test_path, "test", x, y, y_pred)

        with h5py.File(test_path, "r") as f:
            assert "test" in f
            assert "x" not in f["test"]  # type: ignore
            assert np.array_equal(f["test"]["y"][()], y)  # type: ignore
            assert np.array_equal(f["test"]["y_pred"][()], y_pred)  # type: ignore

    def test_appends_prediction_to_h5(self, tmp_path):
        test_path = tmp_path / "test.h5"
        x = torch.ones((1, 10, 10, 10))
        y = torch.zeros((3, 10, 10, 10))
        y_pred = torch.ones((3, 10, 10, 10))
        save_prediction_to_h5(test_path, "test", x, y, y_pred)
        save_prediction_to_h5(test_path, "test2", x, y, y_pred)

        with h5py.File(test_path, "r") as f:
            assert "test" in f
            assert "test2" in f

            assert np.array_equal(f["test"]["x"][()], x)  # type: ignore
            assert np.array_equal(f["test"]["y"][()], y)  # type: ignore
            assert np.array_equal(f["test"]["y_pred"][()], y_pred)  # type: ignore
            assert np.array_equal(f["test2"]["x"][()], x)  # type: ignore
            assert np.array_equal(f["test2"]["y"][()], y)  # type: ignore
            assert np.array_equal(f["test2"]["y_pred"][()], y_pred)  # type: ignore


class TestSavePredictionsToH5:
    def test_save_predictions_to_h5(self, tmp_path):
        test_path = tmp_path / "test.h5"
        x = torch.ones((1, 10, 10, 10))
        y = torch.zeros((3, 10, 10, 10))
        y_preds = [torch.ones((3, 10, 10, 10)) for _ in range(3)]
        save_predictions_to_h5(test_path, "test", x, y, y_preds)

        with h5py.File(test_path, "r") as f:
            assert "test" in f
            assert np.array_equal(f["test"]["x"][()], x)  # type: ignore
            assert np.array_equal(f["test"]["y"][()], y)  # type: ignore
            assert np.array_equal(f["test"]["y_preds"][()], y_preds)  # type: ignore
            assert f["test"]["probability_map"][()].shape == (3, 10, 10, 10)  # type: ignore
            assert f["test"]["variance_map"][()].shape == (3, 10, 10, 10)  # type: ignore
            assert f["test"]["entropy_map"][()].shape == (3, 10, 10, 10)  # type: ignore

    def test_appends_predictions_to_h5(self, tmp_path):
        test_path = tmp_path / "test.h5"
        x = torch.ones((1, 10, 10, 10))
        y = torch.zeros((3, 10, 10, 10))
        y_preds = [torch.ones((3, 10, 10, 10)) for _ in range(3)]
        save_predictions_to_h5(test_path, "test", x, y, y_preds)
        save_predictions_to_h5(test_path, "test2", x, y, y_preds)

        with h5py.File(test_path, "r") as f:
            assert "test" in f
            assert "test2" in f

            assert np.array_equal(f["test"]["x"][()], x)  # type: ignore
            assert np.array_equal(f["test"]["y"][()], y)  # type: ignore
            assert np.array_equal(f["test"]["y_preds"][()], y_preds)  # type: ignore
            assert np.array_equal(f["test2"]["x"][()], x)  # type: ignore
            assert np.array_equal(f["test2"]["y"][()], y)  # type: ignore
            assert np.array_equal(f["test2"]["y_preds"][()], y_preds)  # type: ignore
            assert f["test"]["probability_map"][()].shape == (3, 10, 10, 10)  # type: ignore
            assert f["test"]["variance_map"][()].shape == (3, 10, 10, 10)  # type: ignore
            assert f["test"]["entropy_map"][()].shape == (3, 10, 10, 10)  # type: ignore
            assert f["test2"]["probability_map"][()].shape == (3, 10, 10, 10)  # type: ignore
            assert f["test2"]["variance_map"][()].shape == (3, 10, 10, 10)  # type: ignore
            assert f["test2"]["entropy_map"][()].shape == (3, 10, 10, 10)  # type: ignore
