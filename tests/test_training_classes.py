import numpy as np
import torch
from .context import uncertainty

H5Dataset = uncertainty.training.classes.H5Dataset
PREPROCESS_DATA_CONFIGURABLE_PATCH = (
    "uncertainty.training.classes._preprocess_data_configurable"
)


class TestH5Dataset:
    # Loading dataset from a valid H5 file path
    def test_loading_dataset_from_valid_h5_file_path(self, mocker):
        mocker.patch(
            PREPROCESS_DATA_CONFIGURABLE_PATCH,
            return_value=lambda x: x,
        )

        # Mocking the h5py.File object
        mock_h5_file = mocker.patch("h5py.File", autospec=True)
        mock_h5_file.return_value.keys.return_value = ["entry1", "entry2"]
        x1, y1 = np.random.rand(1, 30, 24, 15), np.random.rand(3, 30, 24, 15)
        x2, y2 = np.random.rand(1, 30, 24, 15), np.random.rand(3, 30, 24, 15)
        mock_h5_file.return_value.__getitem__.side_effect = lambda key: {
            "entry1": {"x": x1, "y": y1},
            "entry2": {"x": x2, "y": y2},
        }[key]

        dataset = H5Dataset(h5_file_path="valid_path.h5")

        assert len(dataset) == 2
        assert torch.allclose(dataset[0][0], torch.tensor(x1, dtype=torch.float32))  # type: ignore
        assert torch.allclose(dataset[0][1], torch.tensor(y1, dtype=torch.float32))  # type: ignore
        assert torch.allclose(dataset[1][0], torch.tensor(x2, dtype=torch.float32))  # type: ignore
        assert torch.allclose(dataset[1][1], torch.tensor(y2, dtype=torch.float32))  # type: ignore

    def test_preprocesses_data(self, mocker):
        config = {
            "input_height": 20,
            "input_width": 20,
            "input_depth": 20,
            "intensity_range": (-1, 30),
        }

        # Mocking the h5py.File object
        mock_h5_file = mocker.patch("h5py.File", autospec=True)
        mock_h5_file.return_value.keys.return_value = ["entry1"]
        x1, y1 = np.random.rand(1, 30, 24, 15), np.random.rand(3, 30, 24, 15)
        mock_h5_file.return_value.__getitem__.side_effect = lambda key: {
            "entry1": {"x": x1, "y": y1},
        }[key]

        dataset = H5Dataset(h5_file_path="valid_path.h5", config=config)  # type: ignore

        assert len(dataset) == 1
        assert dataset[0][0].shape == (1, 20, 20, 20)  # type: ignore
        assert dataset[0][1].shape == (3, 20, 20, 20)  # type: ignore
        assert torch.all(dataset[0][0] >= -1)  # type: ignore
        assert torch.all(dataset[0][0] <= 30)  # type: ignore
