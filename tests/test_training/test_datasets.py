import tempfile
from datetime import date

import numpy as np
import torch

from ..context import data, training

H5Dataset = training.H5Dataset
save_scans_to_h5 = data.save_scans_to_h5
RandomPatchDataset = training.RandomPatchDataset


def get_dataset(n: int = 2):
    np.random.seed(42)
    return [
        {
            "patient_id": i + 5,
            "volume": np.random.rand(1, 20 + i, 15, 10 + i),
            "dimension_original": (10, 10, 10),
            "spacings": (1.0, 1.0, 1.0),
            "modality": "CT",
            "manufacturer": "GE",
            "scanner": "Optima",
            "study_date": date(2021, 1, 1),
            "masks": np.random.randint(0, 2, (3, 20 + i, 15, 10 + i)),
        }
        for i in range(n)
    ]


class TestH5Dataset:

    # Dataset correctly loads (x,y) pairs from valid H5 file using integer index
    def test_load_xy_pairs_with_integer_index(self):
        with tempfile.NamedTemporaryFile() as tmp:
            test_file = tmp.name
            data = get_dataset()
            save_scans_to_h5(data, test_file)

            # Initialize dataset
            dataset = H5Dataset(test_file)

            assert len(dataset) == 2
            # Get item using integer index
            for i, (x, y) in enumerate(dataset):
                assert np.equal(x, data[i]["volume"]).all()
                assert np.equal(y, data[i]["masks"]).all()

    # Dataset correctly loads (x,y) pairs from valid H5 file using string index
    def test_indices_filters_dataset(self):
        with tempfile.NamedTemporaryFile() as tmp:
            test_file = tmp.name
            data = get_dataset(10)
            save_scans_to_h5(data, test_file)

            indices = [str(num + 5) for num in [0, 1, 3, 5, 7, 9]]
            # Initialize dataset
            dataset = H5Dataset(test_file, indices=indices)

            assert len(dataset) == 6
            # Get item using string index
            for i in indices:
                x, y = dataset[i]
                assert np.equal(x, data[int(i) - 5]["volume"]).all()
                assert np.equal(y, data[int(i) - 5]["masks"]).all()


class TestRandomPatchDataset:

    # Dataset correctly samples patches with specified batch size and patch dimensions
    def test_patch_sampling_dimensions(self):
        with tempfile.NamedTemporaryFile() as tmp:
            test_file = tmp.name
            data = get_dataset(10)
            save_scans_to_h5(data, test_file)

            dataset = RandomPatchDataset(
                h5_path=test_file,
                indices=[str(data[i]["patient_id"]) for i in [0, 5, 2, 3]],
                patch_size=(4, 4, 4),
                batch_size=3,
                foreground_oversample_ratio=0.5,
            )
            it = iter(dataset)

            buffer = []
            for _ in range(9):
                x, y = next(it)
                buffer.append((x, y))
                assert x.shape == (1, 4, 4, 4)
                assert y.shape == (3, 4, 4, 4)

            # Check that the same patch is not repeated
            for i in range(len(buffer)):
                for j in range(i + 1, len(buffer)):
                    assert not torch.all(buffer[i][0] == buffer[j][0])

    # Foreground oversampling ratio produces expected number of foreground patches per batch
    def test_foreground_oversampling_ratio(self):
        np.random.seed(42)
        torch.manual_seed(42)
        with tempfile.NamedTemporaryFile() as tmp:
            test_file = tmp.name
            data = get_dataset(10)
            # set some masks to 0
            data[0]["masks"] = np.zeros_like(data[0]["masks"])
            data[2]["masks"] = np.zeros_like(data[2]["masks"])
            data[3]["masks"] = np.zeros_like(data[3]["masks"])
            # ensure data[5] have at least one foreground
            data[5]["masks"][0, 0, 0, 0] = 1
            save_scans_to_h5(data, test_file)

            dataset = RandomPatchDataset(
                h5_path=test_file,
                indices=[str(data[i]["patient_id"]) for i in [0, 5, 2, 3]],
                patch_size=(4, 4, 4),
                batch_size=3,
                foreground_oversample_ratio=0.5,
            )
            it = iter(dataset)

            buffer = []
            for _ in range(9):
                x, y = next(it)
                buffer.append((x, y))

            # Check that the same patch is not repeated
            for i in range(len(buffer)):
                for j in range(i + 1, len(buffer)):
                    assert not torch.all(buffer[i][0] == buffer[j][0])

            # Check that every 3 patches in buffer, at least 2 have foreground
            for i in range(0, len(buffer), 3):
                count = 0
                for j in range(i, i + 3):
                    if torch.any(buffer[j][1]):
                        count += 1
                assert count >= 2

    # Foreground oversampling ratio produces expected number of foreground patches per batch
    def test_foreground_oversampling_ratio2(self):
        np.random.seed(42)
        torch.manual_seed(42)
        with tempfile.NamedTemporaryFile() as tmp:
            test_file = tmp.name
            data = get_dataset(10)
            # set some masks to 0
            data[0]["masks"] = np.zeros_like(data[0]["masks"])
            data[2]["masks"] = np.zeros_like(data[2]["masks"])
            data[3]["masks"] = np.zeros_like(data[3]["masks"])
            data[4]["masks"] = np.zeros_like(data[4]["masks"])
            # ensure data[5] have at least one foreground
            data[5]["masks"][0, 0, 0, 0] = 1
            save_scans_to_h5(data, test_file)

            dataset = RandomPatchDataset(
                h5_path=test_file,
                indices=[str(data[i]["patient_id"]) for i in [0, 5, 2, 3, 4]],
                patch_size=(4, 4, 4),
                batch_size=10,
                foreground_oversample_ratio=0.33,
            )
            it = iter(dataset)

            buffer = []
            for _ in range(30):
                x, y = next(it)
                buffer.append((x, y))

            # Check that the same patch is not repeated
            for i in range(len(buffer)):
                for j in range(i + 1, len(buffer)):
                    assert not torch.all(buffer[i][0] == buffer[j][0])

            # Check that every 3 patches in buffer, at least 2 have foreground
            for i in range(0, len(buffer), 10):
                count = 0
                for j in range(i, i + 10):
                    if torch.any(buffer[j][1]):
                        count += 1
                assert count >= 3

    # Test that augmentation parameter works
    def test_augmentation_function(self):
        with tempfile.NamedTemporaryFile() as tmp:
            test_file = tmp.name
            data = get_dataset(10)
            save_scans_to_h5(data, test_file)
            aug = lambda x_y: (torch.ones_like(x_y[0]), torch.zeros_like(x_y[1]))

            dataset = RandomPatchDataset(
                h5_path=test_file,
                indices=None,
                patch_size=(4, 4, 4),
                batch_size=3,
                foreground_oversample_ratio=0.5,
                transform=aug,
            )
            it = iter(dataset)

            for _, (x, y) in zip(range(10), it):
                assert torch.equal(x, torch.ones_like(x))
                assert torch.equal(y, torch.zeros_like(y))

    # Test that batch augmentation parameter works
    def test_batch_augmentation_function(self):
        with tempfile.NamedTemporaryFile() as tmp:
            test_file = tmp.name
            data = get_dataset(10)
            save_scans_to_h5(data, test_file)
            aug = lambda x_y: (torch.ones_like(x_y[0]), torch.zeros_like(x_y[1]))

            dataset = RandomPatchDataset(
                h5_path=test_file,
                indices=None,
                patch_size=(4, 4, 4),
                batch_size=3,
                foreground_oversample_ratio=0.5,
                batch_transform=aug,
            )
            it = iter(dataset)

            for _, (x, y) in zip(range(10), it):
                assert torch.equal(x, torch.ones_like(x))
                assert torch.equal(y, torch.zeros_like(y))
