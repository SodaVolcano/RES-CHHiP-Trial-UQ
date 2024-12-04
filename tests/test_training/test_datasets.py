import tempfile
from datetime import date

import numpy as np

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


# class TestRandomPatchDataset:

#     # Dataset correctly samples patches with specified batch size and patch dimensions
#     def test_patch_sampling_dimensions(self):
#         # Setup test data
#         batch_size = 4
#         patch_size = (16, 16, 16)
#         h5_path = "test.h5"

#         # Create dataset
#         dataset = RandomPatchDataset(
#             h5_path=h5_path,
#             indices=None,
#             batch_size=batch_size,
#             patch_size=patch_size,
#             foreground_oversample_ratio=0.5,
#         )

#         # Get first batch
#         dataloader = DataLoader(dataset, batch_size=batch_size)
#         batch = next(iter(dataloader))
#         x, y = batch

#         # Check dimensions
#         assert x.shape[0] == batch_size
#         assert x.shape[2:] == patch_size
#         assert y.shape[0] == batch_size
#         assert y.shape[2:] == patch_size

#     # Foreground oversampling ratio produces expected number of foreground patches per batch
#     def test_foreground_oversampling_ratio(self):
#         import torch

#         from uncertainty.training.datasets import RandomPatchDataset

#         # Mock H5Dataset to return a fixed set of data
#         class MockH5Dataset:
#             def __init__(self, h5_path, indices):
#                 self.indices = [0, 1, 2]
#                 self.data = [
#                     (torch.zeros((1, 10, 10, 10)), torch.zeros((1, 10, 10, 10))),
#                     (torch.zeros((1, 10, 10, 10)), torch.ones((1, 10, 10, 10))),
#                     (torch.zeros((1, 10, 10, 10)), torch.zeros((1, 10, 10, 10))),
#                 ]

#             def __getitem__(self, index):
#                 return self.data[index]

#         # Patch the H5Dataset class in RandomPatchDataset
#         RandomPatchDataset.H5Dataset = MockH5Dataset

#         # Initialize RandomPatchDataset with a foreground oversample ratio
#         batch_size = 4
#         fg_ratio = 0.5
#         dataset = RandomPatchDataset(
#             h5_path="mock_path",
#             indices=None,
#             batch_size=batch_size,
#             patch_size=(5, 5, 5),
#             foreground_oversample_ratio=fg_ratio,
#         )

#         # Get a batch of patches
#         batch = next(iter(dataset))

#         # Count the number of foreground patches in the batch
#         fg_count = sum(torch.any(y) for _, y in batch)

#         # Calculate expected number of foreground patches
#         expected_fg_count = max(1, int(round(batch_size * fg_ratio)))

#         # Assert the number of foreground patches is as expected
#         assert fg_count == expected_fg_count

#     # Data augmentation transforms are applied correctly to batches
#     def test_data_augmentation_transforms(self):
#         import torch
#         from kornia.augmentation import RandomAffine3D

#         from uncertainty.training.datasets import RandomPatchDataset

#         # Mock data and transform
#         mock_data = torch.rand(
#             (2, 3, 32, 32, 32)
#         )  # Example shape (batch, channels, depth, height, width)
#         mock_labels = torch.randint(0, 2, (2, 1, 32, 32, 32))  # Binary labels
#         transform = RandomAffine3D(
#             5, align_corners=True, shears=0, scale=(0.9, 1.1), p=0.15
#         )

#         # Initialize dataset with mock data and transform
#         dataset = RandomPatchDataset(
#             h5_path="mock_path",
#             indices=None,
#             batch_size=2,
#             patch_size=(16, 16, 16),
#             foreground_oversample_ratio=0.5,
#             transform=lambda x_y: (transform(x_y[0]), x_y[1]),
#         )

#         # Mock the dataset loading
#         dataset.dataset = [(mock_data[i], mock_labels[i]) for i in range(2)]

#         # Get a batch and apply the transform
#         batch = next(iter(dataset))
#         transformed_data, transformed_labels = zip(*batch)

#         # Check if the transform was applied correctly
#         assert all(
#             isinstance(x, torch.Tensor) for x in transformed_data
#         ), "Data should be tensors"
#         assert all(
#             isinstance(y, torch.Tensor) for y in transformed_labels
#         ), "Labels should be tensors"
#         assert transformed_data[0].shape == (
#             3,
#             16,
#             16,
#             16,
#         ), "Transformed data shape mismatch"
#         assert transformed_labels[0].shape == (
#             1,
#             16,
#             16,
#             16,
#         ), "Transformed labels shape mismatch"

#     # Test that the batches of data are not repeated
#     def test_data_stream_does_not_repeat(self):
#         pass
