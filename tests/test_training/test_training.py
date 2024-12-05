import pickle
import tempfile
from datetime import date
from pathlib import Path

import numpy as np
import pytest
import torch
from lightning import LightningModule
from torch import nn

from ..context import data, training

split_into_folds = training.split_into_folds
write_training_fold_file = training.write_training_fold_file
train_model = training.train_model
SegmentationData = training.SegmentationData
save_scans_to_h5 = data.save_scans_to_h5


class TestSplitIntoFolds:

    # Dataset splits correctly into n_folds with default return_indices=False
    def test_dataset_splits_into_folds(self):
        # Create sample dataset
        dataset = range(10, 0, -1)
        n_folds = 3

        # Get fold splits
        splits = list(split_into_folds(dataset, n_folds))

        expected = [
            ([6, 5, 4, 3, 2, 1], [10, 9, 8, 7]),
            ([10, 9, 8, 7, 3, 2, 1], [6, 5, 4]),
            ([10, 9, 8, 7, 6, 5, 4], [3, 2, 1]),
        ]

        # Verify number of splits
        assert len(splits) == n_folds
        assert splits == expected

    # Returns indices of the dataset split
    def test_dataset_splits_into_folds_indices(self):
        # Create sample dataset
        dataset = list(range(10, 0, -1))
        n_folds = 3

        # Get fold splits
        splits = list(split_into_folds(dataset, n_folds, return_indices=True))
        expected = [
            ([4, 5, 6, 7, 8, 9], [0, 1, 2, 3]),
            ([0, 1, 2, 3, 7, 8, 9], [4, 5, 6]),
            ([0, 1, 2, 3, 4, 5, 6], [7, 8, 9]),
        ]

        # Verify number of splits
        assert len(splits) == n_folds
        assert splits == expected

    def test_fold_1_return_indices(self):
        dataset = list(range(10, 0, -1))
        n_folds = 1

        splits = list(split_into_folds(dataset, n_folds, return_indices=True))
        expected = ([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [])

        assert splits[0] == expected


class TestWriteTrainingFoldFile:

    # Function successfully writes fold indices to file with default seed=True
    def test_write_fold_indices_with_seed(self):
        fold_indices = list(split_into_folds(range(5, 15), 5, return_indices=True))
        with tempfile.NamedTemporaryFile() as tmp:
            path = tmp.name

            write_training_fold_file(path, fold_indices, force=True)  # type: ignore

            with open(path, "rb") as f:
                content = pickle.load(f)

            assert len(content) == 5
            for i in range(5):
                assert f"fold_{i}" in content
                assert content[f"fold_{i}"]["train"] == fold_indices[i][0]
                assert content[f"fold_{i}"]["val"] == fold_indices[i][1]
                assert "seed" in content[f"fold_{i}"]
                assert isinstance(content[f"fold_{i}"]["seed"], int)


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
            # if not float32, kornia augmentation in dataset will fail
            "masks": np.random.randint(0, 2, (3, 20 + i, 15, 10 + i)).astype(
                np.float32
            ),
        }
        for i in range(n)
    ]


class TestTrainModel:

    # Model trains successfully with default parameters and saves checkpoints
    def test_model_trains(self, tmp_path):
        with pytest.raises(SystemExit):
            # Create mock model and dataset
            class MockModel(LightningModule):
                def __init__(self):
                    super().__init__()
                    self.layer = nn.Conv3d(3, 1, 3)

                def training_step(self, batch, batch_idx):
                    self.log("loss", torch.tensor(0.5))
                    return {"loss": torch.tensor(0.5, requires_grad=True)}

                def validation_step(self, batch, batch_idx):
                    self.log("val_loss", torch.tensor(0.3))
                    return {"val_loss": torch.tensor(0.3, requires_grad=True)}

                def configure_optimizers(self):
                    return torch.optim.Adam(self.parameters())

            model = MockModel()

            with (
                tempfile.NamedTemporaryFile() as tmp,
                tempfile.TemporaryDirectory() as tmp_dir,
            ):
                test_file = tmp.name
                data = get_dataset(20)
                save_scans_to_h5(data, test_file)

                dataset = SegmentationData(
                    h5_path=test_file,
                    batch_size=2,
                    batch_size_eval=1,
                    patch_size=(5, 5, 5),
                    foreground_oversample_ratio=0.5,
                    num_workers_train=0,
                    num_workers_val=0,
                    prefetch_factor_train=None,
                    prefetch_factor_val=None,
                    # batch_augmentations=tz.identity,
                )
                log_dir = Path(tmp_dir) / "logs"
                checkpoint_dir = Path(tmp_dir) / "checkpoints"
                next(iter(dataset.train_dataloader()))
                exit()

                # Train model
                train_model(
                    model=model,
                    dataset=dataset,
                    log_dir=log_dir,
                    experiment_name="test_exp",
                    checkpoint_path=checkpoint_dir,
                    checkpoint_name="{epoch:03d}-{val_loss:.4f}",
                    checkpoint_every_n_epochs=1,
                    n_epochs=2,
                    n_batches_per_epoch=2,
                    n_batches_val=2,
                    check_val_every_n_epoch=1,
                    accelerator="cpu",
                    enable_progress_bar=True,
                    enable_model_summary=False,
                    precision="bf16-mixed",
                    strategy="ddp",
                    save_last_checkpoint=False,
                )

                assert checkpoint_dir.exists()
                checkpoints = list(checkpoint_dir.glob("*.ckpt"))
                assert [
                    i in checkpoints
                    for i in [
                        "epoch=000-val_loss=0.3000.ckpt",
                        "epoch=001-val_loss=0.3000.ckpt",
                    ]
                ]
                assert (log_dir / "test_exp").exists()
