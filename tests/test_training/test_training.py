import os
import pickle
import tempfile
from datetime import date
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from lightning import LightningModule
from torch import nn

from ..context import chhip_uq, data, training

auto_match_config = chhip_uq.auto_match_config
H5Dataset = training.H5Dataset
split_into_folds = training.split_into_folds
write_fold_splits_file = training.write_fold_splits_file
read_fold_splits_file = training.read_fold_splits_file
train_model = training.train_model
train_models = training.train_models
SegmentationData = training.SegmentationData
save_scans_to_h5 = data.save_scans_to_h5
train_test_split = training.train_test_split
init_training_dir = training.init_training_dir
LitModel = training.LitModel
load_model = training.load_model
load_models = training.load_models
load_training_dir = training.load_training_dir
select_single_models = training.select_single_models
select_ensembles = training.select_ensembles


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


class MockLoss(nn.Module):
    def forward(self, x, y, logits):
        return torch.tensor(0.5, requires_grad=True)


class MockTorchModel(nn.Module):
    @auto_match_config(prefixes=["mock"])
    def __init__(self, loss, val_loss):
        super().__init__()
        self.layer = nn.Conv3d(3, 1, 3)
        self.loss = MockLoss()
        self.train_loss = loss
        self.val_loss = val_loss
        self.deep_supervision = False
        self.optimiser = torch.optim.Adam(self.parameters())
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimiser, 1)

    def forward(self, x, logits):
        return torch.rand((2, 3, 5, 5, 5), requires_grad=True)


class MockModel(LightningModule):
    @auto_match_config(prefixes=["mock"])
    def __init__(self, model):
        super().__init__()
        self.model = model

    def training_step(self, batch, batch_idx):
        self.log("loss", torch.tensor(self.model.train_loss))
        return {"loss": torch.tensor(self.model.train_loss, requires_grad=True)}

    def validation_step(self, batch, batch_idx):
        self.log("val_loss", torch.tensor(self.model.val_loss))
        return {"val_loss": torch.tensor(self.model.val_loss, requires_grad=True)}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())


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


class TestWriteFoldSplitsFile:

    # Function successfully writes fold indices to file
    def test_write_fold_indices(self):
        fold_indices = list(split_into_folds(range(5, 15), 5, return_indices=True))
        with tempfile.NamedTemporaryFile() as tmp:
            path = tmp.name

            write_fold_splits_file(path, fold_indices, force=True)  # type: ignore
            content = read_fold_splits_file(path)

            assert len(content) == 5
            for i in range(5):
                assert f"fold_{i}" in content
                assert content[f"fold_{i}"]["train"] == fold_indices[i][0]
                assert content[f"fold_{i}"]["val"] == fold_indices[i][1]


class TestTrainModel:

    # Model trains successfully with default parameters and saves checkpoints
    def test_model_trains(self, tmp_path):
        # Create mock model and dataset
        model_torch = MockTorchModel(0.5, 0.3)
        model = MockModel(model_torch)

        test_file = tmp_path / "test.h5"
        data = get_dataset(20)
        save_scans_to_h5(data, test_file)

        dataset = SegmentationData(
            h5_path=test_file,
            batch_size=2,
            batch_size_eval=2,
            patch_size=(5, 5, 5),
            foreground_oversample_ratio=0.5,
            num_workers_train=0,
            num_workers_val=0,
            prefetch_factor_train=None,
            prefetch_factor_val=None,
            # batch_augmentations=tz.identity,
        )
        log_dir = Path(tmp_path) / "logs"
        checkpoint_dir = Path(tmp_path) / "checkpoints"

        # Train model
        train_model(
            model=model,
            dataset=dataset,
            log_dir=log_dir,
            experiment_name="test_exp",
            checkpoint_path=checkpoint_dir,
            checkpoint_name="{epoch:03d}-{val_loss:.4f}",
            checkpoint_every_n_epoch=1,
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
            num_sanity_val_steps=0,
        )

        assert checkpoint_dir.exists()
        assert (checkpoint_dir / "torch-module.pt").exists()
        checkpoints = list(checkpoint_dir.glob("*.ckpt"))
        assert [
            i in checkpoints
            for i in [
                "epoch=000-val_loss=0.3000.ckpt",
                "epoch=001-val_loss=0.3000.ckpt",
            ]
        ]
        assert (log_dir / "test_exp").exists()

    # Model trains successfully with default parameters and saves checkpoints
    def test_multiple_call_produces_different_model_checkpoints(self, tmp_path):
        # Create mock model and dataset
        model_torch = MockTorchModel(0.5, 0.3)
        model = MockModel(model_torch)

        test_file = tmp_path / "test.h5"
        data = get_dataset(20)
        save_scans_to_h5(data, test_file)

        dataset = SegmentationData(
            h5_path=test_file,
            batch_size=2,
            batch_size_eval=2,
            patch_size=(5, 5, 5),
            foreground_oversample_ratio=0.5,
            num_workers_train=0,
            num_workers_val=0,
            prefetch_factor_train=None,
            prefetch_factor_val=None,
            # batch_augmentations=tz.identity,
        )
        log_dir = Path(tmp_path) / "logs"
        checkpoint_dir = Path(tmp_path) / "checkpoints"

        # Train models
        for _ in range(3):
            train_model(
                model=model,
                dataset=dataset,
                log_dir=log_dir,
                experiment_name="test_exp",
                checkpoint_path=checkpoint_dir,
                checkpoint_name="{epoch:03d}-{val_loss:.4f}",
                checkpoint_every_n_epoch=1,
                n_epochs=2,
                n_batches_per_epoch=2,
                n_batches_val=2,
                check_val_every_n_epoch=1,
                accelerator="cpu",
                enable_progress_bar=True,
                enable_model_summary=False,
                precision="bf16-mixed",
                strategy="ddp",
                save_last_checkpoint=False,  # throws an error for multiple callbacks if true...
                num_sanity_val_steps=2,
            )

        assert (log_dir / "test_exp").exists()
        for i in range(3):
            path = Path(f"{str(checkpoint_dir)}-{i}") if i > 0 else checkpoint_dir
            assert path.exists()
            assert (path / "torch-module.pt").exists()
            checkpoints = list(path.glob("*.ckpt"))
            assert [
                i in checkpoints
                for i in [
                    "epoch=000-val_loss=0.3000.ckpt",
                    "epoch=001-val_loss=0.3000.ckpt",
                ]
            ]


class TestTrainModels:

    # Train single model with valid model name and dataset
    def test_train_single_valid_model(self, tmp_path):
        def mock_get_model(_):
            return MockTorchModel

        with patch("chhip_uq.training.training.get_model", mock_get_model):
            with tempfile.NamedTemporaryFile() as tmp_file:
                # Create test dataset
                data = get_dataset(20)
                save_scans_to_h5(data, tmp_file.name)

                dataset = SegmentationData(
                    h5_path=tmp_file.name,
                    batch_size=2,
                    batch_size_eval=2,
                    patch_size=(5, 5, 5),
                    foreground_oversample_ratio=0.5,
                    num_workers_train=0,
                    num_workers_val=0,
                    prefetch_factor_train=None,
                    prefetch_factor_val=None,
                )
                checkpoint_dir = Path(tmp_path) / "checkpoints"
                # Train models
                train_models(
                    models=["unet"],
                    dataset=dataset,
                    checkpoint_dir=checkpoint_dir,
                    experiment_name="test_exp",
                    n_epochs=2,
                    n_batches_per_epoch=2,
                    n_batches_val=2,
                    check_val_every_n_epoch=1,
                    checkpoint_every_n_epoch=1,
                    accelerator="cpu",
                    precision="32",
                    strategy="ddp",
                    log_dir=tmp_path / "logs",
                    checkpoint_name="{epoch:03d}-{val_loss:.4f}",
                    save_last_checkpoint=False,
                    mock__loss=0.5,
                    mock__val_loss=0.3,
                    num_sanity_val_steps=2,
                )

                # Verify checkpoint directory exists
                assert (checkpoint_dir / "unet").exists()
                checkpoints = list((checkpoint_dir / "unet").glob("*.ckpt"))
                assert [
                    i in checkpoints
                    for i in [
                        "epoch=000-val_loss=0.3000.ckpt",
                        "epoch=001-val_loss=0.3000.ckpt",
                    ]
                ]

    # Train multiple models with quantity suffix (e.g. 'unet_3')
    def test_train_multiple_models_with_quantity_suffix(self, tmp_path):
        # Mock get_model to return MockModel
        def mock_get_model(_):
            return MockTorchModel

        # Patch get_model function
        with patch("chhip_uq.training.training.get_model", mock_get_model):
            test_file = tmp_path / "test.h5"
            data = get_dataset(20)
            save_scans_to_h5(data, test_file)

            dataset = SegmentationData(
                h5_path=test_file,
                batch_size=2,
                batch_size_eval=2,
                patch_size=(5, 5, 5),
                foreground_oversample_ratio=0.5,
                num_workers_train=0,
                num_workers_val=0,
                prefetch_factor_train=None,
                prefetch_factor_val=None,
            )
            checkpoint_dir = Path(tmp_path) / "checkpoints"
            experiment_name = "test_exp"

            # Train multiple models
            train_models(
                models=["unet_3"],
                dataset=dataset,
                checkpoint_dir=checkpoint_dir,
                experiment_name=experiment_name,
                seed=42,
                log_dir=Path(tmp_path) / "logs",
                checkpoint_name="{epoch:03d}-{val_loss:.4f}",
                checkpoint_every_n_epoch=1,
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
                mock__loss=0.5,
                mock__val_loss=0.3,
                num_sanity_val_steps=2,
            )

            # Check if checkpoints are created for each model
            for i in range(3):
                model_checkpoint_dir = checkpoint_dir / f"unet-{i}"
                assert model_checkpoint_dir.exists()
                checkpoints = list((checkpoint_dir / "unet-{i}").glob("*.ckpt"))
                assert [
                    i in checkpoints
                    for i in [
                        "epoch=000-val_loss=0.3000.ckpt",
                        "epoch=001-val_loss=0.3000.ckpt",
                    ]
                ]

    def test_train_different_model_type(self, tmp_path):
        def check_checkpoints(n_models, model_name):
            # Check if checkpoints are created for each model
            for i in range(n_models):
                model_checkpoint_dir = checkpoint_dir / f"{model_name}-{i}"
                if model_name == "wow":
                    model_checkpoint_dir = checkpoint_dir / f"wow"

                assert model_checkpoint_dir.exists()
                checkpoints = list((checkpoint_dir / "{model_name}-{i}").glob("*.ckpt"))
                assert [
                    i in checkpoints
                    for i in [
                        "epoch=000-val_loss=0.3000.ckpt",
                        "epoch=001-val_loss=0.3000.ckpt",
                    ]
                ]

        # Mock get_model to return MockModel
        def mock_get_model(_):
            return MockTorchModel

        # Patch get_model function
        with patch("chhip_uq.training.training.get_model", mock_get_model):
            test_file = tmp_path / "test.h5"
            data = get_dataset(20)
            save_scans_to_h5(data, test_file)

            dataset = SegmentationData(
                h5_path=test_file,
                batch_size=2,
                batch_size_eval=2,
                patch_size=(5, 5, 5),
                foreground_oversample_ratio=0.5,
                num_workers_train=0,
                num_workers_val=0,
                prefetch_factor_train=None,
                prefetch_factor_val=None,
            )
            checkpoint_dir = Path(tmp_path) / "checkpoints"
            experiment_name = "test_exp"

            # Train multiple models
            train_models(
                models=["unet_3", "notunet_2", "wow", "wow2_1"],
                dataset=dataset,
                checkpoint_dir=checkpoint_dir,
                experiment_name=experiment_name,
                seed=42,
                log_dir=Path(tmp_path) / "logs",
                checkpoint_name="{epoch:03d}-{val_loss:.4f}",
                checkpoint_every_n_epoch=1,
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
                mock__loss=0.5,
                mock__val_loss=0.3,
                num_sanity_val_steps=2,
            )

            check_checkpoints(3, "unet")
            check_checkpoints(2, "notunet")
            check_checkpoints(1, "wow")
            check_checkpoints(1, "wow2")


class TestTrainTestSplit:

    # Split dataset with test_split=0.2 returns two sequences of correct proportions
    def test_split_proportions(self):
        # Create test dataset
        dataset = list(range(100, 0, -1))
        test_split = 0.2

        # Split dataset
        train_data, test_data = train_test_split(
            dataset, test_split=test_split, seed=42
        )

        # Check proportions
        assert len(train_data) == 80
        assert len(test_data) == 20

        # Check no data overlap
        assert set(train_data).isdisjoint(set(test_data))

        # Check all data preserved
        assert set(train_data).union(set(test_data)) == set(dataset)


class TestInitTrainingDir:

    # Creates training directory and copies config file when directory doesn't exist
    def test_creates_new_training_dir(self, tmp_path):
        # Setup test data
        config_path = tmp_path / "configuration.yaml"
        config_path.write_text("test config")

        train_dir = tmp_path / "training"
        dataset = list(range(40))

        # Call function
        config_copy, train_test_path, folds_path, fold_dirs = init_training_dir(
            train_dir=train_dir,
            config_path=config_path,
            dataset_indices=dataset,
            n_folds=3,
            test_split=0.2,
        )  # type: ignore

        # Verify directory and files created
        assert train_dir.exists()
        assert (train_dir / "configuration.yaml").exists()
        assert (train_dir / "train-test-split.pkl").exists()
        assert (train_dir / "validation-fold-splits.pkl").exists()
        assert len(fold_dirs) == 3
        assert all(d.exists() for d in fold_dirs)
        assert all(os.listdir(d) == [] for d in fold_dirs)

        # test that training indices do not overlap with validation indices
        split_dict = read_fold_splits_file(folds_path)
        assert isinstance(split_dict, dict)
        with open(train_test_path, "rb") as f:
            train_indices, test_indices = pickle.load(f)

        assert set(train_indices).isdisjoint(set(test_indices))

        for fold in split_dict.values():
            assert set(fold["train"]).isdisjoint(set(fold["val"]))
            assert set(fold["train"]).isdisjoint(set(test_indices))
            assert set(fold["val"]).isdisjoint(set(test_indices))

            # check that indices are from original train_indices
            assert set(fold["train"]).union(set(fold["val"])) == set(train_indices)

    # Handles existing training directory without overwriting
    def test_handles_existing_dir(self, tmp_path):
        # Setup existing training dir
        train_dir = tmp_path / "training"
        train_dir.mkdir()

        config_content = "unet:\n  hello: world\ndata:\n  hauk: twah!\nlogger:\n  sink: loll\ntraining:\n  yes: no\nconfidnet:\n  bleh: blah"
        config_path = tmp_path / "config.yaml"
        config_path.write_text(config_content)

        # Create existing config file with different path
        existing_config = train_dir / "configuration.yaml"
        existing_config.write_text(config_content)

        dataset = list(range(10))

        # Call function with different config path
        result = init_training_dir(
            train_dir=train_dir,
            config_path=config_path,
            dataset_indices=dataset,
            n_folds=3,
            test_split=0.2,
        )
        # If configs are the same, proceed as normal
        assert result is not None

        existing_config.write_text(config_content + "\n  something: else")
        result = init_training_dir(
            train_dir=train_dir,
            config_path=config_path,
            dataset_indices=dataset,
            n_folds=3,
            test_split=0.2,
        )
        # Verify function returns None and doesn't overwrite if config is differnt
        assert result is None


class TestLoadModel:

    # Successfully loads LitModel from valid checkpoint path with corresponding torch-module.pt
    def test_load_model_valid_checkpoint(self, tmp_path):
        # Create mock model and dataset
        model_torch = MockTorchModel(0.5, 0.3)
        model = LitModel(model_torch)

        test_file = tmp_path / "test.h5"
        data = get_dataset(20)
        save_scans_to_h5(data, test_file)

        dataset = SegmentationData(
            h5_path=test_file,
            batch_size=2,
            batch_size_eval=2,
            patch_size=(5, 5, 5),
            foreground_oversample_ratio=0.5,
            num_workers_train=0,
            num_workers_val=0,
            prefetch_factor_train=None,
            prefetch_factor_val=None,
        )
        log_dir = Path(tmp_path) / "logs"
        checkpoint_dir = Path(tmp_path) / "checkpoints"

        # Train model
        train_model(
            model=model,
            dataset=dataset,
            log_dir=log_dir,
            experiment_name="test_exp",
            checkpoint_path=checkpoint_dir,
            checkpoint_name="{epoch:03d}-{val_loss:.4f}",
            checkpoint_every_n_epoch=1,
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
            num_sanity_val_steps=0,
        )

        # Load model
        loaded_model = load_model(checkpoint_dir / "epoch=001-val_loss=0.5000.ckpt")

        # Verify loaded model
        assert isinstance(loaded_model, LitModel)
        assert isinstance(loaded_model.model, MockTorchModel)
        assert loaded_model.model.train_loss == 0.5
        assert loaded_model.model.val_loss == 0.3


class TestLoadModels:

    # Successfully loads multiple LitModels from checkpoints
    def test_load_multiple_models(self, tmp_path):
        # Mock get_model to return MockModel
        def mock_get_model(_):
            return MockTorchModel

        # Patch get_model function
        with patch("chhip_uq.training.training.get_model", mock_get_model):
            test_file = tmp_path / "test.h5"
            data = get_dataset(20)
            save_scans_to_h5(data, test_file)

            dataset = SegmentationData(
                h5_path=test_file,
                batch_size=2,
                batch_size_eval=2,
                patch_size=(5, 5, 5),
                foreground_oversample_ratio=0.5,
                num_workers_train=0,
                num_workers_val=0,
                prefetch_factor_train=None,
                prefetch_factor_val=None,
            )
            checkpoint_dir = Path(tmp_path) / "checkpoints"
            experiment_name = "test_exp"

            # Train multiple models
            train_models(
                models=["unet_2", "notunet", "wow2"],
                dataset=dataset,
                checkpoint_dir=checkpoint_dir,
                experiment_name=experiment_name,
                seed=42,
                log_dir=Path(tmp_path) / "logs",
                checkpoint_name="{epoch:03d}-{val_loss:.4f}",
                checkpoint_every_n_epoch=1,
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
                mock__loss=0.5,
                mock__val_loss=0.3,
                num_sanity_val_steps=2,
            )

        # Load models
        loaded_models = load_models(checkpoint_dir, "epoch=001-val_loss=0.5000.ckpt")

        assert set(loaded_models.keys()) == set(["unet-0", "unet-1", "notunet", "wow2"])
        for litmodel in loaded_models.values():
            assert isinstance(litmodel, LitModel)
            assert isinstance(litmodel.model, MockTorchModel)
            assert litmodel.model.train_loss == 0.5
            assert litmodel.model.val_loss == 0.3


class TestLoadTrainingDir:

    # Successfully loads configuration, data splits, indices and checkpoints from valid training directory
    def test_load_valid_training_dir(self, tmp_path):
        mock_get_model = lambda _: MockTorchModel
        mock_get_config = lambda _: {}
        # Setup test data
        config_path = tmp_path / "configuration.yaml"
        config_path.write_text("test config")

        train_dir = tmp_path / "training"
        test_file = tmp_path / "test.h5"
        data = get_dataset(20)
        save_scans_to_h5(data, test_file)

        dataset_h5 = H5Dataset(test_file)

        config_copy, train_test_path, folds_path, fold_dirs = init_training_dir(
            train_dir=train_dir,
            config_path=config_path,
            dataset_indices=dataset_h5.indices,
            n_folds=3,
            test_split=0.2,
        )  # type: ignore

        # Patch get_model function
        with patch("chhip_uq.training.training.get_model", mock_get_model):
            for fold_idx, ckpt_path in enumerate(fold_dirs):
                res = read_fold_splits_file(folds_path, fold_idx)
                assert not isinstance(res, dict)
                train_indices, val_indices = res

                dataset = SegmentationData(
                    h5_path=test_file,
                    train_indices=train_indices,  # type: ignore
                    val_indices=val_indices,  # type: ignore
                    batch_size=2,
                    batch_size_eval=2,
                    patch_size=(5, 5, 5),
                    foreground_oversample_ratio=0.5,
                    num_workers_train=0,
                    num_workers_val=0,
                    prefetch_factor_train=None,
                    prefetch_factor_val=None,
                )
                train_models(
                    models=["unet", "notunet_2"],
                    dataset=dataset,
                    checkpoint_dir=ckpt_path,
                    experiment_name=f"fold_{fold_idx}",
                    log_dir=Path(tmp_path) / "logs",
                    checkpoint_name="{epoch:03d}-{val_loss:.4f}",
                    checkpoint_every_n_epoch=1,
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
                    mock__loss=0.5,
                    mock__val_loss=0.3,
                    num_sanity_val_steps=2,
                )

        # Load training directory
        with patch("chhip_uq.training.training.configuration", mock_get_config):
            config, fold_splits, train_test, checkpoints = load_training_dir(
                train_dir, "epoch=001-val_loss=0.5000.ckpt"
            )

        # Verify loaded data
        assert config == {}
        # just output from read_fold_split_file so no need to test anymore
        assert set(fold_splits.keys()) == set([f"fold_{i}" for i in range(3)])
        assert len(train_test[0]) == 16
        assert len(train_test[1]) == 4

        assert set(checkpoints.keys()) == set([f"fold_{i}" for i in range(3)])
        assert all(isinstance(v, dict) for v in checkpoints.values())
        assert all(
            set(["unet", "notunet-0", "notunet-1"]) == set(v.keys())
            for v in checkpoints.values()
        )
        assert all(
            all(isinstance(m, LitModel) for m in v.values())
            for v in checkpoints.values()
        )


class TestSelectSingleModels:

    # Successfully selects single models from multiple models
    def test_select_single_models(self):
        # Create mock models
        models = {
            "model-0": 4,
            "model-2": 5,
            "model-3": 6,
            "model2-0": 7,
            "model2-2": 8,
            "model2-3": 9,
            "model3": 10,
            "model4-023": 12,
            "model4": 11,
            "model4-0": 12,
        }

        # Select models
        selected_models = select_single_models(models)  # type: ignore

        # Verify selected models
        assert selected_models == {"model": 4, "model2": 7, "model3": 10, "model4": 11}


class TestSelectEnsembles:

    # Successfully selects ensemble models from multiple models
    def test_select_ensembles(self):
        # Create mock models
        models = {
            "model-0": 4,
            "model-2": 5,
            "model-3": 6,
            "model2-0": 7,
            "model2-2": 8,
            "model2-3": 9,
            "model3": 10,
            "model4": 11,
            "model4-0": 12,
        }

        # Select models
        selected_models = select_ensembles(models)  # type: ignore

        # Verify selected models
        assert selected_models == {
            "model": [4, 5, 6],
            "model2": [7, 8, 9],
            "model4": [11, 12],
        }
