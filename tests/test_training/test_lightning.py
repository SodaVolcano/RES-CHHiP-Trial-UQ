from datetime import date
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torchmetrics.aggregation import RunningMean

from chhip_uq.data.h5 import save_scans_to_h5

from ..context import constants as c
from ..context import data, models, training

LitModel = training.LitModel
UNet = models.UNet
train_model = training.train_model
SegmentationData = training.SegmentationData
save_scans_to_h5 = data.save_scans_to_h5


def get_dataset(n: int = 2):
    np.random.seed(42)
    return [
        {
            "patient_id": i + 5,
            "volume": np.random.rand(1, 100, 100, 100).astype(np.float32),
            "dimension_original": (10, 10, 10),
            "spacings": (1.0, 1.0, 1.0),
            "modality": "CT",
            "manufacturer": "GE",
            "scanner": "Optima",
            "study_date": date(2021, 1, 1),
            # if not float32, kornia augmentation in dataset will fail
            "masks": np.random.randint(0, 2, (3, 100, 100, 100)).astype(np.float32),
        }
        for i in range(n)
    ]


class TestLitModel:

    # Model initialization with default parameters creates correct attributes and metrics
    def test_default_init(self):
        class MockModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer = nn.Linear(10, 1)

            def forward(self, x, logits=False):
                return self.layer(x)

            def loss(self, y_pred, y, logits=True):
                return torch.tensor(0.5)

        model = MockModel()
        lit_model = LitModel(model=model)

        assert isinstance(lit_model.model, nn.Module)
        assert lit_model.class_names == list(c.ORGAN_MATCHES.keys())
        assert lit_model.dump_tensors_every_n_epoch == 0
        assert lit_model.dump_tensors_dir == "tensor_dump"
        assert callable(lit_model.dice)
        assert callable(lit_model.dice_classwise)
        assert isinstance(lit_model.running_loss, RunningMean)
        assert isinstance(lit_model.running_dice, RunningMean)
        assert lit_model.running_loss.window == 10
        assert lit_model.running_dice.window == 10

    def test_training_with_unet(self, tmp_path):
        config = {
            "unet__n_kernels_max": 256,
            "unet__n_kernels_init": 8,
            "unet__n_convolutions_per_block": 2,
            "unet__kernel_size": 3,
            "unet__use_instance_norm": True,
            "unet__activation": nn.ReLU,
            "unet__dropout_rate": 0.1,
            "unet__instance_norm_epsilon": 1e-5,
            "unet__instance_norm_momentum": 0.1,
            "unet__instance_norm_decay": 0.1,
            "unet__n_levels": 3,
            "unet__input_channels": 1,
            "unet__output_channels": 3,
            "unet__final_layer_activation": nn.Sigmoid,
            "unet__input_height": 128,
            "unet__input_width": 128,
            "unet__input_depth": 128,
            "unet__deep_supervision": True,
            "unet__optimiser": torch.optim.SGD,
            "unet__optimiser_kwargs": {"momentum": 0.9},
            "unet__lr_scheduler": torch.optim.lr_scheduler.PolynomialLR,
            "unet__lr_scheduler_kwargs": {"total_iters": 750},
            "unet__initialiser": torch.nn.init.kaiming_normal_,
        }

        # Initialize UNet
        model = UNet(**config)
        lit_model = LitModel(model=model)

        test_file = tmp_path / "test.h5"
        data = get_dataset(20)
        save_scans_to_h5(data, test_file)

        dataset = SegmentationData(
            h5_path=test_file,
            batch_size=2,
            batch_size_eval=2,
            patch_size=(50, 50, 50),
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
            model=lit_model,
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
            num_sanity_val_steps=2,
        )

        assert checkpoint_dir.exists()
        checkpoints = list(checkpoint_dir.glob("*.ckpt"))
        assert [
            i in checkpoints for i in ["epoch=000-val_loss=", "epoch=001-val_loss="]
        ]
        assert (log_dir / "test_exp").exists()
