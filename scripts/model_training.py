from context import uncertainty as un
from lightning.pytorch.callbacks import TQDMProgressBar, ModelCheckpoint, EarlyStopping
from lightning import Trainer
import os
import torch
from lightning.pytorch.strategies import DeepSpeedStrategy

from scripts.__helpful_parser import HelpfulParser
from tests.test_unet import UNet
from uncertainty.config import Configuration
import torch


def main(
    config: Configuration,
    ensemble_size: int,
    checkpoint_path: str,
    deep_supervision: bool = True,
):
    torch.set_float32_matmul_precision("high")

    if ensemble_size > 1:
        model = un.models.DeepEnsemble(
            lambda x: UNet(x, deep_supervision=deep_supervision),
            ensemble_size,
            config=config,
        )
    else:
        model = UNet(config=config, deep_supervision=deep_supervision)

    model = un.training.LitSegmentation(model, config=config)
    data = un.training.SegmentationData(config)
    pbar = TQDMProgressBar()

    checkpoint = ModelCheckpoint(
        monitor="val/loss",
        mode="min",
        dirpath=f"{checkpoint_path}/model1",
        filename="{epoch:02d}-{val/loss:.2f}",
        auto_insert_metric_name=False,
    )
    early_stopping = EarlyStopping(
        monitor="val/loss", min_delta=0.00, patience=5, verbose=False, mode="min"
    )

    trainer = Trainer(
        max_epochs=10,
        callbacks=[checkpoint, early_stopping, pbar],
        strategy=DeepSpeedStrategy(
            stage=3,
            offload_optimizer=True,
            offload_parameters=True,
        ),
        accelerator="gpu",
        precision="16-mixed",
    )
    trainer.fit(model, data)


if __name__ == "__main__":
    config = un.config.configuration()
    parser = HelpfulParser(
        description="Train a model on a dataset of DICOM files or h5 files."
    )
    parser.add_argument(
        "--data_path",
        type=str,
        help="Path to the dataset.h5 file containing list of (x, y) pairs.",
        default=os.path.join(config["staging_dir"], config["staging_fname"]),
    )
    parser.add_argument(
        "--ensemble_size",
        type=int,
        help="Number of models in the ensemble. Default is 1 (no ensemble training).",
        default=1,
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        help="Path to save model checkpoints.",
        default=config["model_checkpoint_path"],
    )
    parser.add_argument(
        "--deep_supervision",
        action="store_true",
        help="Use deep supervision during training.",
        default=config["deep_supervision"],
    )

    args = parser.parse_args()
    config["staging_dir"] = os.path.dirname(args.data_path)
    config["staging_fname"] = os.path.basename(args.data_path)

    main(config, args.ensemble_size, args.checkpoint_path)
