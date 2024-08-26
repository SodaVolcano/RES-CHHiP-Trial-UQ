from context import uncertainty as un
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning import Trainer
import os
import torch
from lightning.pytorch.strategies import DeepSpeedStrategy, DDPStrategy

from scripts.__helpful_parser import HelpfulParser
from uncertainty.config import Configuration
import torch
from lightning.pytorch.loggers import TensorBoardLogger, CSVLogger

def main(
    config: Configuration,
    ensemble_size: int,
    checkpoint_path: str,
    deep_supervision: bool = True,
):
    torch.set_float32_matmul_precision("high")

    if ensemble_size > 1:
        model = un.models.DeepEnsemble(
            lambda x: un.models.UNet(x, deep_supervision=deep_supervision),
            ensemble_size,
            config=config,
        )
    else:
        model = un.models.UNet(config=config, deep_supervision=deep_supervision)

    model = un.training.LitSegmentation(model, config=config)
    data = un.training.SegmentationData(config)

    tb_logger = TensorBoardLogger(save_dir="/media/tin/Expansion/honours/logs/", name="lightning_log_tb")
    csv_logger = CSVLogger(save_dir="/media/tin/Expansion/honours/logs/", name="lightning_log_csv")
    checkpoint = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        dirpath=f"{checkpoint_path}/unet1",
        filename="{epoch:02d}-{val/loss:.2f}",
        auto_insert_metric_name=False,
    )

    strategy = DeepSpeedStrategy(
            stage=2,
            offload_optimizer=True,
            offload_parameters=False,
        )
    trainer = Trainer(
        max_epochs=config["n_epochs"],
        limit_train_batches=config["n_batches_per_epoch"],
        limit_val_batches=config["n_batches_val"],
        callbacks=[checkpoint],
        strategy=strategy,
        accelerator="gpu",
        precision="16-mixed",
        enable_progress_bar=True,
        enable_model_summary=True,
        logger=(tb_logger, csv_logger)
    )
    trainer.fit(model, data)
    # don't train, prevent possible data-snooping :(
    #trainer.test(model, data)


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
        "--no_deep_supervision",
        action="store_false",
        help="Disable deep supervision during training.",
        default=not config["deep_supervision"],
    )

    args = parser.parse_args()
    config["staging_dir"] = os.path.dirname(args.data_path)
    config["staging_fname"] = os.path.basename(args.data_path)
    main(config, args.ensemble_size, args.checkpoint_path, deep_supervision=not args.no_deep_supervision)
