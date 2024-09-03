from numpy import isin
from context import uncertainty as un
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning import Trainer
import os
import torch

from scripts.__helpful_parser import HelpfulParser
from uncertainty.config import Configuration
import torch
from lightning.pytorch.loggers import TensorBoardLogger


def main(
    config: Configuration,
    ensemble_size: int,
    checkpoint_path: str,
    retrain: bool,
    deep_supervision: bool,
):

    torch.set_float32_matmul_precision("medium")
    torch.autograd.set_detect_anomaly(True)

    lit_model_cls = (
        un.training.LitDeepEnsemble
        if ensemble_size > 1
        else un.training.LitSegmentation
    )

    if retrain:
        model = lit_model_cls.load_from_checkpoint(checkpoint_path)
    else:
        model = un.models.UNet(config=config, deep_supervision=deep_supervision)
        if isinstance(lit_model_cls, un.training.LitDeepEnsemble):
            model = lit_model_cls(model, ensemble_size, config=config)
        elif isinstance(lit_model_cls, un.training.LitSegmentation):
            model = lit_model_cls(model, config=config)

    data = un.training.SegmentationData(ensemble_size, config)

    tb_logger = TensorBoardLogger(
        save_dir="/media/tin/Expansion/honours/logs/", name="lightning_log_tb"
    )

    checkpoint = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        dirpath=f"{checkpoint_path}",
        filename="{epoch:02d}-{val_loss:.3f}",
        every_n_epochs=100,
        save_top_k=-1,
    )

    trainer = Trainer(
        max_epochs=config["n_epochs"],
        limit_train_batches=config["n_batches_per_epoch"],
        limit_val_batches=config["n_batches_val"],
        callbacks=[checkpoint],
        strategy="ddp",
        check_val_every_n_epoch=5,
        accelerator="gpu",
        precision="16-mixed",
        enable_progress_bar=True,
        enable_model_summary=True,
        logger=tb_logger,
    )
    trainer.fit(model, data)
    # don't train, prevent possible data-snooping :(
    # trainer.test(model, data)


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
        help="Path to save or load model checkpoints.",
        default=config["model_checkpoint_path"],
    )
    parser.add_argument(
        "--retrain",
        type=bool,
        help="Whether to retrain a model saved in the model checkpoint",
        default=False,
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
    checkpoint_path = args.checkpoint_path

    main(
        config,
        args.ensemble_size,
        checkpoint_path,
        args.retrain,
        deep_supervision=not args.no_deep_supervision,
    )
