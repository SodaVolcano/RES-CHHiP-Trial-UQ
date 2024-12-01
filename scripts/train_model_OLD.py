import os

import sys
import torch
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint

sys.path.append("..")
sys.path.append(".")
import dill
import torch
from lightning.pytorch.loggers import TensorBoardLogger

from __helpful_parser import HelpfulParser


def main(
    config: dict,
    checkpoint_path: str,
    retrain: bool,
    deep_supervision: bool,
):
    torch.set_float32_matmul_precision("medium")
    torch.autograd.set_detect_anomaly(True)

    model = un.models.UNet(config, deep_supervision)
    model = un.training.LitSegmentation(model, config=config, save_hyperparams=True)

    train_val_indices = None
    if retrain:
        indices = torch.load(os.path.join(checkpoint_path, "indices.pt"))
        train_val_indices = (indices["train_indices"], indices["val_indices"])
    data = un.training.SegmentationData(config, checkpoint_path, train_val_indices)

    tb_logger = TensorBoardLogger(save_dir="./logs/", name="lightning_log_tb")

    checkpoint = ModelCheckpoint(
        monitor="val_loss" if config["val_split"] > 0 else "train_loss",
        mode="min",
        dirpath=f"{checkpoint_path}",
        filename="{epoch:02d}-{val_loss:.3f}",
        every_n_epochs=100,
        save_top_k=-1,
        save_on_train_epoch_end=config["val_split"] == 0,
    )

    checkpoint_last = ModelCheckpoint(
        monitor="val_loss" if config["val_split"] > 0 else "train_loss",
        mode="min",
        dirpath=f"{checkpoint_path}",
        filename="last",
        save_top_k=1,
        save_on_train_epoch_end=config["val_split"] == 0,
    )

    trainer = Trainer(
        max_epochs=config["n_epochs"],
        limit_train_batches=config["n_batches_per_epoch"],
        limit_val_batches=config["n_batches_val"] if config["n_batches_val"] > 0 else 0,
        num_sanity_val_steps=0 if config["val_split"] == 0 else 2,
        callbacks=[checkpoint, checkpoint_last],
        strategy="ddp",
        check_val_every_n_epoch=5,
        accelerator="gpu",
        precision="16-mixed",
        enable_progress_bar=True,
        enable_model_summary=True,
        logger=tb_logger,
    )
    if retrain:
        trainer.fit(model, data, ckpt_path=checkpoint_path)
    else:
        trainer.fit(model, data)
    # don't test, prevent possible data-snooping :(
    # trainer.test(model, data)


if __name__ == "__main__":
    parser = HelpfulParser(
        description="Train a U-Net model on a dataset of DICOM files or h5 files."
    )
    parser.add_argument(
        "--data_path",
        type=str,
        help="Path to the H5 patient scan file.",
        default=None,
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        help="Path to save or load model checkpoints. If the folder already exist, a number will be appended to the folder name.",
        default=None,
    )
    parser.add_argument(
        "--retrain",
        action="store_true",
        help="Whether to retrain a model saved in the model checkpoint",
        default=False,
    )

    args = parser.parse_args()
    config["staging_dir"] = os.path.dirname(args.data_path)
    config["staging_fname"] = os.path.basename(args.data_path)
    checkpoint_path = args.checkpoint_path
    if args.retrain:
        with open(os.path.join(checkpoint_path, "config.pkl"), "rb") as f:
            config = dill.load(f)

    main(
        config,
        checkpoint_path,
        args.retrain,
        deep_supervision=not args.no_deep_supervision,
    )
