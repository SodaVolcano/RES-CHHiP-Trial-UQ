from context import uncertainty as un
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning import Trainer
import os
import torch

import dill
from scripts.__helpful_parser import HelpfulParser
from uncertainty.config import Configuration
import torch
from lightning.pytorch.loggers import TensorBoardLogger

def main(
    config: Configuration,
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
    if retrain:
        trainer.fit(model, data, ckpt_path=checkpoint_path)
    else:
        trainer.fit(model, data)
    # don't test, prevent possible data-snooping :(
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
        "--checkpoint_path",
        type=str,
        help="Path to save or load model checkpoints.",
        default=config["model_checkpoint_path"],
    )
    parser.add_argument(
        "--retrain",
        action="store_true",
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
    if args.retrain:
        with open(os.path.join(checkpoint_path, "config.pkl"), "rb") as f:
            config = dill.load(f)

    main(
        config,
        checkpoint_path,
        args.retrain,
        deep_supervision=not args.no_deep_supervision,
    )
