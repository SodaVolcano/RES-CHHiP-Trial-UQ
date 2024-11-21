from context import uncertainty as un
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning import Trainer
import os
import torch


sys.path.append("..")
sys.path.append(".")
import dill
from scripts.__helpful_parser import HelpfulParser
import torch
from lightning.pytorch.loggers import TensorBoardLogger


def main(
    checkpoint_path: str,
):
    torch.set_float32_matmul_precision("medium")
    torch.autograd.set_detect_anomaly(True)

    with open("./checkpoints/unet_3/config.pkl", "rb") as f:
        config = dill.load(f)

    unet = un.models.UNet(config)

    model = un.training.LitSegmentation.load_from_checkpoint(
        f"./checkpoints/unet_3/last.ckpt",
        config=config,
        model=unet,
        save_hyperparams=False,
    )
    model = un.models.UNetConfidNet(model.model, config)
    model = un.training.lightning.LitConfidNet(model, config)
    train_val_indices = torch.load("./checkpoints/unet_3/indices.pt")

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

    checkpoint_last = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        dirpath=f"{checkpoint_path}",
        filename="last",
        save_top_k=1,
    )

    trainer = Trainer(
        max_epochs=config["n_epochs"],
        limit_train_batches=config["n_batches_per_epoch"],
        limit_val_batches=config["n_batches_val"],
        callbacks=[checkpoint, checkpoint_last],
        strategy="ddp",
        check_val_every_n_epoch=5,
        accelerator="gpu",
        precision="16-mixed",
        enable_progress_bar=True,
        enable_model_summary=True,
        logger=tb_logger,
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
        "--checkpoint_path",
        type=str,
        help="Path to save or load model checkpoints.",
        default=config["model_checkpoint_path"],
    )

    args = parser.parse_args()
    config["staging_dir"] = os.path.dirname(args.data_path)
    config["staging_fname"] = os.path.basename(args.data_path)
    checkpoint_path = args.checkpoint_path

    main(checkpoint_path)
