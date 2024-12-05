import os
import sys

import torch
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint

sys.path.append("..")
sys.path.append(".")
import torch
from lightning.pytorch.loggers import TensorBoardLogger


def main(
    config: dict,
    checkpoint_path: str,
    retrain: bool,
    deep_supervision: bool,
):
    model = un.models.UNet(config, deep_supervision)
    model = un.training.LitSegmentation(model, config=config, save_hyperparams=True)

    train_val_indices = None
    # if retrain:
    #     indices = torch.load(os.path.join(checkpoint_path, "indices.pt"))
    # train_val_indices = (indices["train_indices"], indices["val_indices"])
    data = un.training.SegmentationData(config, checkpoint_path, train_val_indices)

    # don't test, prevent possible data-snooping :(
    # trainer.test(model, data)
