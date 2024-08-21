from typing import Callable, Optional

import lightning as lit
import numpy as np
import toolz as tz
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split

from ..config import Configuration
from ..data.patient_scan import PatientScan, from_h5_dir
from .data_handling import augment_dataset, augmentations, preprocess_dataset


class PatientScanDataset(Dataset):
    def __init__(
        self,
        patient_scans: list["PatientScan"],
        transform: Optional[
            Callable[[tuple[np.ndarray, np.ndarray]], tuple[np.ndarray, np.ndarray]]
        ] = tz.identity,
    ):
        """
        Parameters
        ----------
        patient_scans : list[PatientScan]
            List of PatientScan objects to be converted to
            (volume, masks) pairs. The volume and masks will have shape
            (H, W, D, C) where C is the number of channels which
            PatientScanDataset will convert to (C, H, W, D).
        transform : Optional[Callable]
            A function that take in a (volume, masks) pair and returns a new
            (volume, masks) pair. Default is the identity function.
        buffer_size : int
            Size of the buffer used by the Shuffler to randomly shuffle the dataset.
            Set to 1 to disable shuffling.
        """
        super().__init__()
        self.data: list[tuple[np.ndarray, np.ndarray]] = preprocess_dataset(
            patient_scans
        )
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return tz.pipe(
            self.data[index],
            self.transform,
            lambda vol_mask: (torch.tensor(vol_mask[0]), torch.tensor(vol_mask)[1]),
        )  # type: ignore


class LitSegmentation(lit.LightningModule):
    """
    Wrapper class for PyTorch model to be used with PyTorch Lightning
    """

    def __init__(self, model: nn.Module, config: Configuration):
        super().__init__()
        self.model = model
        self.config = config
        self.loss_fn = config["loss"]()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch: torch.Tensor):
        x, y = batch
        y_pred = self.model(x)
        loss = self.loss_fn(y_pred, y)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):  # type: ignore
        optimiser = self.config["optimiser"](
            self.model.parameters(), **self.config["optimiser_kwargs"]
        )
        lr_scheduler = self.config["lr_scheduler"](self.optimizer)
        return {"optimizer": optimiser, "lr_scheduler": lr_scheduler}


class SegmentationData(lit.LightningDataModule):
    """
    Wrapper class for PyTorch dataset to be used with PyTorch Lightning

    WARNING: Only loads h5 files
    """

    def __init__(self, config: Configuration):
        super().__init__()
        self.config = config
        self.data_dir = config["staging_dir"]
        self.batch_size = config["batch_size"]
        self.val_split = config["val_split"]
        self.augmentations = augmentations(p=1)

        scans = list(filter(lambda x: x is not None, from_h5_dir(self.data_dir)))
        self.train, self.val = random_split(
            scans, (1 - config["val_split"]) * len(scans), config["val_split"] * len(scans)  # type: ignore
        )

    def train_dataloader(self):
        dataset = PatientScanDataset(list(self.train), transform=augment_dataset(augmentor=self.augmentations))  # type: ignore
        return DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True, num_workers=2
        )

    def val_dataloader(self):
        val = list(self.val)  # type: ignore
        dataset = PatientScanDataset(val)
        return DataLoader(dataset, batch_size=len(val), num_workers=2)

    def test_dataloader(self):
        pass
