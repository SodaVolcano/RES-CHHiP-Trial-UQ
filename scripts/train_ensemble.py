from random import shuffle

from context import uncertainty
from torch.utils.data import random_split

from uncertainty.config import configuration
from uncertainty.data.patient_scan import from_h5_dir
from uncertainty.models import DeepEnsemble, UNet
from uncertainty.training import PatientScanDataset, augmentations
from uncertainty.training.data_handling import augment_dataset, preprocess_dataset
from uncertainty.training.framework import train

aug = augmentations()
config = configuration()
scans = list(from_h5_dir(config["data_dir"]))
train, val = random_split(
    scans, (1 - config["val_split"]) * len(scans), config["val_split"] * len(scans)  # type: ignore
)

train_data = PatientScanDataset(list(train), transform=augment_dataset(augmentor=aug))
val_data = PatientScanDataset(list(val))

model = DeepEnsemble(UNet, ensemble_size=5)
