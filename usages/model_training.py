import argparse
from typing import Callable, Literal
import toolz as tz

import numpy as np
from uncertainty.data.dicom import load_patient_scans
from uncertainty.data.patient_scan import from_h5_dir
from uncertainty.training.data_handling import (
    augment_data,
    preprocess_dataset,
    construct_augmentor,
)
from uncertainty.models.config import model_config
import tensorflow as tf


def main(data_path: str, datatype: Literal["dicom", "h5"]):
    data_loader_map: dict[str, Callable] = {
        "dicom": load_patient_scans,
        "h5": from_h5_dir,
    }
    aug = construct_augmentor()
    config = model_config()

    @tf.numpy_function(Tout=(tf.float32, tf.float32))  # type: ignore
    def augmentation(x: np.ndarray, y: np.ndarray):
        """Wrap in tf.numpy_function to apply augmentor to numpy arrays"""
        return augment_data(x, y, aug)

    dataset_it = tz.pipe(
        data_path,
        data_loader_map[datatype],
        preprocess_dataset(config=config),
    )

    dataset = (
        tf.data.Dataset.from_generator(
            lambda: dataset_it,
            output_signature=(
                tf.TensorSpec(shape=(config["input_height"], config["input_width"], config["input_depth"]), dtype=tf.float32),  # type: ignore
                tf.TensorSpec(shape=(config["input_height"], config["input_width"], config["input_depth"], config["input_channel"]), dtype=tf.float32),  # type: ignore
            ),
        )
        .repeat()
        .shuffle(
            config["batch_size"] * 20
        )  # buffer size should be larger than dataset size
        .batch(config["batch_size"])
        .map(augmentation, num_parallel_calls=tf.data.AUTOTUNE)
        .prefetch(tf.data.AUTOTUNE)
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Load and preprocess data for training."
    )
    parser.add_argument(
        "data_path",
        type=str,
        help="Path to the folder containing folders of folders of DICOM files, or folders of h5 files.",
    )
    parser.add_argument(
        "--datatype",
        type=str,
        choices=["dicom", "h5"],
        help='Extension of data file to load, "dicom" or "h5". Default is "dicom".',
        default="dicom",
    )
