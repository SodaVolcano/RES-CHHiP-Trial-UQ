"""
Functions for hanlding dataset
"""

from uncertainty.utils.common import apply_if_truthy
from ..data.mask import get_organ_names, masks_as_array
from ..data.patient_scan import PatientScan
from ..data.preprocessing import (
    filter_roi,
    find_organ_roi,
    map_interval,
)
from ..utils.logging import logger_wraps
from ..utils.wrappers import curry
from ..constants import BODY_THRESH, HU_RANGE, ORGAN_MATCHES
from ..config import configuration, Configuration

from typing import Iterable, Optional

import numpy as np
import toolz as tz
import toolz.curried as curried
import torchio as tio
from volumentations import (
    Compose,
    Rotate,
    ElasticTransform,
    Flip,
    RandomGamma,
)


@logger_wraps(level="INFO")
@curry
def preprocess_data(
    scan: PatientScan, config: Configuration = configuration()
) -> tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Preprocess a PatientScan object into (volume, masks) pairs of shape (C, H, W, D)

    Mask for multiple organs are stacked along the first dimension to have
    shape (organ, height, width, depth). Mask is `None` if not all organs are present.
    """
    # Centre image using body mask and to (channel, height, width, depth)
    BODY_MASK = np.expand_dims(scan.volume > BODY_THRESH, axis=0)

    def to_torchio_subject(volume_mask: tuple[np.ndarray, np.ndarray]) -> tio.Subject:
        # Allow centre crop from torchio
        # volume and mask should have shape (C, H, W, D)
        return tio.Subject(
            volume=tio.ScalarImage(tensor=volume_mask[0]),
            mask=tio.LabelMap(
                # ignore channel dimension from volume_mask[0]
                tensor=tio.CropOrPad(volume_mask[0].shape[1:], padding_mode="minimum")(  # type: ignore
                    volume_mask[1]
                ),
            ),
        )

    def from_torchio_subject(subject: tio.Subject) -> tuple[np.ndarray, np.ndarray]:
        return subject["volume"].data, subject["mask"].data

    def torchio_crop_or_pad(
        volume_mask: tuple[np.ndarray, np.ndarray]
    ) -> tuple[np.ndarray, np.ndarray]:
        return tz.pipe(
            volume_mask,
            to_torchio_subject,
            tio.CropOrPad(
                (
                    config["input_height"],
                    config["input_width"],
                    config["input_depth"],
                ),
                padding_mode=np.min(volume_mask[0]),
                mask_name="mask",
            ),
            from_torchio_subject,
            lambda x: (x[0].numpy(), x[1].numpy()),
        )

    def preprocess_volume(scan: PatientScan) -> np.ndarray:
        return tz.pipe(
            scan.volume,
            lambda vol: np.expand_dims(
                vol, axis=0
            ),  # to (channel, height, width, depth)
            lambda vol: (vol, BODY_MASK),
            torchio_crop_or_pad,
            tz.first,
            lambda vol: np.clip(vol, *HU_RANGE),
            map_interval(HU_RANGE, (0, 1)),
            lambda vol: np.astype(vol, np.float32),
        )  # type: ignore

    def preprocess_mask(scan: PatientScan) -> Optional[np.ndarray]:
        """
        Returns a mask with all organs present, or None if not all organs are present
        """
        names = tz.pipe(
            get_organ_names(scan.masks[""]),
            filter_roi,
            lambda mask_names: [
                find_organ_roi(organ, mask_names) for organ in ORGAN_MATCHES
            ],
            curried.filter(lambda m: m is not None),
            list,
        )

        # If not all organs are present, return None
        if len(names) != len(ORGAN_MATCHES):
            return None

        return tz.pipe(
            scan.masks[""],
            masks_as_array(organ_ordering=names),
            lambda arr: np.moveaxis(arr, -1, 0),  # to (organ, height, width, depth)
            lambda mask: (mask, BODY_MASK),
            torchio_crop_or_pad,
            tz.first,
            lambda mask: np.astype(mask, np.float32),
        )  # type: ignore

    return tz.juxt(preprocess_volume, preprocess_mask)(scan)


@logger_wraps(level="INFO")
@curry
def preprocess_dataset(
    dataset: Iterable[PatientScan], config: Configuration = configuration()
) -> Iterable[tuple[np.ndarray, np.ndarray]]:
    """
    Preprocess a dataset of PatientScan objects into (volume, masks) pairs

    Mask for multiple organs are stacked along the last dimension to have
    shape (height, width, depth, n_organs). An instance is filtered out if
    not all organs are present.
    """
    return tz.pipe(
        dataset,
        curried.map(preprocess_data(config=config)),
        curried.filter(lambda x: x[1] is not None),
    )  # type: ignore


@logger_wraps(level="INFO")
def construct_augmentor(p: float = 1.0) -> Compose:
    """
    Preset augmentor to apply rotation, elastic transformation, flips, and gamma adjustments

    Using Volumentations-3D here because torchio is a pain to work with

    Parameters
    ---------
    p: float
        Probability of applying the augmentor
    """
    return Compose(
        [
            Rotate((-15, 15), (-15, 15), (-15, 15), p=0.5),
            ElasticTransform((0, 0.15), interpolation=1, p=0.2),
            Flip(0, p=0.2),
            Flip(1, p=0.2),
            RandomGamma(gamma_limit=(80, 120), p=0.2),
        ],
        p=p,
    )


@logger_wraps(level="INFO")
@curry
def augment_data(
    image: np.ndarray, masks: np.ndarray, augmentor: Compose
) -> tuple[np.ndarray, np.ndarray]:
    """
    Augment data using the provided augmentor
    """
    return tz.pipe(
        # Pack in dictionary to work with augmentor
        {
            "image": image,
            "mask": masks,
        },
        lambda x: augmentor(**x),
        lambda x: (x["image"], x["mask"]),
    )  # type: ignore


@logger_wraps(level="INFO")
@curry
def augment_dataset(
    dataset: Iterable[tuple[np.ndarray, np.ndarray]], augmentor: Compose
) -> Iterable[tuple[np.ndarray, np.ndarray]]:
    """
    Augment data using the provided augmentor
    """
    return map(augment_data(augmentor=augmentor), dataset)
