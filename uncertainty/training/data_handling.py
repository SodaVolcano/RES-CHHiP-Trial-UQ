"""
Functions for hanlding dataset
"""

from ..data.mask import get_organ_names, masks_as_array
from ..data.patient_scan import PatientScan
from ..data.preprocessing import (
    crop_nd,
    filter_roi,
    find_organ_roi,
    map_interval,
    shift_centre,
)
from ..utils.logging import logger_wraps
from ..utils.wrappers import curry
from ..constants import BODY_THRESH, HU_RANGE, ORGAN_MATCHES
from ..config import configuration, Configuration

from typing import Iterable, Optional

import numpy as np
import toolz as tz
import toolz.curried as curried
from volumentations import (
    Compose,
    Rotate,
    ElasticTransform,
    Flip,
    Transpose,
    RandomGamma,
)


@logger_wraps(level="INFO")
@curry
def preprocess_data(
    scan: PatientScan, config: Configuration = configuration()
) -> tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Preprocess a PatientScan object into (volume, masks) pairs

    Mask for multiple organs are stacked along the last dimension to have
    shape (height, width, depth, n_organs). Mask is `None` if not all organs are present.
    """
    # use same centroid for both volume and mask
    centroid = np.mean(np.argwhere(scan.volume > BODY_THRESH), axis=0)

    def preprocess_volume(scan: PatientScan) -> np.ndarray:
        return tz.pipe(
            scan.volume,
            shift_centre(points=centroid),
            crop_nd(
                new_shape=(
                    config["input_height"],
                    config["input_width"],
                    config["input_depth"],
                    config["input_channel"],
                ),
                pad=True,
            ),
            lambda vol: np.clip(vol, *HU_RANGE),
            map_interval(HU_RANGE, (0, 1)),
            lambda vol: vol.astype(np.float32),
        )

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
            lambda arr: np.moveaxis(arr, -1, 0),  # to allow map()
            curried.map(shift_centre(points=centroid)),
            curried.map(
                crop_nd(
                    new_shape=(
                        config["input_height"],
                        config["input_width"],
                        config["input_depth"],
                        1,  # this is a single mask
                    ),
                    pad=True,
                )
            ),
            list,
            lambda masks: np.stack(masks, axis=-1),
            lambda mask: mask.astype(np.float32),
        )

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

    Parameters
    ---------
    targets: list[list]
        Target to apply augmentation to, e.g. [['image'], ['mask]]
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
