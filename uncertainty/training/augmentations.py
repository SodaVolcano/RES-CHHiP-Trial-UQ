"""
Data augmentation preset
"""

from typing import Callable

from ..data.utils import to_torchio_subject, from_torchio_subject
from ..utils.logging import logger_wraps

import numpy as np
import toolz as tz
from torchio.transforms import (
    Compose,
    RandomFlip,
    RandomAffine,
    RandomElasticDeformation,
    RandomBlur,
    RandomGamma,
)


@logger_wraps(level="INFO")
def augmentations(
    p: float = 1.0,
) -> Callable[[tuple[np.ndarray, np.ndarray]], tuple[np.ndarray, np.ndarray]]:
    """
    Returns a function to augment a (volume, masks) pair

    Augmentation involves flipping, affine transformations, elastic deformation,
    blurring and gamma correction. All augmentations have a probability of 0.5
    of applying meaning on average, around 3% of data will have no augmentations
    applied.

    Parameters
    ---------
    p: float
        Probability of applying the augmentor, default is 1.0
    """
    return lambda arr: tz.pipe(
        arr,
        to_torchio_subject,
        Compose(
            [
                RandomFlip(p=0.5),
                RandomAffine(isotropic=True, p=0.5),
                RandomElasticDeformation(p=0.5),
                RandomBlur(p=0.5),
                RandomGamma(p=0.5),
            ],
            p=p,
        ),
        from_torchio_subject,
    )
