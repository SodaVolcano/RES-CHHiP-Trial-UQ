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
    RandomAnisotropy,
)


@logger_wraps(level="INFO")
def augmentations(
    p: float = 1.0,
) -> Callable[[tuple[np.ndarray, np.ndarray]], tuple[np.ndarray, np.ndarray]]:
    """
    Returns a function to augment a (volume, masks) pair

    Augmentation involves flipping, affine transformations, elastic deformation,
    blurring and gamma correction.

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
                RandomAnisotropy(
                    (0, 1, 2), scalars_only=True, p=0.15
                ),  # simulate low quality
                RandomBlur(p=0.2),
                RandomGamma(p=0.2),
                # RandomElasticDeformation(num_control_points=5, p=0.15),
                RandomFlip(axes=(0, 1, 2), p=0.2),
                # RandomAffine(scales=0.2, degrees=5, isotropic=True, p=0.15),
            ],
            p=p,
        ),
        from_torchio_subject,
    )
