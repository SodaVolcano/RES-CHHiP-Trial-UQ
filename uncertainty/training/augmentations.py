"""
Data augmentation preset
"""

from typing import Callable, Literal

from ..data.utils import to_torchio_subject, from_torchio_subject
from ..utils.logging import logger_wraps

import torch
import torchio as tio
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
from kornia.geometry.transform import get_affine_matrix3d, warp_affine3d


def inverse_affine_transform(
    affine_params: dict[str, torch.Tensor],
    flags: Literal["nearest", "bilinear"] = "bilinear",
    align_corners: bool = True,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """
    Return inverse of a `kornia.augmentation.RandomAffine3D` transform

    Parameters
    ----------
    affine_params: dict
        Dictionary of affine parameters. Assume we have `affine = RandomAffine3D(...)`,
        then `affine_params` is `affine._params` after `affine` has been called
        with `return_transform=True`, i.e. `affine(data, return_transform=True)`.

    Returns
    -------
    np.ndarray
        Function that applies the inverse affine transformation using input
        of shape (N, C, D, H, W)
    """
    affine_matrix = get_affine_matrix3d(
        affine_params["translations"],
        affine_params["center"],
        affine_params["scale"],
        affine_params["angles"],
        sxy=affine_params["sxy"],
        sxz=affine_params["sxz"],
        syx=affine_params["syx"],
        syz=affine_params["syz"],
        szx=affine_params["szx"],
        szy=affine_params["szy"],
    )
    affine_inv = torch.inverse(affine_matrix)
    # expects (B, 3, 4) affine matrix so cut off last row
    return lambda x: warp_affine3d(
        x, affine_inv[:, :3], x.shape[2:], flags=flags, align_corners=align_corners
    )


@logger_wraps(level="DEBUG")
def torchio_augmentation(p: float = 1.0) -> tio.Compose:
    """
    Returns a torchio Compose object with a set of augmentations
    """
    return Compose(
        [
            RandomAnisotropy(
                (0, 1, 2), scalars_only=True, p=0.15
            ),  # simulate low quality
            RandomBlur(p=0.2),
            RandomGamma(p=0.2),
            # RandomElasticDeformation(num_control_points=5, p=0.15),
            RandomFlip(axes=(0, 1, 2), p=0.2),
        ],
        p=p,
    )


@logger_wraps(level="INFO")
def augmentations(
    p: float = 1.0,
) -> Callable[
    [tuple[np.ndarray, np.ndarray]], tuple[np.ndarray, np.ndarray] | tio.Subject
]:
    """
    Returns a function to augment a (volume, masks) pair

    Augmentation involves flipping, affine transformations, elastic deformation,
    blurring and gamma correction.

    Wrapper around `torchio_augmentation` that handles conversion to and from
    `torchio.Subject`.

    Parameters
    ---------
    p: float
        Probability of applying the augmentor, default is 1.0
    """
    return lambda arr: tz.pipe(
        arr, to_torchio_subject, torchio_augmentation(p=p), from_torchio_subject
    )


def augmentations_batch(p: float = 1.0):
    """
    Apply augmentations to a batch of (volume, masks) pairs
    """
    pass  # TODO
