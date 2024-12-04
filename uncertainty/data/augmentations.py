"""
Preset data transformations for augmenting 3D volumes and masks
"""

from typing import Callable, Literal

import numpy as np
import toolz as tz
import torch
import torchio as tio
from kornia.geometry.transform import get_affine_matrix3d, warp_affine3d
from kornia.augmentation import RandomAffine3D
from torchio.transforms import (
    Compose,
    RandomAnisotropy,
    RandomBlur,
    RandomFlip,
    RandomGamma,
)

from uncertainty.utils.common import unpack_args

from ..utils.logging import logger_wraps
from .processing import from_torchio_subject, to_torchio_subject


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
        x.double(),
        affine_inv[:, :3].double(),  # cut off last row
        x.shape[2:],  # type: ignore
        flags=flags,
        align_corners=align_corners,
    )


@logger_wraps(level="DEBUG")
def torchio_augmentations(
    p: float = 1.0, ps: list[float] = [0.15, 0.2, 0.2, 0.2]
) -> tio.Compose:
    """
    Returns a torchio Compose object with a set of augmentations
    """
    return Compose(
        [
            RandomAnisotropy(
                (0, 1, 2), scalars_only=True, p=ps[0]
            ),  # simulate low quality
            RandomBlur(p=ps[1]),
            RandomGamma(p=ps[2]),
            # RandomElasticDeformation(num_control_points=5, p=0.15),
            RandomFlip(axes=(0, 1, 2), p=ps[3]),
        ],
        p=p,
    )


@logger_wraps(level="INFO")
def augmentations(
    ps: list[float] = [0.15, 0.2, 0.2, 0.2]
) -> Callable[[tuple[torch.Tensor, torch.Tensor]], tuple[torch.Tensor, torch.Tensor]]:
    """
    Returns a function to augment a (volume, masks) pair

    Augmentation involves flipping, affine transformations, elastic deformation,
    blurring and gamma correction.

    Wrapper around `torchio_augmentation` that handles conversion to and from
    `torchio.Subject`.

    Parameters
    ---------
    ps: float
        Probability of applying each augmentation
    """
    return lambda arr: tz.pipe(
        arr, to_torchio_subject, torchio_augmentations(ps=ps), from_torchio_subject
    )


def batch_augmentations(
    ps: list[float] = [0.15],
) -> Callable[[tuple[torch.Tensor, torch.Tensor]], tuple[torch.Tensor, torch.Tensor]]:
    """
    Apply augmentations to a batch of volumes and masks of shape (B, C, D, H, W)

    Parameters
    ----------
    x_y: tuple[torch.Tensor, torch.Tensor]
        Batch of tensors for instances and labels, each of shape (B, C, D, H, W)

    Returns
    -------
    ps: float
        Probability of applying each augmentation
    """
    # Affine is faster when applied batch-wise!
    affine = RandomAffine3D(5, align_corners=True, shears=0, scale=(0.9, 1.1), p=ps[0])
    affine_mask = RandomAffine3D(
        5,
        align_corners=True,
        shears=0,
        scale=(0.9, 1.1),
        resample="nearest",
        p=ps[0],
    )

    def _affine(x, y):
        x_augmented = affine(x)
        y_augmented = affine_mask(y, affine._params)
        return x_augmented, y_augmented

    return unpack_args(_affine)
