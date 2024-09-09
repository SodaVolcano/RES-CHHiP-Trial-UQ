"""
Functions for performing inference on a model.

Based off of the following code:
    https://github.com/bnsreenu/python_for_microscopists/blob/master/229_smooth_predictions_by_blending_patches/smooth_tiled_predictions.py
which is based off of
    https://github.com/Vooban/Smoothly-Blend-Image-Patches
"""

import numpy as np
from toolz import curried


from typing import Callable, Iterable
import toolz as tz
from skimage.util import view_as_windows
from functools import reduce
from scipy.signal.windows import triang
import torch
import numpy as np

from uncertainty.utils.wrappers import curry


def _calc_pad_amount(
    patch_sizes: tuple[int, int, int], subdivisions: tuple[int, int, int]
):
    """
    Calculate padding on each side of the image for each dimension
    """
    calc_amount = (
        lambda subdiv, patch_size: (int(round(patch_size * (1 - 1.0 / subdiv))),) * 2
    )
    return [
        calc_amount(subdiv, patch_size)
        for subdiv, patch_size in zip(subdivisions, patch_sizes)
    ]


def _pad_image(
    image: np.ndarray,
    patch_sizes: tuple[int, int, int],
    subdivisions: tuple[int, int, int],
):
    """
    Parameters
    ----------
    image : np.ndarray
        Image to pad of shape (C, D, H, W)
    """
    print([(0, 0)] + _calc_pad_amount(patch_sizes, subdivisions))
    return np.pad(
        image,
        [(0, 0)] + _calc_pad_amount(patch_sizes, subdivisions),
        mode="reflect",
    )


@curry
def _spline_window_1d(window_size: int, power: int = 2):
    """
    Create a 1d spline window of size `patch_size` with power `power`
    """
    intersection = int(window_size / 4)
    window_outer = (abs(2 * (triang(window_size))) ** power) / 2
    window_outer[intersection:-intersection] = 0

    window_inner = 1 - (abs(2 * (triang(window_size) - 1)) ** power) / 2
    window_inner[:intersection] = 0
    window_inner[-intersection:] = 0

    window = window_inner + window_outer
    return window / np.average(window)


def _spline_window_3d(window_sizes: tuple[int, int, int], power: int = 2):
    """
    Create a 3d spline window of size `patch_size` with power `power`
    """
    # Generate 1D windows for each dimension
    window_xyz = list(map(_spline_window_1d(power=power), window_sizes))

    # Compute the outer product to form a 3D window
    window_3d = reduce(np.outer, window_xyz).reshape(window_sizes)

    return window_3d / np.average(window_3d)


def _unpad_image(
    image: np.ndarray,
    patch_sizes: tuple[int, int, int],
    subdivisions: tuple[int, int, int],
):
    """
    Parameters
    ----------
    image : np.ndarray
        Image to unpad of shape (C, D, H, W)
    """
    pad_amounts = _calc_pad_amount(patch_sizes, subdivisions)
    return image[
        :,
        pad_amounts[0][0] : -pad_amounts[0][1],
        pad_amounts[1][0] : -pad_amounts[1][1],
        pad_amounts[2][0] : -pad_amounts[2][1],
    ]


def _reconstruct_image(
    img_size: tuple[int, int, int],
    idx_patches: Iterable[tuple[tuple[int, ...], np.ndarray]],
    window: np.ndarray,
    stride: tuple[int, int, int],
):
    reconstructed_arr = np.zeros(img_size)
    for idx, patch in idx_patches:
        x_pos, y_pos, z_pos = np.multiply(idx[1:], stride)
        patch *= window
        reconstructed_arr[
            :,
            x_pos : x_pos + patch.shape[1],
            y_pos : y_pos + patch.shape[2],
            z_pos : z_pos + patch.shape[3],
        ] += patch

    return reconstructed_arr / np.average(window)


def _get_stride(
    patch_size: tuple[int, int, int], subdivisions: tuple[int, int, int]
) -> tuple[int, int, int]:
    return tuple(p_size // subdiv for p_size, subdiv in zip(patch_size, subdivisions))  # type: ignore


@torch.no_grad()
def sliding_inference(
    model: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    patch_size: tuple[int, int, int],
    subdivisions: tuple[int, int, int] | int,
    batch_size: int,
):
    """
    Perform inference on full image using sliding patch approach

    The model predicts patches of the image which are then stitched together to
    form the full prediction.

    Parameters
    ----------
    model : Callable[[torch.Tensor], torch.Tensor]
        Model to perform inference with, should take in a tensor of shape (B, C, D, H, W)
        and output a tensor of shape (B, C, D, H, W)
    x : torch.Tensor
        Full image to perform inference on of shape (C, D, H, W)
    patch_size : tuple[int, int, int]
        Size of the patches to use for inference
    subdivisions : tuple[int, int, int] | int
        Number of subdivisions to use when stitching patches together, e.g. 2 means
        half of the patch will overlap.
    batch_size : int
        Size to batch the patches when performing inference
    """
    window = _spline_window_3d(patch_size)
    if isinstance(subdivisions, int):
        subdivisions = (subdivisions, subdivisions, subdivisions)

    x_padded = _pad_image(x.numpy(), patch_size, subdivisions)
    stride = (1,) + _get_stride(patch_size, subdivisions)
    x_patches = view_as_windows(x_padded, (x_padded.shape[0], *patch_size), stride)  # type: ignore

    patch_indices = [idx for idx in np.ndindex(x_patches.shape[:-4])]
    # iterator yielding (idx, y_pred_patch) tuples
    y_pred_patches_it = tz.pipe(
        patch_indices,
        # get the x_patch at each index
        curried.map(lambda idx: x_patches[idx]),
        curried.map(lambda x_patch: torch.from_numpy(x_patch)),
        list,
        torch.stack,
        # split into batches
        lambda x_patches: torch.split(x_patches, batch_size),
        curried.map(model),
        # unbatch the predictions and concat the result tuples into a single list
        curried.map(lambda y_pred_patches: torch.unbind(y_pred_patches, dim=0)),
        tz.concat,
        curried.map(lambda y_pred: y_pred.numpy()),
        lambda y_pred_patches: zip(patch_indices, y_pred_patches),
    )

    stride = _get_stride(x_patches.shape[5:], subdivisions)
    y_pred = _reconstruct_image(x_padded.shape, y_pred_patches_it, window, stride)
    y_pred = _unpad_image(y_pred, patch_size, subdivisions)
    return (y_pred > 0).astype(np.uint8)
