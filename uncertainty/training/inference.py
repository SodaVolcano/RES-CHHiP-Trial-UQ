"""
Functions for performing inference on a model.

Based off of the following code:
    https://github.com/bnsreenu/python_for_microscopists/blob/master/229_smooth_predictions_by_blending_patches/smooth_tiled_predictions.py
which is based off of
    https://github.com/Vooban/Smoothly-Blend-Image-Patches
"""


# spline
# sliding_inference: take model, 1 (x, y), and output full img prediction
#   - sliding window - get slides (change slidingdataset)
#   - feed B size batch to model
#   - reconstruct full image
#
#   - smooth stitching

from typing import Callable
import toolz as tz
from skimage.util import view_as_windows
from functools import reduce
from scipy.signal.windows import triang
from torch import nn
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
    patches: np.ndarray,
    subdivisions: tuple[int, int, int],
    window: np.ndarray,
):
    reconstructed_arr = np.zeros(img_size)
    stride = _get_stride(patches.shape[5:], subdivisions)

    for idx in np.ndindex(patches.shape[:-4]):
        x_pos, y_pos, z_pos = np.multiply(idx[1:], stride)
        patch = patches[idx] * window
        reconstructed_arr[
            :,
            x_pos : x_pos + patch.shape[1],
            y_pos : y_pos + patch.shape[2],
            z_pos : z_pos + patch.shape[3],
        ] += patch

    return reconstructed_arr / np.average(window)


def _get_stride(patch_size: tuple[int, int, int], subdivisions: tuple[int, int, int]):
    return tuple(p_size // subdiv for p_size, subdiv in zip(patch_size, subdivisions))


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
        Number of subdivisions to use when stitching patches together
    batch_size : int
        Size to batch the patches when performing inference
    """
    window = _spline_window_3d(patch_size)
    if isinstance(subdivisions, int):
        subdivisions = (subdivisions, subdivisions, subdivisions)

    x_padded = _pad_image(x.detach().numpy(), patch_size, subdivisions)
    stride = (1,) + _get_stride(patch_size, subdivisions)
    x_patches = view_as_windows(x_padded, (x_padded.shape[0], *patch_size), stride)

    y_pred_patches = np.zeros_like(x_patches)
    for idx in np.ndindex(x_patches.shape[:-4]):
        y_pred_patches[idx] = tz.pipe(
            x_patches[idx],
            model,
            lambda y_pred: y_pred.detach().numpy(),
        )

    y_pred = _reconstruct_image(x_padded.shape, y_pred_patches, subdivisions, window)
    y_pred = _unpad_image(y_pred, patch_size, subdivisions)
    return (y_pred > 0).astype(np.uint8)


# batched inference
# partition buffer into list of 2D list of idx
# for each partition
#     get patches
#     stack into batch
#     run model
#     add to y_pred_patches using idx


# for idx in patches...   (0, 0, 1), (0, 0, 2) etc
#     get patch in idx
#     add to zero array
# so... delay evaluation until reconstruction...
# make iterator that keep buffer of 2 idx, when need to return next patch, run
# model eval and return (idx, patch)!   idx is np.ndindex
