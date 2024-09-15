"""
Functions for performing inference on a model.

Based off of the following code:
    https://github.com/bnsreenu/python_for_microscopists/blob/master/229_smooth_predictions_by_blending_patches/smooth_tiled_predictions.py
which is based off of
    https://github.com/Vooban/Smoothly-Blend-Image-Patches
"""

from tqdm import tqdm
import random
import numpy as np
from toolz import curried
from torch import nn
import lightning as lit

from typing import Callable, Iterable
import toolz as tz
from skimage.util import view_as_windows
from functools import reduce
from scipy.signal.windows import triang
import torch
import numpy as np

from ..utils.wrappers import curry
from ..models import MCDropoutUNet


def _calc_pad_amount(
    patch_sizes: tuple[int, int, int], subdivisions: tuple[int, int, int]
) -> list[tuple[int, ...]]:
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
) -> np.ndarray:
    """
    Parameters
    ----------
    image : np.ndarray
        Image to pad of shape (C, D, H, W)
    """
    return np.pad(
        image,
        [(0, 0)] + _calc_pad_amount(patch_sizes, subdivisions),
        mode="reflect",
    )


@curry
def _spline_window_1d(window_size: int, power: int = 2) -> np.ndarray:
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


def _spline_window_3d(window_sizes: tuple[int, int, int], power: int = 2) -> np.ndarray:
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
) -> np.ndarray:
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
    img_size: tuple[int, int, int, int],
    idx_patches: Iterable[tuple[tuple[int, ...], np.ndarray]],
    window: np.ndarray,
    stride: tuple[int, int, int],
) -> np.ndarray:
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


@curry
@torch.no_grad()
def sliding_inference(
    model: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    patch_size: tuple[int, int, int],
    batch_size: int,
    subdivisions: tuple[int, int, int] | int = 2,
    output_channels: int = 3,
    prog_bar: bool = True,
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
        half of the patch will overlap
    batch_size : int
        Size to batch the patches when performing inference
    output_channels : int
        Number of output channels of the model
    prog_bar : bool
        Whether to display a progress bar
    """
    window = _spline_window_3d(patch_size)
    if isinstance(subdivisions, int):
        subdivisions = (subdivisions, subdivisions, subdivisions)

    x_padded = _pad_image(x.numpy(), patch_size, subdivisions)
    stride = _get_stride(patch_size, subdivisions)
    x_patches = view_as_windows(x_padded, (x_padded.shape[0], *patch_size), (1,) + stride)  # type: ignore

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
        (lambda it: tqdm(it, total=len(patch_indices))) if prog_bar else tz.identity,
    )

    y_pred = _reconstruct_image(
        (output_channels,) + x_padded.shape[1:], y_pred_patches_it, window, stride
    )
    y_pred = _unpad_image(y_pred, patch_size, subdivisions)
    return y_pred


@torch.no_grad()
def mc_dropout_inference(
    base_model: nn.Module | lit.LightningModule,
    x: torch.Tensor,
    patch_size: tuple[int, int, int],
    batch_size: int,
    n_outputs: int,
    subdivisions: tuple[int, int, int] | int = 2,
    output_channels: int = 3,
    prog_bar: bool = True,
) -> Iterable[torch.Tensor]:
    """
    Return iterator of `n_outputs` each performing MC Dropout inference on the full image

    The iterator runs `base_model` `n_outputs` times with dropout enabled to
    get multiple predictions of the image. Patch-based inference is used where
    the full image is split into patches and the model is run on each patch before
    stitching back into the original full image. For the same image, the model will
    have the same dropout mask applied when predicting its patches.

    Parameters
    ----------
    base_model : nn.Module | lit.LightningModule
        Model to perform inference with, produces deterministic predictions. Must
        have `nn.dropout` layers and be trained with dropout enabled.
    x : torch.Tensor
        Full image to perform inference on of shape (C, D, H, W)
    n_outputs : int
        Number of predictions to make
    subdivisions : tuple[int, int, int] | int
        Number of subdivisions to use when stitching patches together, e.g. 2 means
        half of the patch will overlap
    batch_size : int
        Size to batch the patches when performing inference
    output_channels : int
        Number of output channels of the model
    prog_bar : bool
        Whether to display a progress bar

    Returns
    -------
    Iterable[torch.Tensor]
        Iterator of `n_outputs` predictions of the full image
    """
    mcdo_model = MCDropoutUNet(base_model)
    mcdo_model.eval()

    @curry
    def consistent_dropout_model(seed: int, x: torch.Tensor):
        """
        Use the same `seed` for all forward pass of `mcdo_model`
        """
        torch.manual_seed(seed)
        return mcdo_model(x)

    return tz.pipe(
        # Generate seed to use for each inference
        [random.randint(0, 2**32 - 1) for _ in range(n_outputs)],
        curried.map(consistent_dropout_model),
        curried.map(
            sliding_inference(
                patch_size=patch_size,
                subdivisions=subdivisions,
                batch_size=batch_size,
                output_channels=output_channels,
                prog_bar=prog_bar,
            )
        ),
        curried.map(lambda inference: inference(x)),
        curried.map(torch.from_numpy),
    )  # type: ignore
