"""
Collection of functions to preprocess numpy arrays
"""

from operator import add
from typing import Iterable, Literal, Optional

import SimpleITK as sitk
import numpy as np
import toolz as tz
from toolz import curried

from uncertainty.utils.logging import logger_wraps

from ..utils.wrappers import curry
from ..utils.common import conditional, unpack_args
from ..common import constants as c


@logger_wraps()
@curry
def map_interval(
    from_range: tuple[np.number, np.number],
    to_range: tuple[np.number, np.number],
    array: np.ndarray,
) -> np.ndarray:
    """
    Map values in an array in range from_range to to_range
    """
    return tz.pipe(
        array,
        lambda a: np.array(
            (a - from_range[0]) / float(from_range[1] - from_range[0]), dtype=float
        ),
        lambda arr_scaled: to_range[0] + (arr_scaled * (to_range[1] - to_range[0])),
    )


@logger_wraps()
def z_score_scale(array: np.ndarray) -> np.ndarray:
    """
    Z-score normalise array to have mean 0 and standard deviation 1

    Calculated as (x - mean) / std
    """
    return (array - array.mean()) / array.std()


@logger_wraps()
@curry
def _isotropic_grid(coords: tuple[Iterable[np.number]]) -> tuple[Iterable[np.number]]:
    """
    Create an isotropic grid (1 unit spacing) from a list of coordinate arrays

    coords is a N-tuple of 1D arrays, each representing the coordinates of an axis.
    """
    return tz.pipe(
        coords,
        tz.curried.map(
            # +1 because arange does not include the stop value
            lambda coord: (
                np.arange(min(np.ceil(coord)), max(np.floor(coord) + 1), 1)
                if len(coord) != 0
                else []
            )
        ),
        tuple,
        lambda coords: np.meshgrid(*coords, indexing="ij"),
        tuple,
    )


@logger_wraps()
@curry
def _get_spaced_coords(spacing: np.number, length: int) -> list[np.number]:
    """
    Return list of coordinates with specified length spaced by spacing
    """
    return tz.pipe(
        [0] + [spacing] * (length - 1),
        tz.curried.accumulate(add),
        list,
    )


@logger_wraps()
@curry
def make_isotropic(
    spacings: Iterable[np.number],
    array: np.ndarray,
    method: Literal["nearest", "linear", "b_spline", "gaussian"] = "linear",
) -> np.ndarray:
    """
    Return an isotropic array with uniformly 1 unit of spacing between coordinates

    Array can ONLY be **2D** or **3D**

    Parameters
    ----------
    spacings : Iterable[npt.Number]
        Spacing of coordinate points for each axis
    array : npt.NDArray[Any, npt.Number]
        Either a **2D** or **3D** array to interpolate
    method : str, optional
        Interpolation method, by default "linear"

    Returns
    -------
    npt.NDArray[Any, npt.Number]
        Interpolated array on an isotropic grid
    """
    interpolators = {
        "nearest": sitk.sitkNearestNeighbor,
        "linear": sitk.sitkLinear,
        "b_spline": sitk.sitkBSpline,
        "gaussian": sitk.sitkGaussian,
    }
    new_size = [
        int(round(old_size * old_spacing))
        for old_size, old_spacing, in zip(array.shape, spacings)
    ]

    img = tz.pipe(
        array,
        lambda arr: arr.astype(np.float32),
        # sitk moves (H, W, D) to (D, W, H) >:( move axis here so img is (H, W, D)
        lambda arr: np.moveaxis(arr, 1, 0),  # width to first axis
        lambda arr: conditional(
            len(arr.shape) == 3, np.moveaxis(arr, -1, 0), arr  # depth to first axis
        ),
        lambda arr: sitk.GetImageFromArray(arr),
    )
    # cannot do functionally, sitk sucks
    img.SetOrigin((0, 0, 0))  # truncates to (0, 0) if array is 2D
    img.SetSpacing(spacings)

    return tz.pipe(
        img,
        lambda img: sitk.Resample(
            img,
            new_size,
            sitk.Transform(),
            interpolators[method],
            img.GetOrigin(),
            (1, 1, 1),  # new spacing
            img.GetDirection(),
            0,
            img.GetPixelID(),
        ),
        lambda img: sitk.GetArrayFromImage(img),
        # arr is (D, W, H) move back to (H, W, D)
        lambda arr: conditional(
            len(arr.shape) == 3, np.moveaxis(arr, 0, -1), arr  # depth to last axis
        ),
        lambda arr: np.moveaxis(arr, 1, 0),  # height to first axis
    )


@logger_wraps()
@curry
def filter_roi(
    roi_names: list[str],
    keep_list: list[str] = c.ROI_KEEP_LIST,
    exclusion_list: list[str] = c.ROI_EXCLUSION_LIST,
) -> list[str]:
    """
    Filter out ROIs based on keep and exclusion lists

    Parameters
    ----------
    roi_names : list[str]
        Array of ROI names
    keep_list : list[str]
        List of substrings to keep
    exclusion_list : list[str]
        List of substrings to exclude

    Returns
    -------
    list[str]
        Array of ROI names not in exclusion list and containing any substring in keep list
    """

    def is_numeric(name):
        try:
            float(name)
            return True
        except ValueError:
            return False

    def not_excluded(name):
        return not any(
            exclude in name for exclude in exclusion_list
        ) and not is_numeric(name)

    return tz.pipe(
        roi_names,
        curried.filter(not_excluded),
        curried.filter(lambda name: any(keep in name for keep in keep_list)),
        list,
    )


def find_organ_roi(organ: str, roi_lst: list[str]) -> Optional[str]:
    """
    Find a unique ROI name in the list that contains the organ names (possibly multiple)

    If more than one ROI name correspond to the organ, the shortest name is chosen.
    If multiple ROI names have the shortest length, the first one in alphabetical
    order is chosen. None is returned if no ROI name is found.
    """
    return tz.pipe(
        roi_lst,
        sorted,
        lambda lst: [name for name in lst if name in c.ORGAN_MATCHES[organ]],
        # encase [min(...)] in list as we will use lst[0] later
        lambda names: [min(names, key=len)] if len(names) > 1 else names,
        lambda name: None if name == [] else name[0],
    )
