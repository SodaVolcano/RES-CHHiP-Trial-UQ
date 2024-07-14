"""
Collection of functions to preprocess numpy arrays
"""

from operator import add
from typing import Iterable, Optional

import numpy as np
import toolz as tz
from toolz import curried
from scipy.interpolate import interpn

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
    method: str = "linear",
) -> np.ndarray:
    """
    Return an isotropic array with uniformly 1 unit of spacing between coordinates

    As an example, consider a 2D array of shape `(N, M)` defined on a grid with
    spacings `(dx, dy)` on the x and y axes respectively. Let f(x, y) be a function
    that maps the coordinates `(x, y)` on the grid to a value in the array. i.e.
    `array[i, j] = f(x_i, y_j)`. This function interpolates the array to a new
    grid with spacings `(1, 1)` on the x and y axes respectively.

    Parameters
    ----------
    spacings : Iterable[npt.Number]
        Spacing of coordinate points for each axis
    array : npt.NDArray[Any, npt.Number]
        Array of any dimension to interpolate
    method : str, optional
        Interpolation method, by default "linear"

    Returns
    -------
    npt.NDArray[Any, npt.Number]
        Interpolated array on an isotropic grid
    """
    return tz.pipe(
        zip(spacings, array.shape),
        tz.curried.map(
            unpack_args(lambda spacing, length: _get_spaced_coords(spacing, length))
        ),
        tuple,
        # Resample array using isotropic grid
        lambda anisotropic_coords: interpn(
            anisotropic_coords,
            array,
            _isotropic_grid(anisotropic_coords),
            method=method,
        ),
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
