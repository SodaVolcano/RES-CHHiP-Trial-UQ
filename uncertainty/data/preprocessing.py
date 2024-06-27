"""
Collection of functions to preprocess numpy arrays
"""

from operator import add
from typing import Any, Generator, Iterable

import numpy as np
import nptyping as npt
import toolz as tz
import toolz.curried as curried
from scipy.interpolate import interpn

from ..common.utils import curry


@curry
def map_interval(
    from_range: tuple[npt.Number, npt.Number],
    to_range: tuple[npt.Number, npt.Number],
    array: npt.NDArray[Any, npt.Number],
) -> npt.NDArray[Any, npt.Float]:
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


def z_score_scale(array: npt.NDArray[Any, npt.Number]) -> npt.NDArray[Any, npt.Float]:
    """
    Z-score normalise array to have mean 0 and standard deviation 1

    Calculated as (x - mean) / std
    """
    return (array - array.mean()) / array.std()


@curry
def _isotropic_grid(coords: tuple[Iterable[npt.Number]]) -> tuple[Iterable[npt.Number]]:
    """
    Create an isotropic grid (1 unit spacing) from a list of coordinate arrays

    coords is a N-tuple of 1D arrays, each representing the coordinates of an axis.
    """
    return tz.pipe(
        coords,
        tz.curried.map(lambda coord: np.arange(min(coord), max(coord), 1)),
        tuple,
        lambda coords: np.meshgrid(*coords, indexing="ij"),
        tuple,
    )


@curry
def _get_spaced_coords(spacing: npt.Number, length: int) -> list[npt.Number]:
    """
    Return list of coordinates with specified length spaced by spacing
    """
    return tz.pipe(
        [0] + [spacing] * (length - 1),
        tz.curried.accumulate(add),
        list,
    )


@curry
def make_isotropic(
    spacings: Iterable[npt.Number],
    array: npt.NDArray[Any, npt.Number],
    method: str = "linear",
) -> npt.NDArray[Any, npt.Number]:
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
            lambda spacing_len: _get_spaced_coords(spacing_len[0], spacing_len[1])
        ),
        tuple,
        # Resample array using isotropic grid
        lambda anisotropic_coords: interpn(
            anisotropic_coords,
            array,
            _isotropic_grid(anisotropic_coords),
            method=method,
        ),
    )  # type: ignore (doesn't recognise interpn)
