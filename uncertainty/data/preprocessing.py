"""
Collection of functions to preprocess numpy arrays
"""

from operator import sub
import operator
from typing import Iterable, Literal, Optional, Sequence, Tuple

import SimpleITK as sitk
import numpy as np
import toolz as tz
from toolz import curried

from uncertainty.utils.logging import logger_wraps

from ..utils.wrappers import curry
from ..utils.common import call_method, conditional, unpack_args
from .. import constants as c


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

    return tz.pipe(
        array,
        # sitk don't work with bool datatypes in mask array
        lambda arr: arr.astype(np.float32),
        # sitk moves (H, W, D) to (D, W, H) >:( move axis here so img is (H, W, D)
        lambda arr: np.moveaxis(arr, 1, 0),  # width to first axis
        lambda arr: conditional(
            len(arr.shape) == 3, np.moveaxis(arr, -1, 0), arr  # depth to first axis
        ),
        lambda arr: sitk.GetImageFromArray(arr),
        call_method("SetOrigin", (0, 0, 0)),
        call_method("SetSpacing", spacings),
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


@logger_wraps()
@curry
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
    )  # type: ignore


@logger_wraps()
@curry
def center_box_slice(
    background_shape: Sequence[int], box_shape: Sequence[int]
) -> tuple[slice, ...]:
    """
    Return slices from imposing box of specific shape in the center of background

    The centered box is obtained when slices is used to index the background array
    """
    ceil_divide = lambda x, y: np.ceil(np.divide(x, y)).astype(int)
    return tuple(
        [
            slice(bg_mid - box_mid, bg_mid + box_mid)
            for bg_mid, box_mid in zip(
                ceil_divide(background_shape, 2), ceil_divide(box_shape, 2)
            )
        ]
    )


@logger_wraps()
@curry
def enlarge_array(
    array: np.ndarray, scale: int, fill: Literal["min", "max"] | int = "min"
) -> np.ndarray:
    """
    Enlarge an array by scale by padding with zero, min or max value

    If `fill` is a number, that number will be used to pad the array. If it
    is "min" or "max", then the min/max value from the array is used to pad
    the new array.
    """
    fill_map = {"min": np.min(array), "max": np.max(array)}
    new_shape = tuple(np.multiply(array.shape, scale))
    big_array = np.full(
        new_shape,
        fill if isinstance(fill, int) else fill_map[fill],
        dtype=array.dtype,
    )
    slices = center_box_slice(new_shape, array.shape)
    big_array[slices] = array

    return big_array


@logger_wraps()
@curry
def shift_center(array: np.ndarray, points: Sequence[int]) -> np.ndarray:
    """
    Move a point in the array to the center of the array
    """

    def shift(enlarged_arr: np.ndarray, d_dims: Iterable[int]) -> np.ndarray:
        """
        d_dims is the difference between the new and old dimensions
        """
        return tz.pipe(
            zip(array.shape, d_dims),
            curried.map(lambda x: (x[0], x[0] - x[1])),  # (dim_old, dim_new)
            curried.map(
                lambda x: slice(x[1], x[1] + x[0])
            ),  # [dim_new:dim_new + dim_old]
            list,
            lambda slices: enlarged_arr[*slices],
        )  # type: ignore

    new_pos = [round(dim / 2 - points[i]) for i, dim in enumerate(array.shape)]
    return shift(enlarge_array(array, 3), new_pos)


@logger_wraps()
@curry
def crop_nd(array: np.ndarray, new_shape: Tuple[int], pad: bool = False):
    """
    Center-crop the array into the new_shape, pad with min value if needed
    """
    SCALE = 2 if pad else 1
    slices = tz.pipe(
        array.shape,
        lambda s: np.multiply(s, SCALE),
        tuple,
        center_box_slice(box_shape=new_shape),
    )

    return tz.pipe(
        array,
        enlarge_array(scale=SCALE, fill="min") if pad else tz.identity,
        lambda arr: arr[slices],
    )
