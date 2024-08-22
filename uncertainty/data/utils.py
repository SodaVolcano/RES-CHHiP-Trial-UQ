"""
Utility functions for data manipulation
"""

import numpy as np
import torchio as tio


def to_torchio_subject(volume_mask: tuple[np.ndarray, np.ndarray]) -> tio.Subject:
    """
    Transform a (volume, mask) pair into a torchio Subject object

    Parameters
    ----------
    volume_mask : tuple[np.ndarray, np.ndarray]
        Tuple containing the volume and mask arrays, must have shape
        (channel, height, width, depth)
    """
    return tio.Subject(
        volume=tio.ScalarImage(tensor=volume_mask[0]),
        mask=tio.LabelMap(
            # ignore channel dimension from volume_mask[0]
            tensor=tio.CropOrPad(volume_mask[0].shape[1:], padding_mode="minimum")(  # type: ignore
                volume_mask[1]
            ),
        ),
    )


def from_torchio_subject(subject: tio.Subject) -> tuple[np.ndarray, np.ndarray]:
    """
    Transform a torchio Subject object into a (volume, mask) pair
    """
    return subject["volume"].data, subject["mask"].data
