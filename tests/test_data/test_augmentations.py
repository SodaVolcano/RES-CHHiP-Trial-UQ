import torch
import numpy as np
from kornia.augmentation import RandomAffine3D
from ..context import data

inverse_affine_transform = data.inverse_affine_transform


class TestInverseAffineTransform:

    # Correctly computes the inverse affine transformation for given parameters
    def test_inverse_affine_transform_correctness(self):
        # Randomly generate padded 3D rectangular data
        inner_shape = (5, 10, 15)
        padding = 40
        full_shape = (
            inner_shape[0] + 2 * padding,
            inner_shape[1] + 2 * padding,
            inner_shape[2] + 2 * padding,
        )
        volume = torch.tensor(np.zeros(full_shape, dtype=np.float64))
        volume[padding:-padding, padding:-padding, padding:-padding] = 1.0

        affine = RandomAffine3D(
            degrees=90,
            translate=(0.1, 0.1, 0.1),
            scale=(0.8, 1.2),
            resample="nearest",
            p=1.0,
        )
        augmented = affine(volume, return_transform=True)

        # Get the inverse transform function
        inverse_transform = inverse_affine_transform(affine._params)

        # Apply the inverse transform
        reconstruction = inverse_transform(augmented)

        # Assert that the recovered data is close to the original data
        assert torch.allclose(volume, reconstruction, atol=1)
