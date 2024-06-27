from .context import uncertainty
import pytest
import numpy as np

# Import aliases
Mask = uncertainty.data.datatypes.Mask


class TestMask:

    # Retrieve mask for a valid organ using get_organ_mask
    def test_retrieve_mask_for_valid_organ(self):

        # Initialize a Mask object with organ-mask pairs
        mask = Mask(
            {
                "organ_1": np.array([[1, 1], [0, 1]]),
                "organ_2": np.array([[0, 1], [1, 0]]),
            }
        )

        # Retrieve the mask for "organ_1"
        organ_1_mask = mask.get_organ_mask("organ_1")

        # Assert that the retrieved mask is correct
        np.testing.assert_array_equal(organ_1_mask, np.array([[1, 1], [0, 1]]))

    # Retrieve mask for a non-existent organ using get_organ_mask
    def test_retrieve_mask_for_non_existent_organ(self):

        # Initialize a Mask object with organ-mask pairs
        mask = Mask(
            {
                "organ_1": np.array([[1, 1], [0, 1]]),
                "organ_2": np.array([[0, 1], [1, 0]]),
            }
        )

        # Attempt to retrieve the mask for a non-existent organ and assert that it raises a KeyError
        with pytest.raises(KeyError):
            mask.get_organ_mask("non_existent_organ")
