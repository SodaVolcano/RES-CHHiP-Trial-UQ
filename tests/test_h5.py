from uncertainty.data.h5 import save_xy_to_h5
from .context import uncertainty
import numpy as np
import h5py as h5

save_xy_to_h5 = uncertainty.data.h5.save_xy_to_h5


class TestSaveXyToH5:

    # Save a dataset of (volume, mask) pairs to an H5 file successfully
    def test_save_xy_to_h5_success(self, tmp_path, mocker):

        dataset = [
            (np.random.rand(10, 10), np.random.randint(0, 2, (10, 10)))
            for _ in range(5)
        ]
        path = tmp_path / "test.h5"

        mocker.patch("uncertainty.data.h5.tqdm", lambda x, desc: x)

        save_xy_to_h5(dataset, str(tmp_path), "test.h5")

        with h5.File(path, "r") as hf:
            for i in range(5):
                assert f"{i}" in hf
                assert "x" in hf[f"{i}"]  # type: ignore
                assert "y" in hf[f"{i}"]  # type: ignore

    # Handle an empty dataset without errors
    def test_save_xy_to_h5_empty_dataset(self, tmp_path, mocker):
        import h5py as h5
        from uncertainty.data.h5 import save_xy_to_h5

        dataset = []
        path = tmp_path / "test_empty.h5"

        mocker.patch("uncertainty.data.h5.tqdm", lambda x, desc: x)

        save_xy_to_h5(dataset, str(tmp_path), "test_empty.h5")

        with h5.File(path, "r") as hf:
            assert len(hf.keys()) == 0
