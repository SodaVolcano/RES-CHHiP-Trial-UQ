import tempfile
from datetime import date

import numpy as np

from ..context import data, training

H5Dataset = training.H5Dataset
save_scans_to_h5 = data.save_scans_to_h5


def get_dataset(n: int = 2):
    np.random.seed(42)
    return [
        {
            "patient_id": i + 5,
            "volume": np.random.rand(1, 20 + i, 15, 10 + i),
            "dimension_original": (10, 10, 10),
            "spacings": (1.0, 1.0, 1.0),
            "modality": "CT",
            "manufacturer": "GE",
            "scanner": "Optima",
            "study_date": date(2021, 1, 1),
            "masks": np.random.randint(0, 2, (3, 20 + i, 15, 10 + i)),
        }
        for i in range(n)
    ]


class TestH5Dataset:

    # Dataset correctly loads (x,y) pairs from valid H5 file using integer index
    def test_load_xy_pairs_with_integer_index(self):
        with tempfile.NamedTemporaryFile() as tmp:
            test_file = tmp.name
            data = get_dataset()
            save_scans_to_h5(data, test_file)

            # Initialize dataset
            dataset = H5Dataset(test_file)

            assert len(dataset) == 2
            # Get item using integer index
            for i, (x, y) in enumerate(dataset):
                assert np.equal(x, data[i]["volume"]).all()
                assert np.equal(y, data[i]["masks"]).all()

    # Dataset correctly loads (x,y) pairs from valid H5 file using string index
    def test_indices_filters_dataset(self):
        with tempfile.NamedTemporaryFile() as tmp:
            test_file = tmp.name
            data = get_dataset(10)
            save_scans_to_h5(data, test_file)

            indices = [str(num + 5) for num in [0, 1, 3, 5, 7, 9]]
            # Initialize dataset
            dataset = H5Dataset(test_file, indices=indices)

            assert len(dataset) == 6
            # Get item using string index
            for i in indices:
                x, y = dataset[i]
                assert np.equal(x, data[int(i) - 5]["volume"]).all()
                assert np.equal(y, data[int(i) - 5]["masks"]).all()
