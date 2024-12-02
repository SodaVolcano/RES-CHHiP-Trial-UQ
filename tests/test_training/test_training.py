from pathlib import Path
import pickle
from pathlib import Path
import tempfile
from ..context import training


split_into_folds = training.split_into_folds
write_training_fold_file = training.write_training_fold_file
init_checkpoint_dir = training.init_checkpoint_dir


class TestSplitIntoFolds:

    # Dataset splits correctly into n_folds with default return_indices=False
    def test_dataset_splits_into_folds(self):
        # Create sample dataset
        dataset = range(10, 0, -1)
        n_folds = 3

        # Get fold splits
        splits = list(split_into_folds(dataset, n_folds))

        expected = [
            ([6, 5, 4, 3, 2, 1], [10, 9, 8, 7]),
            ([10, 9, 8, 7, 3, 2, 1], [6, 5, 4]),
            ([10, 9, 8, 7, 6, 5, 4], [3, 2, 1]),
        ]

        # Verify number of splits
        assert len(splits) == n_folds
        assert splits == expected

    # Returns indices of the dataset split
    def test_dataset_splits_into_folds_indices(self):
        # Create sample dataset
        dataset = list(range(10, 0, -1))
        n_folds = 3

        # Get fold splits
        splits = list(split_into_folds(dataset, n_folds, return_indices=True))
        expected = [
            ([4, 5, 6, 7, 8, 9], [0, 1, 2, 3]),
            ([0, 1, 2, 3, 7, 8, 9], [4, 5, 6]),
            ([0, 1, 2, 3, 4, 5, 6], [7, 8, 9]),
        ]

        # Verify number of splits
        assert len(splits) == n_folds
        assert splits == expected

    def test_fold_1_return_indices(self):
        dataset = list(range(10, 0, -1))
        n_folds = 1

        splits = list(split_into_folds(dataset, n_folds, return_indices=True))
        expected = ([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [])

        assert splits[0] == expected


class TestWriteTrainingFoldFile:

    # Function successfully writes fold indices to file with default seed=True
    def test_write_fold_indices_with_seed(self):
        fold_indices = list(split_into_folds(range(5, 15), 5, return_indices=True))
        with tempfile.NamedTemporaryFile() as tmp:
            path = tmp.name

            write_training_fold_file(path, fold_indices, force=True)  # type: ignore

            with open(path, "rb") as f:
                content = pickle.load(f)

            assert len(content) == 5
            for i in range(5):
                assert f"fold_{i}" in content
                assert content[f"fold_{i}"]["train"] == fold_indices[i][0]
                assert content[f"fold_{i}"]["val"] == fold_indices[i][1]
                assert "seed" in content[f"fold_{i}"]
                assert isinstance(content[f"fold_{i}"]["seed"], int)
