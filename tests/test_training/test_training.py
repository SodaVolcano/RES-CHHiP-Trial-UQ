from ..context import training


split_into_folds = training.split_into_folds


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
