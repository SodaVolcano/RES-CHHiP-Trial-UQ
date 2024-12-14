from ..context import utils

merge_with_reduce = utils.merge_with_reduce


class TestMergeWithReduce:

    # Merge multiple dictionaries with same keys using sum reduction function
    def test_merge_dicts_with_list_append(self):
        dicts = [{"b": 1, "c": 2}, {"b": 3, "c": 4}, {"b": 5, "c": 6}]
        result = merge_with_reduce(
            dicts, lambda x, y: [x] + [y] if not isinstance(x, list) else x + [y]
        )
        assert result == {"b": [1, 3, 5], "c": [2, 4, 6]}

    def test_merge_dicts_with_power(self):
        dicts = [{"b": 1, "c": 2}, {"b": 3, "c": 4}, {"b": 5, "c": 6}]
        result = merge_with_reduce(dicts, lambda x, y: x**y)
        assert result == {"b": 1, "c": 16777216}
