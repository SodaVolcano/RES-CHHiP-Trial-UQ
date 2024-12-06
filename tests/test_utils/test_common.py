from ..context import utils

iterate_while = utils.iterate_while
unpacked_map = utils.unpacked_map


class TestIterateWhile:

    # Function correctly applies func repeatedly until pred returns False
    def test_func_applies_until_pred_false(self):

        def func(x):
            return x + 1

        def pred(x):
            return x < 5

        result = iterate_while(func, pred, 0)
        assert result == 5

    # Handles infinite loops gracefully if pred never returns False
    def test_handles_infinite_loops_gracefully(self):

        import pytest

        def func(x):
            return x + 1

        def pred(x):
            return True

        with pytest.raises(RecursionError):
            iterate_while(func, pred, 0)

    # Returns initial value if pred returns False on the first call
    def test_returns_initial_value_if_pred_false_on_first_call(self):

        def func(x):
            return x + 1

        def pred(x):
            return x < 0

        result = iterate_while(func, pred, 10)
        assert result == 10


class TestUnpackedMap:

    # Function correctly maps over iterable of tuples and unpacks arguments
    def test_unpacks_and_maps_tuples(self):
        def add(x, y, z):
            return x + y + z

        input_tuples = [(1, 2, 5), (3, 4, 3), (5, 6, 8)]
        result = list(unpacked_map(add, input_tuples))

        assert result == [8, 10, 19]

    def test_curried_unpacks_and_maps_tuples(self):
        def add(x, y, z):
            return x + y + z

        input_tuples = [(1, 2, 5), (3, 4, 3), (5, 6, 8)]
        result = list(unpacked_map(add)(input_tuples))

        assert result == [8, 10, 19]
