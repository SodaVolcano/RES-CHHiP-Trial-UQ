from tests.context import uncertainty

iterate_while = uncertainty.utils.common.iterate_while


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
