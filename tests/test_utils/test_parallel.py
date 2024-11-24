from ..context import utils
import time

pmap = utils.pmap


class TestPmap:
    # Test that pmap applies the function to each element of the iterable
    def test_applies_function_to_iterable(self):
        def func(x):
            return x * 2

        iterable = [1, 2, 3, 4]
        result = list(pmap(func, iterable, n_workers=2, executor="thread"))
        assert result == [2, 4, 6, 8]

    # Test that pmap works with multiple iterables
    def test_applies_function_to_multiple_iterables(self):
        def func(x, y):
            return x + y

        iterable1 = [1, 2, 3]
        iterable2 = [4, 5, 6]
        result = list(pmap(func, iterable1, iterable2, n_workers=3, executor="process"))
        assert result == [5, 7, 9]

    # Test that pmap utilises parallelism to match the expected runtime
    def test_runtime_matches_parallel_execution(self):
        def func(x):
            time.sleep(2)
            return x * 2

        iterable = [1, 2, 3, 4]
        n_workers = 4
        start_time = time.time()

        result = list(pmap(func, iterable, n_workers=n_workers, executor="process"))
        end_time = time.time()

        # Ensure the results are correct
        assert result == [2, 4, 6, 8]

        # Ensure the total runtime matches the sleep time (~1 second)
        runtime = end_time - start_time
        assert runtime < 2.5, f"Runtime too long: {runtime} seconds"
