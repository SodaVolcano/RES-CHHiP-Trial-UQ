"""
Functions for parallel processing.
"""

from multiprocessing.pool import IMapIterator  # purely for typing
from typing import Any, Callable, Generator, Literal, Optional

from pathos.multiprocessing import ProcessingPool, ThreadingPool

from .wrappers import curry


@curry
def pmap(
    f: Callable[..., Any],
    iterable: Any,
    *iterables: Any,
    n_workers: Optional[int] = None,
    executor: Literal["process", "thread"] = "process",
) -> IMapIterator | Generator[Any, None, None] | Any:
    """
    Parallel map function using Process or Thread pool

    Parameters
    ----------
    f: Callable
        Function to apply to each element of the iterable.
    n_workers: Optional[int]
        Number of workers to use. If None, the number of workers is set to the number of CPUs.
    executor: Literal["process", "thread"]
        Executor to use, process or thread workers.
    """
    Pool = ProcessingPool if executor == "process" else ThreadingPool
    with Pool(n_workers) as pool:
        results = pool.imap(f, iterable, *iterables)
    return results
