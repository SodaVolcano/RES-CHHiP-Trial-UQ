"""
Overload for embarrassingly parallel tasks
"""

from typing import Generator, Literal, Optional
from .wrappers import curry
from pathos.multiprocessing import ProcessingPool, ThreadingPool


@curry
def pmap(
    f,
    iterable,
    *iterables,
    n_workers: Optional[int] = None,
    executor: Literal["pool", "thread"] = "pool",
) -> Generator:
    """
    Parallel map function using Pool or Thread

    If `executor` is `"pool"`, `n_workers` specifies the number of processes to use.
    If `executor` is `"thread"`, `n_workers` specifies the number of threads to use.
    """
    pool = ProcessingPool if executor == "pool" else ThreadingPool
    return pool(n_workers).imap(f, iterable, *iterables)
