"""
Overload for embarrassingly parallel tasks
"""

from typing import Literal, Optional
from .wrappers import curry
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from pathos.multiprocessing import ProcessingPool, ThreadingPool


@curry
def pmap(
    f,
    iterable,
    *iterables,
    n_workers: Optional[int] = None,
    executor: Literal["pool", "thread"] = "pool",
) -> list:
    """
    Parallel map function using Pool or Thread

    If `executor` is `"pool"`, `n_workers` specifies the number of processes to use.
    If `executor` is `"thread"`, `n_workers` specifies the number of threads to use.
    """
    pool = ProcessingPool if executor == "pool" else ThreadingPool
    return pool(n_workers).map(f, iterable, *iterables)
