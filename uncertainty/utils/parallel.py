"""
Overload for embarrassingly parallel tasks
"""

from typing import Literal, Optional
from .wrappers import curry
from multiprocessing import Pool
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

from toolz.curried import map


@curry
def pmap(
    f,
    iterable,
    *iterables,
    n_workers: Optional[int] = None,
    executor: Literal["pool", "thread"] = "pool",
):
    """
    Parallel map function using Pool or Thread

    If `executor` is `"pool"`, `n_workers` specifies the number of processes to use.
    If `executor` is `"thread"`, `n_workers` specifies the number of threads to use.
    """
    executor = ProcessPoolExecutor if executor == "pool" else ThreadPoolExecutor
    return executor(n_workers).map(f, iterable, *iterables)
