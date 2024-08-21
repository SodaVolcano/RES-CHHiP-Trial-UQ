"""
Overload for embarrassingly parallel tasks
"""

from multiprocessing.pool import IMapIterator
from typing import Any, Generator, Literal, Optional

from pathos.multiprocessing import ProcessingPool, ThreadingPool

from .wrappers import curry


@curry
def pmap(
    f,
    iterable,
    *iterables,
    n_workers: Optional[int] = None,
    executor: Literal["process", "thread"] = "process",
) -> IMapIterator | Generator[Any, None, None] | Any:
    """
    Parallel map function using Process or Thread pool

    If `executor` is `"process"`, `n_workers` specifies the number of processes to use.
    If `executor` is `"thread"`, `n_workers` specifies the number of threads to use.
    """
    pool = ProcessingPool if executor == "process" else ThreadingPool
    return pool(n_workers).imap(f, iterable, *iterables)
