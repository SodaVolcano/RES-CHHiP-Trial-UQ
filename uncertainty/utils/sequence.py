"""
Utility functions for generating sequences
"""

from itertools import islice
from typing import Callable, Generator, Iterable, Iterator

import toolz as tz
import toolz.curried as curried

from .wrappers import curry


@curry
def growby[
    T, R
](init: T, f: Callable[[T | R], R], length: int | None = None,) -> Iterator[T | R]:
    """
    Grow a sequence by repeatedly applying f to last item in sequence

    Parameters
    ----------
    f: Callable
        Function to be called repeatedly on the last element of the sequence
    init: any
        Initial value of the sequence
    length: int | None
        Length of the sequence to be generated, if None, sequence is infinite

    Returns
    -------
    Generator[T | R, None, None]
        Sequence constructed by repeatedly applying f, `[init, f(init), f(f(init)), ...]`
    """
    return islice(tz.iterate(f, init), length)


@curry
def growby_accum[
    T, R
](init: T, fs: Callable[[T | R], R]) -> Generator[T | R, None, None]:
    """
    Grow a sequence by applying list of functions to the last element of the current sequence

    Parameters
    ----------
    fs: Callable
        List of functions to be called on the last element of the sequence
    init: any
        Initial value of the sequence

    Returns
    -------
    Generator[T | R, None, None]
        Sequence constructed by repeatedly applying each f in fs,
        `[init, f1(init), f2(f1(init)), ...]`
    """
    return tz.pipe(
        fs,
        curried.cons(init),  # [init, f1, f2, ...]
        curried.accumulate(lambda x, f: f(x)),
    )  # type: ignore


@curry
def transform_nth(n: int, func: Callable, seq: Iterable) -> Iterable:
    """
    Apply a function to the nth element of a sequence

    Parameters
    ----------
    n: int
        Index of the element to be transformed
    func: Callable
        Function to be applied to the nth element
    seq: Iterable
        Sequence to be transformed

    Returns
    -------
    Iterable
        Sequence with the nth element transformed
    """
    return (func(x) if i == n else x for i, x in enumerate(seq))
