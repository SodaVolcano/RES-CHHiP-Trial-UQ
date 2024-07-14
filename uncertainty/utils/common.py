"""
Collection of common utility functions
"""

from itertools import takewhile
from typing import Callable, Generator, Optional, TypeVar

from .wrappers import curry

import toolz as tz

def yield_val[T](func: Callable[..., T], *args, **kwargs) -> Generator[T, None, None]:
    """
    Yield the result of a function, delaying execution until the generator is iterated over
    """
    yield func(*args, **kwargs)


def unpack_args[T](func: Callable[..., T]) -> Callable[..., T]:
    """
    Unpack args from a tuple and pass them to func
    """
    return lambda args: func(*args)


@curry
def conditional[T, R](cond: bool, val1: T, val2: Optional[R] = None) -> Optional[T | R]:
    """
    Return val1 if cond is True, otherwise return val2 (default None)
    """
    return val1 if cond else val2


@curry
def apply_if_truthy[T, R, Q](func: Callable[[T], R], arg: T, val2: Optional[Q] = None) -> Optional[R | Q]:
    """
    Apply unary func to arg if arg is truthy (e.g. not None, [], ...), otherwise return val2
    """
    return func(arg) if arg else val2


@curry
def iterate_while[T, R](
    func: Callable[[T], R], pred: Callable[[T | R], bool], initial: T
) -> R | T:
    """
    Repeatedly apply func to a value until predicate(value) is false

    Parameters
    ----------
    func: Callable
        Function to be called repeatedly
    condition: Callable
        Function that takes the output of func and returns a boolean
    initial: any
        Initial value to be passed to func

    Returns
    -------
    any
        Output of func when condition is met
    """
    
    return (
        iterate_while(func, pred, func(initial))
        if pred(initial)
        else initial
    )

