"""
Wrappers for functions
"""

from functools import wraps
from typing import Callable

from toolz import curry as _curry


def curry(func: Callable) -> Callable:
    """
    Partially parameterise a function.

    A wrapper that passes docstring of the wrapped function
    to the @curry decorator from toolz.
    """

    @wraps(func)
    def curried(*args, **kwargs) -> Callable:
        return _curry(func)(*args, **kwargs)  # type: ignore

    return curried
