"""
Collection of common utility functions
"""

from typing import Callable, Generator, Optional, TypeVar

from .wrappers import curry



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
def apply_if_truthy[T, R](func: Callable[[T], R], arg: T) -> Optional[R]:
    """
    Apply unary func to arg if arg is truthy (e.g. not None, [], ...), otherwise return None
    """
    return func(arg) if arg else None


@curry
def iterate_while[T, R](
    func: Callable[[T], R], condition: Callable[[T], bool], initial: T
) -> R | T:
    """
    Repeatedly apply func to a value until condition is met

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
        iterate_while(func, condition, func(initial))
        if not condition(initial)
        else initial
    )

