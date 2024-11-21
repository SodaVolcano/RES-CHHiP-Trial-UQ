"""
A monad is simply a monoid in the category of endofunctors :P
"""

from typing import Callable
import functools


class Maybe[T]:
    def __init__(self, value: T):
        self.value = value

    def __repr__(self) -> str:
        return f"Maybe({self.value})"

    def get(self) -> T:
        return self.value


def with_maybe[T](transform: Callable[[T], Maybe[T]]) -> Callable:
    @functools.wraps(transform)
    def wrapper(input_: Maybe[T]) -> Maybe[T]:
        if input_.value is None:
            return Maybe(None)
        return Maybe(transform(input_.value))

    return wrapper


class Maybe:
    def __init__(self, value):
        self.value = value

    def is_nothing(self):
        return self.value is None

    def map(self, func):
        if self.is_nothing():
            return self
        try:
            return Maybe(func(self.value))
        except Exception:
            return Maybe(None)

    def __repr__(self):
        return f"Nothing" if self.is_nothing() else f"Just({self.value})"


def maybe(func):
    """
    A decorator to wrap a function into the Maybe monad.
    If the function raises an exception or returns None, it results in Nothing.
    """

    def wrapper(*args, **kwargs):
        try:
            result = func(*args, **kwargs)
            return Maybe(result)
        except Exception:
            return Maybe(None)

    return wrapper
