"""
Wrappers for functions
"""

import inspect
from functools import wraps
from typing import Callable

import toolz as tz
from toolz import curry as _curry


def curry[T](f: Callable[..., T], fallback: bool = False):
    """
     A curried function `f` can be partially parameterised.

     If a curried function is passed only some of its arguments, a new
     function is returned with those arguments filled in. The new
     function can be called with the remaining arguments later on.

     Evaluation of the function only occurs when all mandatory arguments
     are provided. Unlike `toolz.curry` and similar libraries, this
     function does not use `functools.partial`, meaning decorators of
     the wrapped function will only be applied once all mandatory
     arguments are provided. This means, e.g. a decorator that validates
     the arguments to a function will not throw error that mandatory
     arguments are missing if the function is still partially applied -
     it'll only run when all mandatory arguments are provided.

    Parameters
     ----------
     f : Callable
         The function to curry
     fallback : bool
         If `True`, fallback on `toolz.curry` if `inspect` fails to extract
         parameters from `f`. This is useful for built-in CPython functions.

     Caveats
     -------
     If you use the curried function `f(a, b)` with inputs `f(a=2)(5)`, the
     value `5` will the fill the first positional argument `a`, and you'll get
     a `ValueError` for duplicate values because `a` is also filled by the
     keyword argument `a=2`. This is because `f(a=2)(5)` is equivalent to
     `f(5, a=2)` which will throw an error for non-curried functions as well.

     Note that built-in CPython functions are not supported by the `inspect`
     module. If `fallback` is set to `True`, the function will curry the
     function using `toolz.curry` instead.

     Example
     -------
     >>> @curry
     ... def add(a, b, c=3):
     ...     return a + b + c
     >>> # If not all required parameters are provided, a function is returned
     >>> add_5 = add(5)  # equivalent to lambda b: 5 + b + 3
     >>> add_5(3) # call with remaining argument
     11
     >>> add(5, 3)  # using function normally
     11
    """

    @wraps(f)
    def toolz_curry(*args, **kwargs) -> Callable:
        return _curry(f)(*args, **kwargs)  # type: ignore

    try:
        params = inspect.signature(f).parameters
    except ValueError:
        if fallback:
            return toolz_curry
        raise ValueError(
            "Cannot extract parameters from function. Use `fallback=True` to use `toolz.curry` instead."
        )

    required_args = tz.pipe(
        params,
        tz.curried.valfilter(
            lambda param: param.default is inspect.Parameter.empty
            and param.kind
            # include POSITION_ONLY - if user passed these using keywords, function
            # will throw error, otherwise *args will fill in these parameters anyway
            in {
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
                inspect.Parameter.POSITIONAL_ONLY,
            }
        ),
        list,
    )

    def curried(*args, **kwargs):
        # Evaluate only if mandatory keyword args are provided, OR args fill remaining params
        if not (remaining_args := [k for k in required_args if k not in kwargs]) or len(
            args
        ) >= len(remaining_args):
            return f(*args, **kwargs)

        # Define a function instead of using lambda to let docstring etc be copied over
        @wraps(f)
        def curried_fn(*args2, **kwargs2):
            # unpack both kwargs separately instead of merging into one dict to throw error
            # for duplicate keyword arguments!
            return curried(*args, *args2, **kwargs, **kwargs2)

        return curried_fn

    return curried
