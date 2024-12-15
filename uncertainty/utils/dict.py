from functools import reduce
from typing import Callable

import toolz as tz

from .common import unpack


def merge_with_reduce[
    K, V, V2
](dicts: list[dict[K, V]], func: Callable[[V | V2, V], V2]) -> dict[K, V] | dict[K, V2]:
    """
    Reductively merge a list of dictionaries with the same keys into a single dictionary.

    Reduced version of `toolz.merge_with`.

    Parameters
    ----------
    dicts : list[dict[K, V]]
        List of dictionaries to merge, all with the same keys.
    func : Callable[[V | V2, V], V2]
        Reduction function taking two values and returning a single value.

    Examples
    --------
    >>> dicts = [{"b": 1, "c": 2}, {"b": 3, "c": 4}, {"b": 5, "c": 6}]
    >>> merge_with_reduce(dicts, lambda x, y: [x] + [y] if not isinstance(x, list) else x + [y])
    {'b': [1, 3, 5], 'c': [2, 4, 6]}
    >>> merge_with_reduce(dicts, lambda x, y: x ** y)
    {'b': 1, 'c': 16777216}
    """
    merge_dicts = lambda dict1, dict2: tz.merge_with(unpack(func), dict1, dict2)
    return reduce(merge_dicts, dicts)
