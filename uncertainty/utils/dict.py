from functools import reduce
from typing import Callable, Iterable

import toolz as tz

from .common import star
from .wrappers import curry


@curry
def merge_with_reduce[
    K, V, V2
](dicts: Iterable[dict[K, V]], func: Callable[[V | V2, V], V2]) -> (
    dict[K, V] | dict[K, V2]
):
    """
    Reductively merge a list of dictionaries with the same keys into a single dictionary.

    Reduced version of `toolz.merge_with`.

    Parameters
    ----------
    dicts : Iterable[dict[K, V]]
        Iterable of dictionaries to merge, all with the same keys.
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
    merge_dicts = lambda dict1, dict2: tz.merge_with(star(func), dict1, dict2)
    return reduce(merge_dicts, dicts)


@curry
def rename_key(old_name: str, new_name: str, dictionary: dict) -> dict:
    """
    Rename a key in a dictionary.

    Parameters
    ----------
    old_name : str
        Old key name.
    new_name : str
        New key name.
    dictionary : dict
        Dictionary to rename the key.

    Returns
    -------
    dict
        Dictionary with the key renamed.
    """
    return {
        new_name if key == old_name else key: value for key, value in dictionary.items()
    }
