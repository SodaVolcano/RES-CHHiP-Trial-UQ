"""
Collection of misc utility functions
"""

import pickle
from typing import Any, Callable, Iterable, List, Generator, TypeVar
import os
from functools import reduce, wraps
import re

import toolz as tz
import toolz.curried as curried
from toolz import curry as _curry


def curry(func: Callable) -> Callable:
    """
    Pass docstring to the @curry decorator from toolz
    """

    @wraps(func)
    def curried(*args, **kwargs):
        return _curry(func)(*args, **kwargs)

    return curried


T = TypeVar("T")
R = TypeVar("R")


def yield_val(func: Callable[..., T], *args, **kwargs) -> Generator[T, None, None]:
    """
    Yield the result of a function, delaying execution until the generator is iterated over
    """
    yield func(*args, **kwargs)


@curry
def list_files(path: str, list_hidden: bool = False) -> list[str]:
    """
    Return list of files in a given path

    Parameters
    ----------
    list_hidden: bool
        Whether to include hidden files (starting with a dot '.')
    """
    return [
        os.path.join(root, file)
        for root, dirs, files in os.walk(path)
        for file in files
        if list_hidden or not file.startswith(".")
    ]


@curry
def generate_full_paths(
    root: str, path_generator: Callable
) -> Generator[str, None, None]:
    """
    Concatenate root onto paths generated by path_generator
    """
    return (os.path.join(root, path) for path in path_generator(root))


def unpack_args(func: Callable[..., T]) -> Callable[..., T]:
    """
    Unpack args from a tuple and pass them to func
    """
    return lambda args: func(*args)


@curry
def apply_if_truthy_else_None(func: Callable[[T], R], arg: T) -> R | None:
    """
    Apply unary func to arg if arg is truthy (e.g. not None, [], ...), otherwise return None
    """
    return func(arg) if arg else None


@curry
def save_pickle(path: str, obj: Any) -> None:
    """
    Save object to a pickle file
    """
    with open(path, "wb") as file:
        pickle.dump(obj, file)


@curry
def capture_placeholders(
    s: str, placeholders: List[str], re_pattern: str = r".*?"
) -> str:
    """
    Replace placeholders in a string `s` with `"(re_pattern)"`

    Placeholders are strings in the form of ``"{placeholder}"``. They can be any
    combination of numbers, characters, and underscore. Placeholders in `s`
    but not in the list of `placeholders` will not be encased in parentheses
    (i.e., not in a capturing group) but will still be replaced.

    Parameters
    ----------
    s : str
        String containing placeholders, e.g. ``"somestuff{a}_{b}.nii.gz"``.
    placeholders : list[str]
        List of placeholders to match in the pattern, e.g. ``["a", "b"]``.
    re_pattern : str, optional
        Regex pattern to replace placeholders with, by default any valid Python
        identifier. Matches any character except line terminators by default.

    Returns
    -------
    str
        String with placeholders replaced by the specified `re_pattern`.
    """

    return tz.pipe(
        [s] + placeholders,
        # Replace placeholders to avoid them being escaped
        curried.reduce(
            lambda string, placeholder: string.replace("{" + placeholder + "}", "\x00")
        ),
        # Replace all non-capturing placeholders with different symbol
        lambda string: re.sub(r"{[a-zA-Z0-9_]*}", "\x01", string),
        re.escape,
        # Encase provided placeholders in parentheses to create capturing groups
        lambda string: string.replace("\x00", f"({re_pattern})"),
        lambda string: string.replace("\x01", re_pattern),
    )


@curry
def placeholder_matches(
    str_list: list[str], pattern: str, placeholders: list[str], re_pattern: str = r".*?"
) -> list[tuple[str, ...]]:
    """
    Return placeholder values for each string in a list that match pattern

    Placeholders are string in the form of "{placeholder}". They can be any
    combination of numbers, characters, and underscore. Placeholders not in
    pattern will still be matched but not captured in the output.

    Parameters
    ----------
    str_list: list[str]
        List of strings to match against the pattern
    pattern: str
        Pattern containing placeholders to match file names, e.g.
        `"/path/to/{organ}_{observer}.nii.gz"`
    placeholders: list[str]
        List of placeholders to match in the pattern, e.g. `["organ", "observer"]`
    re_pattern: str, optional
        Regex pattern filter placeholder matches by, by default any character
        except line terminators
    Returns
    -------
    set[tuple[str, ...]]
        List of tuples containing the matches for each placeholder.

    Example
    -------
    ```
    placeholder_matches(
        [
            "/path/to/bladder_jd.nii.gz",
            "/path/to/brain_md.nii.gz",
            "/path/to/eye_sp.nii.gz",
        ],
        "/path/to/{organ}_{observer}.nii.gz",
        ["organ", "observer"],
    )
    # Output
    [("bladder", "jd"), ("brain", "md"), ("eye", "sp")]

        placeholder_matches(
        [
            "/path/to/bladder_jd.nii.gz",
            "/path/to/brain_md.nii.gz",
            "/path/to/eye_sp.nii.gz",
        ],
        "/path/to/{organ}_{observer}.nii.gz",
        ["organ", "observer"],
        r"[^b]+",    # Any string that does not contain the letter 'b'
    )
    # Output
    [("eye", "sp")]

    ```
    """
    return tz.pipe(
        str_list,
        curried.map(
            lambda string: re.match(
                capture_placeholders(pattern, placeholders, re_pattern), string
            ),
        ),
        curried.filter(lambda match: match is not None),
        curried.map(lambda re_match: re_match.groups()),
        list,
    )


@curry
def iterate_while(func: Callable, condition: Callable, initial: any) -> any:
    """
    Iterate over func until condition is met

    Parameters
    ----------
    func: Callable
        Function to be iterated over
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


@curry
def resolve_path_placeholders(
    path_pattern: str, placeholders: list[str]
) -> Generator[str, None, None]:
    """
    Search directory using path_pattern and resolve placeholders with actual values

    Parameters
    ----------
    path_pattern: str
        Pattern to match directory paths, e.g. "/a/{b}/{c}"
    placeholders: list[str]
        List of placeholders to match in the pattern, e.g. ["b", "c"]

    Returns
    list[str]
        List of paths with placeholders replaced with actual values

    Example
    -------
    ```
    resolve_path_placeholders("/a/{b}/{c}/d/{e}.jpg", ["b", "c"])
    # Output
    ["/a/val1/val2/d/{e}.jpg", "/a/val3/val4/d/{e}.jpg", ...]
    -------
    """
    VALID_IDENTIFIER = r"[a-zA-Z0-9_]*"
    if not placeholders:
        return [path_pattern]

    def placeholder_in_list() -> Iterable[str]:
        """
        Find all placeholders in the path pattern that are in the list
        """
        return filter(
            lambda placeholder: placeholder[1:-1] in placeholders,
            re.findall(f"{{{VALID_IDENTIFIER}}}", path_pattern),
        )

    return tz.pipe(
        path_pattern,
        # Longest directory path excluding placeholders, e.g. "/a/{b}/c" -> "/a"
        iterate_while(
            os.path.dirname, lambda s: re.search(f"{{{VALID_IDENTIFIER}}}", s) is None
        ),
        list_files,
        placeholder_matches(pattern=path_pattern, placeholders=placeholders),
        # Zip matches with placeholders, e.g. for path "/a/{b}/{c}" and matches ["1"], return ("{b}", "1")
        curried.map(
            lambda matches: zip(
                placeholder_in_list(),
                matches,
            ),
        ),
        # For each match list, get string by replacing placeholders with actual values
        curried.map(
            lambda zip_matches: reduce(
                lambda path, match_: path.replace(*match_), zip_matches, path_pattern
            )
        ),
        list,
    )


# Prevent polluting the namespace
del (
    pickle,
    Any,
    Callable,
    Iterable,
    List,
    Generator,
    TypeVar,
    os,
    reduce,
    wraps,
    re,
    tz,
    curried,
    _curry,
)