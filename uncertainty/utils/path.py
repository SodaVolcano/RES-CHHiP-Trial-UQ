"""
Utility functions for working with file paths
"""

from .wrappers import curry
from .common import iterate_while
from .string import placeholder_matches

from functools import reduce
import os
import re
from typing import Callable, Generator, Iterable

import toolz.curried as curried
import toolz as tz


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


@curry
def resolve_path_placeholders(path_pattern: str, placeholders: list[str]) -> list[str]:
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
            os.path.dirname,
            lambda s: re.search(f"{{{VALID_IDENTIFIER}}}", s) is not None,
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
