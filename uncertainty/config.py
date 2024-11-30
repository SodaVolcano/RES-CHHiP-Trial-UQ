"""
Reads configuration from a yaml file and returns a dictionary of configuration settings

A configuration object is a dictionary where each key has the format <prefix>__<parameter>.
"""

from functools import cache, wraps, reduce
import re
import inspect
import sys
import yaml
from torch import nn, optim
import toolz as tz
from toolz import curried

_with_prefix = lambda prefix, dict_: tz.keymap(lambda k: f"{prefix}__{k}", dict_)


@cache
def data_config(config_path: str = "configuration.yaml") -> dict:
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return _with_prefix("data", config["data"])


@cache
def unet_config(config_path: str = "configuration.yaml") -> dict:
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    # Convert from string to function
    config["unet"]["activation"] = getattr(nn, config["unet"]["activation"])
    config["unet"]["final_layer_activation"] = getattr(
        nn, config["unet"]["final_layer_activation"]
    )

    return _with_prefix("unet", config["unet"])


def confidnet_config(config_path: str = "configuration.yaml") -> dict:
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    config["confidnet"]["activation"] = getattr(nn, config["unet"]["activation"])
    config["confidnet"]["final_layer_activation"] = getattr(
        nn, config["unet"]["final_layer_activation"]
    )
    return _with_prefix("confidnet", config["confidnet"])


@cache
def logger_config(config_path: str = "configuration.yaml") -> dict:
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    config = config["logger"]

    # Replace string with corresponding file object if present
    mapping = {"stdout": sys.stdout, "stderr": sys.stderr}
    config["sink"] = mapping.get(config["sink"], config["sink"])

    # No retention for stdout and stderr
    if config["sink"] in [sys.stdout, sys.stderr]:
        config["retention"] = config.get("retention", None)
    return _with_prefix("logger", config)


@cache
def training_config(config_path: str = "configuration.yaml") -> dict:
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    # Convert from string to function
    config["training"]["initialiser"] = getattr(
        nn.init, config["training"]["initialiser"]
    )
    config["training"]["optimiser"] = getattr(optim, config["training"]["optimiser"])

    # Convert from string to function that pass in the optimiser
    lr_scheduler = getattr(optim.lr_scheduler, config["training"]["lr_scheduler"])
    config["training"]["lr_scheduler"] = lambda optimiser: lr_scheduler(
        optimiser, **config["training"]["lr_scheduler_kwargs"]
    )

    return _with_prefix("training", config["training"])


def configuration(config_path: str = "configuration.yaml") -> dict:
    """
    The entire configuration for the project
    """
    return reduce(
        tz.merge,
        [
            data_config(config_path),
            unet_config(config_path),
            logger_config(config_path),
            training_config(config_path),
            confidnet_config(config_path),
        ],
    )


def auto_match_config(*, prefixes: list[str] = []):
    """
    Automatically pass configuration values to function parameters

    A configuration object is a dictionary where each key has the format
    `<prefix>__<parameter>.` Prefixes are stripped before being passed to
    the function parameters. Subset of the dictionary can be selected
    by specifying the prefixes of the keys to select.

    Note that if a function have the parameter `kwargs`, the remaining
    configuration dictionary after passing the values to the other
    parameters will be passed to the `kwargs` parameter. This is useful
    for passing the configuration dictionary to inner functions.

    If a function is called with explicit keyword arguments with the same
    name in the configuration dictionary, the explicit keyword argument
    will override the value in the configuration dictionary. E.g. `a=10` in
    `my_func(a=10, **config)` will take precedence over `config["a"]`.

    Parameters
    ----------
    prefixes : str (optional)
        Beginning string of the keys delimited by "__" in the configuration
        dictionary. If specified, only the keys that have the prefix
        are passed to the function. If two keys have the same prefix,
        the value in the dictionary that appears later in the configuration
        dictionary is used.

    Examples
    --------
    >>> @auto_match_config(prefixes=["1"])
    ... def test2(c, d):
    ...    print(c, d)
    >>> @auto_match_config(prefixes=["1", "2"])
    ... def test(a, b, **kwargs):
    ...    test2(**kwargs)
    ...    print(a + b)
    >>> config = {"1__c": 30, "1__d": 4, "2__a": 10, "2__b": 2000}
    >>> test(b=5, **config)  # b=5 overrides config["2__b"]
    30 4
    15
    """

    def wrapper(func):

        @wraps(func)
        def wrapped(*args, **kwargs):
            strip_prefix = lambda k: re.sub(r"^[^_]*__", "", k)

            params = inspect.signature(func).parameters
            # length 1 means key don't begin with a prefix, hence not from config
            non_config_kwargs = tz.keyfilter(lambda k: len(k.split("__")) == 1, kwargs)
            filtered_kwargs = tz.pipe(
                kwargs,
                # all keys with a prefix (i.e. from config)
                curried.keyfilter(lambda k: len(k.split("__")) > 1),
                (
                    curried.keyfilter(lambda k: k.split("__")[0] in prefixes)
                    if prefixes
                    else tz.identity
                ),
                # let non_config_kwargs override config values
                lambda config_kwargs: tz.merge(config_kwargs, non_config_kwargs),
                curried.keymap(strip_prefix),
                (
                    curried.keyfilter(lambda k: k in params)
                    if "kwargs" not in params
                    else tz.identity  # don't filter, pass rest of args to kwargs
                ),
            )

            return func(*args, **filtered_kwargs)

        return wrapped

    return wrapper
