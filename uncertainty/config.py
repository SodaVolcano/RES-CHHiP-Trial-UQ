"""
Reads configuration from a yaml file and returns a dictionary of configuration settings
"""

from functools import cache, wraps
import inspect
import sys
import yaml
from torch import nn, optim


@cache
def data_config(config_path: str = "configuration.yaml") -> dict:
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config["data"]


@cache
def unet_config(config_path: str = "configuration.yaml") -> dict:
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    # Convert from string to function
    config["unet"]["activation"] = getattr(nn, config["unet"]["activation"])
    config["unet"]["final_layer_activation"] = getattr(
        nn, config["unet"]["final_layer_activation"]
    )

    return config["unet"]


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
    return config


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

    return config["training"]


def configuration(config_path: str = "configuration.yaml") -> dict:
    """
    The entire configuration for the project
    """
    return {
        "data": data_config(config_path),
        "unet": unet_config(config_path),
        "logger": logger_config(config_path),
        "training": training_config(config_path),
    }


def auto_match_config(*, prefix: str = ""):
    """
    Automatically pass configuration values to function parameters

    Parameters
    ----------
    prefix : str (optional)
        Key in the configuration dictionary whose value is the
        dictionary of configuration settings to pass to the function.
        E.g. "data", "unet", "logger", "training"
    """

    def wrapper(func):

        @wraps(func)
        def wrapped(*args, **config):
            params = inspect.signature(func).parameters
            config = config[prefix] if prefix else config
            filtered_kwargs = {k: v for k, v in config.items() if k in params}
            return func(*args, **filtered_kwargs)

        return wrapped

    return wrapper
