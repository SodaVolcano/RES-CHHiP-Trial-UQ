"""
Reads configuration from a yaml file and returns a dictionary of configuration settings
"""

from functools import cache

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
    return config["logger"]


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


def configuration() -> dict:
    """
    Preset configuration for U-Net model
    """
    return data_config() | unet_config() | training_config() | logger_config()  # type: ignore
