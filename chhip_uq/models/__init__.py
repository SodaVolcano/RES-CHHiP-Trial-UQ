from . import unet_modules
from .confidnet import UNetConfidNet
from .ensemble import DeepEnsemble
from .mcdo import MCDropout
from .unet import UNet

__all__ = [
    "MCDropout",
    "UNet",
    "DeepEnsemble",
    "UNetConfidNet",
    "unet_modules",
]


def get_model(name: str):
    return {
        "unet": UNet,
        "confidnet": UNetConfidNet,
    }.get(name, None)
