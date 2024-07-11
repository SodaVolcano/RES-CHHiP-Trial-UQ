from uncertainty.common.constants import model_config
from typing import Callable
import toolz as tz


def conv_block(x, filters, last_block, config: dict = model_config()):
    pass


def unet_encoder():
    pass


def unet_decoder():
    pass


def unet():
    pass


def build_model(
    model_constructor: Callable, batches_per_epoch: int, config: dict = model_config()
):
    pass
