from ..config import configuration, Configuration
from ..utils.logging import logger_wraps
from .unet import _GenericUNet

import keras


@logger_wraps(level="INFO")
def MCDropoutUNet(config: Configuration = configuration()) -> keras.Model:
    return _GenericUNet(mc_dropout=True, config=config)
