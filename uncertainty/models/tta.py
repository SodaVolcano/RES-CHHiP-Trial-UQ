"""
Wrapper around a model that produces multiple outputs using test-time augmentation (TTA)
"""

from re import T
from uncertainty.utils.common import unpack_args
from ..config import Configuration, configuration
from ..utils.logging import logger_wraps
from ..utils.wrappers import curry

from typing import Callable

import tensorflow as tf
from volumentations import Compose
import toolz as tz
