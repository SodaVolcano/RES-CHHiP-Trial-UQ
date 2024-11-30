from loguru import logger

from .utils import config_logger
from .config import configuration

logger.enable("uncertainty")
config_logger(**configuration())  # type: ignore
