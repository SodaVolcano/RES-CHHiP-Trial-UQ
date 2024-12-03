from loguru import logger

from .config import configuration
from .utils import config_logger

logger.enable("uncertainty")
config_logger(**configuration())  # type: ignore
