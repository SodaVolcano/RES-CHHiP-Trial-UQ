from loguru import logger

from .utils import config_logger
from .config import logger_config

logger.enable("uncertainty")
config_logger(logger_config())  # type: ignore
