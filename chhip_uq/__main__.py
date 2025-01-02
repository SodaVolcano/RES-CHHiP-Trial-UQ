from loguru import logger

from .config import configuration
from .utils import config_logger

logger.enable("chhip_uq")
config_logger(**configuration())  # type: ignore
