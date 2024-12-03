"""
Functions for logging
"""

import inspect
import logging
import sys
import warnings
from enum import auto
from functools import wraps
from typing import TextIO

from loguru import logger

from ..config import auto_match_config


class __InterceptHandler(logging.Handler):
    """
    Intercept standard logging messages and redirect them to Loguru logger
    """

    def emit(self, record: logging.LogRecord) -> None:
        # Get corresponding Loguru level if it exists.
        level: str | int
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Find caller from where originated the logged message.
        frame, depth = inspect.currentframe(), 0
        while frame and (depth == 0 or frame.f_code.co_filename == logging.__file__):
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )


@auto_match_config(prefixes=["logger"])
def config_logger(
    sink: str | TextIO = sys.stderr,
    format: str = "{time:YYYY-MM-DD at HH:mm:ss} {level} {message}",
    level: str = "INFO",
    backtrace: bool = True,
    diagnose: bool = True,
    retention: str | None = None,
):
    """
    Configure loguru logger settings and set it as as the default logger
    """
    logger.remove()
    logger.add(
        sink,  # type: ignore
        format=format,
        level=level,
        retention=retention,
        backtrace=backtrace,
        diagnose=diagnose,
    )

    # Intercept standard logging messages and redirect them to Loguru logger
    logging.basicConfig(handlers=[__InterceptHandler()], level=0, force=True)
    warnings.showwarning = lambda msg, *args, **kwargs: logger.warning(msg)


def logger_wraps(*, entry=True, exit=True, level="DEBUG"):
    """
    Logs entry and exit of a function
    """

    def wrapper(func):
        name = func.__name__

        @wraps(func)
        def wrapped(*args, **kwargs):
            result = None
            logger_ = logger.opt(depth=1)
            if entry:
                logger.log(
                    level,
                    f"Entering '{name}' (args={args}, kwargs={kwargs})",
                )
                result = func(*args, **kwargs)
            if exit:
                logger.log(level, f"Exiting '{name}' (result={result})")
            return result

        return wrapped

    return wrapper
