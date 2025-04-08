from typing import Union
import logging

import colorlog

from .about import PACKAGE_NAME

root_logger = logging.getLogger(PACKAGE_NAME)

_handler = colorlog.StreamHandler()
_formatter = colorlog.ColoredFormatter(
    "%(log_color)s[%(levelname)s][%(pathname)s:%(lineno)s]%(reset)s %(blue)s%(message)s",
    datefmt=None,
    reset=True,
    log_colors={
        "DEBUG": "cyan",
        "INFO": "green",
        "WARNING": "yellow",
        "ERROR": "red",
        "CRITICAL": "red,bg_white",
    },
    secondary_log_colors={},
    style="%",
)
if not root_logger.hasHandlers():
    _handler.setFormatter(_formatter)
    root_logger.addHandler(_handler)

test_logger = root_logger.getChild("test")


def set_logging_verbosity(level: Union[int, str]):
    if isinstance(level, str):
        level = getattr(logging, level.upper())
    root_logger.setLevel(level)
    for handler in root_logger.handlers:
        handler.setLevel(level)

    root_logger.info(f"Logging verbosity set to {level}")
