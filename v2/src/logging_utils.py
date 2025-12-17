# src/logging_utils.py

from __future__ import annotations

import logging
import sys
from typing import Optional


def get_logger(
    name: str,
    level: int = logging.INFO,
    stream: Optional[object] = None,
) -> logging.Logger:
    """
    Create and configure a logger with a simple, consistent format.

    Parameters
    ----------
    name:
        Logger name (usually __name__ of the module).
    level:
        Logging level (default: logging.INFO).
    stream:
        Stream to write logs to. Defaults to sys.stdout.

    Returns
    -------
    logger : logging.Logger
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        # Logger already configured
        logger.setLevel(level)
        return logger

    logger.setLevel(level)
    handler = logging.StreamHandler(stream or sys.stdout)
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False
    return logger
