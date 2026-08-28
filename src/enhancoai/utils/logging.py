"""Centralised logging configuration for EnhancoAI."""

from __future__ import annotations

import logging
import sys

_CONFIGURED = False


def get_logger(name: str = "enhancoai", level: int = logging.INFO) -> logging.Logger:
    """Return a configured logger, initialising handlers only once per process."""
    global _CONFIGURED
    root = logging.getLogger("enhancoai")
    if not _CONFIGURED:
        root.setLevel(level)
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        root.addHandler(handler)
        root.propagate = False
        _CONFIGURED = True
    return logging.getLogger(name)
