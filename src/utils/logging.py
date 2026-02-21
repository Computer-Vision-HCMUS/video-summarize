"""Logging setup for training and inference."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import List, Optional


def setup_logging(
    level: int = logging.INFO,
    log_file: Optional[str | Path] = None,
    format_string: Optional[str] = None,
) -> None:
    """
    Configure root logger: console and optional file.
    """
    fmt = format_string or "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    handlers: List[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(
        level=level,
        format=fmt,
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
        force=True,
    )


def get_logger(name: str) -> logging.Logger:
    """Return a named logger."""
    return logging.getLogger(name)
