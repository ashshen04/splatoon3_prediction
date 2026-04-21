"""Central logging configuration.

Every module should call `get_logger(__name__)` instead of using print().
Logs go to stdout and to `logs/pipeline.log` with rotation.
"""

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

_LOG_DIR = Path("logs")
_LOG_FILE = _LOG_DIR / "pipeline.log"
_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_DATEFMT = "%H:%M:%S"

_configured = False


def configure_logging(level: int = logging.INFO) -> None:
    """Idempotent root-logger setup. Safe to call multiple times."""
    global _configured
    if _configured:
        return

    _LOG_DIR.mkdir(exist_ok=True)

    root = logging.getLogger()
    root.setLevel(level)
    # Wipe any pre-existing handlers so we don't double-log
    root.handlers.clear()

    formatter = logging.Formatter(_FORMAT, datefmt=_DATEFMT)

    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    root.addHandler(stream)

    file_handler = RotatingFileHandler(
        _LOG_FILE, maxBytes=10_000_000, backupCount=3, encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    # Quiet down noisy libraries
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)

    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Return a logger. Auto-configures on first call."""
    configure_logging()
    return logging.getLogger(name)
