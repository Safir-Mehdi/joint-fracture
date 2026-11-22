import logging
import sys
from pathlib import Path
from datetime import datetime


def get_logger(name: str, log_file: str | Path | None = None,
               level: str = "INFO") -> logging.Logger:
    """
    Get a configured logger.

    Args:
        name: logger name (usually __name__)
        log_file: optional file path — if given, logs to file AND console
        level: "DEBUG" | "INFO" | "WARNING" | "ERROR"

    Returns:
        logging.Logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level))

    fmt = logging.Formatter(
        "[%(asctime)s] %(levelname)-5s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler — add only if not present
    has_console = any(
        isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
        for h in logger.handlers
    )
    if not has_console:
        console = logging.StreamHandler(sys.stdout)
        console.setFormatter(fmt)
        logger.addHandler(console)

    # File handler — add if log_file given AND no file handler present
    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        has_file = any(
            isinstance(h, logging.FileHandler) for h in logger.handlers
        )
        if not has_file:
            file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
            file_handler.setFormatter(fmt)
            logger.addHandler(file_handler)

    return logger

def close_handlers(name: str | None = None) -> None:
    """
    Close logger handlers to release file locks.

    Call this at the end of training or in tests to allow
    deleting / moving log files (required on Windows).

    Args:
        name: close handlers for a specific logger.
              If None, closes handlers on ALL loggers.
    """
    if name is not None:
        _close_one(name)
    else:
        for logger_name in list(logging.Logger.manager.loggerDict.keys()):
            _close_one(logger_name)


def _close_one(name: str) -> None:
    """Internal — close + remove all handlers on one logger."""
    logger = logging.getLogger(name)
    for handler in logger.handlers[:]:
        try:
            handler.close()
        except Exception:
            pass
        logger.removeHandler(handler)
