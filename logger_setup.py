"""
Logging configuration module.
"""

import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler
from typing import Optional


class ColoredFormatter(logging.Formatter):
    """Colored log formatter for console output."""

    COLORS = {
        "DEBUG": "\033[36m",
        "INFO": "\033[32m",
        "WARNING": "\033[33m",
        "ERROR": "\033[31m",
        "CRITICAL": "\033[35m",
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelname, self.RESET)
        record.levelname = f"{color}{record.levelname}{self.RESET}"
        return super().format(record)


_logger_initialized = False


def setup_logging(
    log_level: str = "INFO",
    log_dir: Optional[Path] = None,
    enable_file_logging: bool = True,
) -> logging.Logger:
    """Setup logging configuration."""
    global _logger_initialized
    if _logger_initialized:
        return logging.getLogger("g4f_bridge")

    from environment import get_environment

    env = get_environment()
    if log_dir is None:
        log_dir = env.data_dir / "logs"

    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("g4f_bridge")
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    console_format = ColoredFormatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)

    # File handler
    if enable_file_logging:
        try:
            file_handler = RotatingFileHandler(
                log_dir / "bridge.log",
                maxBytes=5 * 1024 * 1024,  # 5MB (lighter)
                backupCount=3,
            )
            file_handler.setLevel(logging.DEBUG)
            file_format = logging.Formatter(
                "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
            file_handler.setFormatter(file_format)
            logger.addHandler(file_handler)
        except Exception as e:
            logger.warning(f"Failed to setup file logging: {e}")

    # Suppress noisy libraries
    for noisy in ("httpx", "httpcore", "asyncio", "urllib3", "g4f"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _logger_initialized = True
    logger.info(f"Logging initialized (level: {log_level})")
    return logger


def get_logger(name: str = "g4f_bridge") -> logging.Logger:
    """Get logger instance."""
    return logging.getLogger(f"g4f_bridge.{name}" if name != "g4f_bridge" else name)