from __future__ import annotations

import logging

from src.config import LoggingSettings

DEFAULT_LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


def configure_logging(settings: LoggingSettings) -> None:
    """Configure process-wide logging once at startup."""

    logging.basicConfig(
        level=getattr(logging, settings.level.upper(), logging.INFO),
        format=DEFAULT_LOG_FORMAT,
        force=True,
    )
