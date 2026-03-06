"""Structured logging factory for the Killswitch Engage project.

All modules should obtain their logger via ``get_logger(__name__)`` instead of
calling ``logging.getLogger`` directly. This ensures uniform formatting and
keeps PII out of log records (see SECURITY_CHECKLIST § 3 – Logs sem PII).

Example:
    >>> from src.utils.logger import get_logger
    >>> logger = get_logger(__name__)
    >>> logger.info("Processing request for user %s", user_id)

Security notes:
    - Never log full ``DATABASE_URL``, ``SECRET_KEY`` or ``REDIS_URL``.
    - Never log raw user inputs that may contain PII.
    - Use ``%s`` formatting (lazy), not f-strings, so expensive string
      interpolation is skipped when the log level is not active.
"""

from __future__ import annotations

import logging
import sys

# ── Constants ────────────────────────────────────────────────────────────────

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%dT%H:%M:%S"

_SENSITIVE_SUBSTRINGS = (
    "password",
    "secret",
    "token",
    "key",
    "authorization",
    "database_url",
)

_configured = False  # ensure root handler is set up only once


class _SanitizingFilter(logging.Filter):
    """Drop (or redact) log records that appear to contain raw secrets.

    This is a best-effort guard. Actual secret management must be enforced
    at code level—do not rely on this filter alone.
    """

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: A003
        msg = record.getMessage().lower()
        for substring in _SENSITIVE_SUBSTRINGS:
            if substring in msg and "=" in record.getMessage():
                # Suppress the record; secret may be embedded in value
                return False
        return True


def _configure_root_logger(level: int = logging.INFO) -> None:
    """Set up the root logger with a stream handler if not already done.

    Args:
        level (int): Logging level (default: ``logging.INFO``).
    """
    global _configured  # noqa: PLW0603
    if _configured:
        return

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))
    handler.addFilter(_SanitizingFilter())

    root = logging.getLogger()
    root.setLevel(level)
    root.addHandler(handler)
    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Return a named logger, configuring the root logger on first call.

    Args:
        name (str): Logger name; use ``__name__`` in every module.

    Returns:
        logging.Logger: Configured logger instance.

    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Model loaded in %.2f s", elapsed)
    """
    _configure_root_logger()
    return logging.getLogger(name)
