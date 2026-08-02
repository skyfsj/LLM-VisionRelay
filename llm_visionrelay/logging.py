"""Structured, redacting logging helpers.

The middleware must never log credentials, raw request bodies, or image payloads.
This module centralizes log configuration and a redacting formatter as a defense
in depth: even if a log line accidentally contains a ``Bearer`` token or an
``image/png;base64,...`` payload, it is masked before being emitted.
"""

from __future__ import annotations

import logging
import re

LOGGER_NAME = "llm_visionrelay"

_BEARER_RE = re.compile(r"(?i)(bearer\s+)([A-Za-z0-9._~+/\-=]{8,})")
_HEADER_SECRET_RE = re.compile(r"(?i)(authorization|api[_-]?key|token|cookie|secret)(\s*[=:]\s*)(\S+)")
_BASE64_IMAGE_RE = re.compile(r"(?i)(base64,)([A-Za-z0-9+/=]{16,})")


def redact(text: str) -> str:
    """Mask secrets that may appear in a log message."""
    text = _BEARER_RE.sub(r"\1[REDACTED]", text)
    text = _HEADER_SECRET_RE.sub(r"\1\2[REDACTED]", text)
    text = _BASE64_IMAGE_RE.sub(r"\1[REDACTED]", text)
    return text


class RedactingFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        record = logging.LogRecord(
            record.name,
            record.levelno,
            record.pathname,
            record.lineno,
            record.msg,
            record.args,
            record.exc_info,
            record.funcName,
            record.stack_info,
        )
        if isinstance(record.msg, str):
            record.msg = redact(record.msg)
        return super().format(record)


def get_logger(name: str | None = None) -> logging.Logger:
    return logging.getLogger(f"{LOGGER_NAME}.{name}" if name else LOGGER_NAME)


def setup_logging(level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(level.upper())
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            RedactingFormatter(
                fmt="%(asctime)s %(levelname)s %(name)s %(message)s",
                datefmt="%Y-%m-%dT%H:%M:%S%z",
            )
        )
        logger.addHandler(handler)
    logger.propagate = False
    return logger
