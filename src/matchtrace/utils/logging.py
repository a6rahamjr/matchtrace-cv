from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if hasattr(record, "event"):
            payload["event"] = record.event
        return json.dumps(payload, sort_keys=True)


class JsonLineHandler(logging.Handler):
    def __init__(self, path: Path):
        super().__init__()
        self.path = path

    def emit(self, record: logging.LogRecord) -> None:
        try:
            line = self.format(record)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
        except Exception:
            self.handleError(record)


def configure_logging(log_path: Path | None = None) -> logging.Logger:
    logger = logging.getLogger("matchtrace")
    logger.setLevel(logging.INFO)
    for handler in logger.handlers:
        handler.close()
    logger.handlers.clear()

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(
        logging.Formatter("%(levelname)s %(name)s: %(message)s")
    )
    logger.addHandler(stream_handler)

    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = JsonLineHandler(log_path)
        file_handler.setFormatter(JsonFormatter())
        logger.addHandler(file_handler)

    return logger
