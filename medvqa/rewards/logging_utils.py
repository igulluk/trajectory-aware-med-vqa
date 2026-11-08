"""Optional per-sample reward logging, enabled with ``DEBUG_MODE=true LOG_PATH=<file>``."""

from __future__ import annotations

import os
from datetime import datetime


def debug_enabled() -> bool:
    return os.getenv("DEBUG_MODE") == "true" and bool(os.getenv("LOG_PATH"))


def debug_log(title: str, **fields) -> None:
    """Append a block of ``key: value`` lines to ``$LOG_PATH`` when debugging is enabled."""
    if not debug_enabled():
        return
    timestamp = datetime.now().strftime("%d-%H-%M-%S-%f")
    with open(os.environ["LOG_PATH"], "a", encoding="utf-8") as f:
        f.write(f"------------- {timestamp} {title} -------------\n")
        for key, value in fields.items():
            f.write(f"{key}: {value}\n")
