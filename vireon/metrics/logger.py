"""
JSON Lines logger.

Writes one JSON object per line to a file.
Designed for prereg evidence:
- no mutation after write
- timestamped
- deterministic ordering of keys (sort_keys=True)
"""

from __future__ import annotations
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class JSONLLogger:
    """
    Simple JSONL logger.

    Args:
        path: output file path (creates parents).
        flush_every: flush after N writes (default 1 = always).
    """
    path: str
    flush_every: int = 1

    def __post_init__(self):
        if self.flush_every < 1:
            raise ValueError("flush_every must be >= 1.")
        p = Path(self.path)
        p.parent.mkdir(parents=True, exist_ok=True)
        self._fp = open(p, "a", encoding="utf-8")
        self._count = 0

    def log(self, record: Dict[str, Any]):
        """
        Append a dict as one JSON line with a wall-clock timestamp.
        """
        if not isinstance(record, dict):
            raise ValueError("record must be a dict.")

        record = dict(record)
        record["_ts_unix"] = time.time()

        line = json.dumps(record, sort_keys=True)
        self._fp.write(line + "\n")
        self._count += 1

        if self._count % self.flush_every == 0:
            self._fp.flush()
            os.fsync(self._fp.fileno())

    def close(self):
        try:
            self._fp.flush()
        finally:
            self._fp.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False
