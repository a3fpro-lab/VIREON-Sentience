"""
RunContext

Single source of truth for run metadata.
Every runner should create one RunContext and pass it into loggers.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Optional


@dataclass
class RunContext:
    bench: str
    variant: str  # "baseline" | "trp" | "trp_rsm"
    seed: int
    steps_or_episodes: int
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "bench": self.bench,
            "variant": self.variant,
            "seed": int(self.seed),
            "steps_or_episodes": int(self.steps_or_episodes),
            "notes": self.notes,
        }
