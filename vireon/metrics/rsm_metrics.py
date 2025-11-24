"""
RSMMetrics

Accumulation + summary of RSM signals:
- kl_mirror
- g_self
- pressure

Runners may call update_step(...) each step/episode.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any
import numpy as np


@dataclass
class RSMMetrics:
    kl_mirror: List[float] = field(default_factory=list)
    g_self: List[float] = field(default_factory=list)
    pressure: List[float] = field(default_factory=list)

    def update_step(
        self,
        kl_mirror: float,
        g_self: float,
        pressure: float,
    ):
        self.kl_mirror.append(float(kl_mirror))
        self.g_self.append(float(g_self))
        self.pressure.append(float(pressure))

    def summary(self) -> Dict[str, Any]:
        def s(x):
            if len(x) == 0:
                return {"mean": None, "std": None}
            arr = np.asarray(x, dtype=np.float64)
            return {"mean": float(arr.mean()), "std": float(arr.std(ddof=0))}

        return {
            "kl_mirror": s(self.kl_mirror),
            "g_self": s(self.g_self),
            "pressure": s(self.pressure),
        }
