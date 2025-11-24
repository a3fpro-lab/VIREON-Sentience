"""
TRPMetrics

Accumulation + summary of TRP signals:
- dt_eff
- kl_policy
- divergence (leash pressure)
- alpha_t

Runners may call update_step(...) each step/episode.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import numpy as np


@dataclass
class TRPMetrics:
    dt_eff: List[float] = field(default_factory=list)
    kl_policy: List[float] = field(default_factory=list)
    divergence: List[float] = field(default_factory=list)
    alpha: List[float] = field(default_factory=list)

    def update_step(
        self,
        dt_eff: float,
        kl_policy: float,
        divergence: float,
        alpha_t: float,
    ):
        self.dt_eff.append(float(dt_eff))
        self.kl_policy.append(float(kl_policy))
        self.divergence.append(float(divergence))
        self.alpha.append(float(alpha_t))

    def summary(self) -> Dict[str, Any]:
        def s(x):
            if len(x) == 0:
                return {"mean": None, "std": None}
            arr = np.asarray(x, dtype=np.float64)
            return {"mean": float(arr.mean()), "std": float(arr.std(ddof=0))}

        return {
            "dt_eff": s(self.dt_eff),
            "kl_policy": s(self.kl_policy),
            "divergence": s(self.divergence),
            "alpha": s(self.alpha),
        }
