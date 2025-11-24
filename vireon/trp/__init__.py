"""
TRP (Time = Reality × Perception) operator modules.

Core pieces:
- warp: dt_eff time dilation
- leash: KL stability corridor
- alpha: adaptive perception gain
- pacing: high-level step integrator
"""
from .warp import dt_eff_from_divergence
from .leash import kl_leash_ok, kl_leash_scale
from .alpha import AlphaSchedule
from .pacing import TRPPacer

__all__ = [
    "dt_eff_from_divergence",
    "kl_leash_ok",
    "kl_leash_scale",
    "AlphaSchedule",
    "TRPPacer",
]
