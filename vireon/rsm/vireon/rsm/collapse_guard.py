"""
CollapseGuard

Combine self-model signals into a single stability pressure.

Inputs:
- kl_policy: KL(pi_{t+1} || pi_t)
- kl_mirror: KL(pi || pi_hat)
- g_self: self-surprise gap

Output:
- pressure >= 1, used to trigger TRP dilation more strongly.
"""

from __future__ import annotations
from dataclasses import dataclass


@dataclass
class CollapseGuard:
    """
    Weighted stability pressure.

    pressure = 1 + w_policy*kl_policy + w_mirror*kl_mirror + w_self*g_self

    All weights preregistered in configs.
    """
    w_policy: float = 1.0
    w_mirror: float = 1.0
    w_self: float = 1.0

    def __post_init__(self):
        if self.w_policy < 0 or self.w_mirror < 0 or self.w_self < 0:
            raise ValueError("weights must be nonnegative.")

    def pressure(self, kl_policy: float, kl_mirror: float, g_self: float) -> float:
        if kl_policy < 0 or kl_mirror < 0 or g_self < 0:
            raise ValueError("inputs must be nonnegative.")
        return 1.0 + self.w_policy * kl_policy + self.w_mirror * kl_mirror + self.w_self * g_self
