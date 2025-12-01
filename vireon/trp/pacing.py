"""
High-level TRP pacing wrapper.

This class ties together:
- KL leash -> divergence pressure
- dt_eff time dilation
- alpha schedule updates

It doesn't assume a specific RL algorithm; it just outputs dt_eff and alpha_next.
"""

from __future__ import annotations
from dataclasses import dataclass, field

from .warp import dt_eff_from_divergence
from .leash import kl_leash_scale
from .alpha import AlphaSchedule


@dataclass
class TRPPacer:
    """
    TRP pacing integrator.

    Params:
        dt: base update dt
        kappa: KL corridor
        leash_power: exponent for leash_scale
        alpha_schedule: AlphaSchedule instance
        dt_min: floor for dt_eff
    """
    dt: float = 1.0
    kappa: float = 0.01
    leash_power: float = 1.0
    alpha_schedule: AlphaSchedule = field(default_factory=AlphaSchedule)
    dt_min: float = 1e-18

    def __post_init__(self):
        if self.dt <= 0:
            raise ValueError("dt must be positive.")
        if self.kappa <= 0:
            raise ValueError("kappa must be positive.")
        if self.leash_power < 0:
            raise ValueError("leash_power must be nonnegative.")

    def step(self, kl_val: float, alpha_t: float):
        """
        One TRP pacing update.

        Args:
            kl_val: current KL(pi_{t+1}||pi_t)
            alpha_t: current alpha

        Returns:
            (dt_eff, alpha_next, divergence)
        """
        if kl_val < 0:
            raise ValueError("kl_val must be nonnegative.")
        if alpha_t < 0:
            raise ValueError("alpha_t must be nonnegative.")

        # D_t is leash pressure
        divergence = kl_leash_scale(kl_val, self.kappa, self.leash_power)

        dt_eff = dt_eff_from_divergence(self.dt, divergence, alpha_t, dt_min=self.dt_min)
        alpha_next = self.alpha_schedule.update(alpha_t, divergence)

        return dt_eff, alpha_next, divergence
