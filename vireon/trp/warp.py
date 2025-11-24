"""
TRP time dilation core.

Mathematics:
    dt_eff(t) = dt * exp(-alpha_t * D_t)

where:
- dt: base step
- D_t: divergence / structure signal (typically KL surge or entropy drag)
- alpha_t: adaptive perception gain
"""

from __future__ import annotations
import math


def dt_eff_from_divergence(
    dt: float,
    divergence: float,
    alpha_t: float,
    dt_min: float = 1e-18,
    dt_max: float = None,
) -> float:
    """
    Compute effective time step under TRP time dilation.

    Args:
        dt: base step size (positive).
        divergence: nonnegative structure/instability signal D_t.
        alpha_t: perception gain (nonnegative).
        dt_min: hard floor to avoid numerical collapse.
        dt_max: optional ceiling (if None, no ceiling).

    Returns:
        dt_eff clamped to [dt_min, dt_max] if dt_max is set.
    """
    if dt <= 0:
        raise ValueError("dt must be positive.")
    if divergence < 0:
        raise ValueError("divergence must be nonnegative.")
    if alpha_t < 0:
        raise ValueError("alpha_t must be nonnegative.")

    dt_eff = dt * math.exp(-alpha_t * divergence)

    if dt_eff < dt_min:
        dt_eff = dt_min
    if dt_max is not None and dt_eff > dt_max:
        dt_eff = dt_max

    return dt_eff
