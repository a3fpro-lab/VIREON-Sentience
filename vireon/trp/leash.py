"""
KL-Leash stability constraint.

Law:
    KL(pi_{t+1} || pi_t) <= kappa

We provide:
- a boolean check
- a scale factor for time dilation when KL exceeds corridor
"""

from __future__ import annotations


def kl_leash_ok(kl_val: float, kappa: float) -> bool:
    """
    True if kl_val is within stability corridor.

    Args:
        kl_val: measured KL divergence (>=0).
        kappa: corridor limit (>0).

    Returns:
        bool
    """
    if kl_val < 0:
        raise ValueError("kl_val must be nonnegative.")
    if kappa <= 0:
        raise ValueError("kappa must be positive.")
    return kl_val <= kappa


def kl_leash_scale(kl_val: float, kappa: float, power: float = 1.0) -> float:
    """
    If KL exceeds kappa, return a multiplicative scale (>1)
    to increase time dilation pressure (i.e., slow updates more).

    scale = (kl_val / kappa)^power for kl_val>kappa, else 1.

    Args:
        kl_val: KL divergence (>=0).
        kappa: corridor limit (>0).
        power: nonnegative exponent.

    Returns:
        scale factor >= 1
    """
    if kl_val < 0:
        raise ValueError("kl_val must be nonnegative.")
    if kappa <= 0:
        raise ValueError("kappa must be positive.")
    if power < 0:
        raise ValueError("power must be nonnegative.")

    if kl_val <= kappa:
        return 1.0
    return (kl_val / kappa) ** power
