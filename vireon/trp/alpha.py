"""
Adaptive perception gain alpha_t.

We keep alpha scheduling flexible but preregistered.
Default: multiplicative update alpha_{t+1} = alpha_t * g(D_t)

We also include the Pulse Signaler dynamic weight F_alpha(k).
"""

from __future__ import annotations
import math
from dataclasses import dataclass


@dataclass
class AlphaSchedule:
    """
    Simple alpha update rule.

    Params:
        alpha0: initial alpha (>0).
        decay: multiplicative decay per step (<=1).
        gain: divergence gain factor (>=0).
        alpha_min: floor to prevent vanishing perception.
        alpha_max: ceiling to prevent runaway perception.
    """
    alpha0: float = 1.0
    decay: float = 0.999
    gain: float = 1.0
    alpha_min: float = 1e-12
    alpha_max: float = 1e6

    def __post_init__(self):
        if self.alpha0 <= 0:
            raise ValueError("alpha0 must be positive.")
        if not (0 < self.decay <= 1.0):
            raise ValueError("decay must be in (0,1].")
        if self.gain < 0:
            raise ValueError("gain must be nonnegative.")
        if self.alpha_min <= 0:
            raise ValueError("alpha_min must be positive.")
        if self.alpha_max <= self.alpha_min:
            raise ValueError("alpha_max must exceed alpha_min.")

    def update(self, alpha_t: float, divergence: float) -> float:
        """
        Update alpha using preregistered g(D_t).

        Default g(D_t) = exp(gain * divergence) with decay applied.

        Args:
            alpha_t: current alpha.
            divergence: D_t (>=0).

        Returns:
            alpha_{t+1}
        """
        if alpha_t < 0:
            raise ValueError("alpha_t must be nonnegative.")
        if divergence < 0:
            raise ValueError("divergence must be nonnegative.")

        # preregistered g(D_t)
        g = math.exp(self.gain * divergence)

        alpha_next = alpha_t * self.decay * g

        if alpha_next < self.alpha_min:
            alpha_next = self.alpha_min
        if alpha_next > self.alpha_max:
            alpha_next = self.alpha_max

        return alpha_next


def F_alpha_k(k: int, mean_spacing: float, T: float) -> float:
    """
    Pulse Signaler dynamic weight:
        F_alpha(k) ≈ e^{-α}
        α = (k / mean_spacing) * 2π / log T

    Args:
        k: harmonic index (>=1).
        mean_spacing: average spacing (>0).
        T: height/scale parameter (>1).

    Returns:
        F_alpha(k)
    """
    if k < 1:
        raise ValueError("k must be >= 1.")
    if mean_spacing <= 0:
        raise ValueError("mean_spacing must be positive.")
    if T <= 1:
        raise ValueError("T must be > 1.")

    alpha = (k / mean_spacing) * (2.0 * math.pi / math.log(T))
    return math.exp(-alpha)


def C_k_prime_weight(k: int) -> float:
    """
    Prime-weighted correction:
        C_k ≈ Π_{p|k} (p-1)/(p-2) * Π_{p∤k} (p-1)/p

    We implement only the finite p|k part and treat the p∤k
    product as a global constant absorbed elsewhere.

    Args:
        k: integer >= 1

    Returns:
        finite prime correction >= 1
    """
    if k < 1:
        raise ValueError("k must be >= 1.")

    # trial-division factorization
    n = k
    primes = []
    p = 2
    while p * p <= n:
        if n % p == 0:
            primes.append(p)
            while n % p == 0:
                n //= p
        p += 1 if p == 2 else 2
    if n > 1:
        primes.append(n)

    corr = 1.0
    for p in primes:
        if p == 2:
            # avoid division by zero at p=2; treat as neutral
            continue
        corr *= (p - 1) / (p - 2)
    return corr
