"""
ShiftingBandit environment.

- K arms.
- Each arm has a Bernoulli reward with probability p_k.
- Every `shift_every` steps, probabilities shift by a small random delta.

This creates a clean, controlled non-stationarity test.
"""

from __future__ import annotations
import numpy as np


class ShiftingBandit:
    def __init__(
        self,
        K: int = 5,
        shift_every: int = 200,
        shift_scale: float = 0.15,
        seed: int = 0,
    ):
        if K < 2:
            raise ValueError("K must be >= 2.")
        if shift_every < 1:
            raise ValueError("shift_every must be >= 1.")
        if not (0.0 < shift_scale <= 1.0):
            raise ValueError("shift_scale must be in (0,1].")

        self.K = K
        self.shift_every = shift_every
        self.shift_scale = shift_scale
        self.rng = np.random.default_rng(seed)

        self.t = 0
        # initialize probabilities away from extremes
        self.p = self.rng.uniform(0.2, 0.8, size=K)

    def reset(self):
        self.t = 0
        self.p = self.rng.uniform(0.2, 0.8, size=self.K)
        return self._obs()

    def step(self, action: int):
        if not (0 <= action < self.K):
            raise ValueError("action out of range.")

        # Bernoulli reward
        r = float(self.rng.random() < self.p[action])

        self.t += 1
        if self.t % self.shift_every == 0:
            self._shift_probs()

        return self._obs(), r, False, {"t": self.t, "p": self.p.copy()}

    def _shift_probs(self):
        # small random drift + clamp
        drift = self.rng.normal(0.0, self.shift_scale, size=self.K)
        self.p = np.clip(self.p + drift, 0.05, 0.95)

    def _obs(self):
        # bandit has no state; return time index only
        return np.array([self.t], dtype=np.float32)
