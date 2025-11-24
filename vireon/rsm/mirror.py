"""
MirrorModel

Goal:
Learn a predictive model of the agent’s own policy π(a|s).
We do this in a lightweight, prereg-friendly way:

- Keep an exponential moving average (EMA) of policy probabilities per state bucket.
- Compute mirror KL: KL(π || π_hat)

This is generic and numpy-only.
"""

from __future__ import annotations
import numpy as np


def kl_categorical(p, q, eps=1e-12):
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    return float(np.sum(p * (np.log(p) - np.log(q))))


class MirrorModel:
    """
    Online mirror of policy probabilities.

    Args:
        n_actions: number of actions.
        n_buckets: number of discrete state buckets.
                   (For stateless bandits, keep n_buckets=1.)
        ema_beta: EMA smoothing (0<beta<1). Higher = slower mirror.
        eps: numerical floor.
    """

    def __init__(
        self,
        n_actions: int,
        n_buckets: int = 1,
        ema_beta: float = 0.98,
        eps: float = 1e-12,
    ):
        if n_actions < 2:
            raise ValueError("n_actions must be >= 2.")
        if n_buckets < 1:
            raise ValueError("n_buckets must be >= 1.")
        if not (0.0 < ema_beta < 1.0):
            raise ValueError("ema_beta must be in (0,1).")

        self.n_actions = n_actions
        self.n_buckets = n_buckets
        self.ema_beta = ema_beta
        self.eps = eps

        # mirror probs per bucket
        self.pi_hat = np.ones((n_buckets, n_actions), dtype=np.float64) / n_actions
        self.counts = np.zeros(n_buckets, dtype=np.int64)

    def bucketize(self, obs) -> int:
        """
        Default bucketizer: all obs -> bucket 0.
        Override if you want state-dependent mirroring.
        """
        return 0

    def update(self, obs, pi: np.ndarray) -> float:
        """
        Update mirror toward current policy probabilities pi(a|s).

        Returns:
            mirror KL = KL(pi || pi_hat_bucket).
        """
        b = self.bucketize(obs)
        if not (0 <= b < self.n_buckets):
            raise ValueError("bucket index out of range.")

        pi = np.asarray(pi, dtype=np.float64)
        if pi.shape != (self.n_actions,):
            raise ValueError("pi shape must be (n_actions,)")

        pi = np.clip(pi, self.eps, 1.0)
        pi = pi / np.sum(pi)

        old = self.pi_hat[b].copy()
        beta = self.ema_beta

        self.pi_hat[b] = beta * self.pi_hat[b] + (1 - beta) * pi
        self.pi_hat[b] = np.clip(self.pi_hat[b], self.eps, 1.0)
        self.pi_hat[b] /= np.sum(self.pi_hat[b])

        self.counts[b] += 1

        return kl_categorical(pi, old)

    def predict(self, obs) -> np.ndarray:
        """
        Return mirror prediction π_hat(a|s).
        """
        b = self.bucketize(obs)
        return self.pi_hat[b].copy()

    def mirror_kl(self, obs, pi: np.ndarray) -> float:
        """
        Compute KL(pi || pi_hat) without updating.
        """
        b = self.bucketize(obs)
        pi_hat = self.pi_hat[b]
        return kl_categorical(pi, pi_hat)
