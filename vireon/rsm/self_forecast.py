"""
SelfForecaster

Forecast upcoming error/loss spikes using a simple online Gaussian model:
- EMA mean
- EMA variance

We then compute a "self-surprise gap" by comparing predicted vs realized
error distributions over a rolling window.

This is lightweight and prereg-safe.
"""

from __future__ import annotations
import numpy as np


def kl_gaussian_1d(mu_p, var_p, mu_q, var_q, eps=1e-12):
    """
    KL( N_p || N_q ) for 1D Gaussians.
    """
    var_p = max(var_p, eps)
    var_q = max(var_q, eps)
    return float(
        0.5 * (np.log(var_q / var_p) + (var_p + (mu_p - mu_q) ** 2) / var_q - 1.0)
    )


class SelfForecaster:
    """
    Online Gaussian forecaster of scalar errors e_t.

    Args:
        beta: EMA smoothing (0<beta<1)
        window: size for realized distribution estimate
        eps: numerical floor
    """

    def __init__(self, beta: float = 0.98, window: int = 50, eps: float = 1e-12):
        if not (0.0 < beta < 1.0):
            raise ValueError("beta must be in (0,1).")
        if window < 5:
            raise ValueError("window must be >= 5.")

        self.beta = beta
        self.window = window
        self.eps = eps

        self.mu = 0.0
        self.var = 1.0
        self.hist = []

    def update(self, e_t: float):
        """
        Update EMA mean/var with new error e_t.

        Returns:
            (mu_pred_next, var_pred_next)
        """
        beta = self.beta
        e_t = float(e_t)

        # update mean
        mu_new = beta * self.mu + (1 - beta) * e_t

        # update variance (EMA of squared residual)
        resid = e_t - mu_new
        var_new = beta * self.var + (1 - beta) * (resid ** 2)

        self.mu, self.var = mu_new, max(var_new, self.eps)

        self.hist.append(e_t)
        if len(self.hist) > self.window:
            self.hist.pop(0)

        return self.mu, self.var

    def predicted_distribution(self):
        """
        Return predicted (mu, var) for next-step error.
        """
        return self.mu, self.var

    def realized_distribution(self):
        """
        Estimate realized (mu, var) from rolling window history.
        """
        if len(self.hist) < 2:
            return self.mu, self.var
        x = np.array(self.hist, dtype=np.float64)
        return float(x.mean()), float(max(x.var(ddof=1), self.eps))

    def self_surprise_gap(self):
        """
        G_self = KL(predicted || realized)
        """
        mu_p, var_p = self.predicted_distribution()
        mu_q, var_q = self.realized_distribution()
        return kl_gaussian_1d(mu_p, var_p, mu_q, var_q, eps=self.eps)
