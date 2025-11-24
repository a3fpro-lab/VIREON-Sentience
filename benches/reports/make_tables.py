"""
Make a tiny comparison table for baseline vs TRP on bandits_shift.

Run:
    python benches/reports/make_tables.py
"""

from __future__ import annotations
import numpy as np

from benches.runners.train_baselines import train_reinforce_bandit
from benches.runners.train_trp import train_trp_reinforce_bandit


def summarize(x):
    return float(np.mean(x)), float(np.std(x))


def main():
    seeds = list(range(5))
    base_means = []
    trp_means = []

    for s in seeds:
        b = train_reinforce_bandit(seed=s)
        t = train_trp_reinforce_bandit(seed=s)
        base_means.append(b["rewards"].mean())
        trp_means.append(t["rewards"].mean())

    b_m, b_sd = summarize(base_means)
    t_m, t_sd = summarize(trp_means)

    print("=== bandits_shift summary (5 seeds) ===")
    print(f"Baseline  mean reward: {b_m:.4f} ± {b_sd:.4f}")
    print(f"TRP       mean reward: {t_m:.4f} ± {t_sd:.4f}")
    print(f"Δ(TRP-Base): {(t_m-b_m):.4f}")


if __name__ == "__main__":
    main()
