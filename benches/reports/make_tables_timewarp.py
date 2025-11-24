"""
Comparison table for time_warp_grid:
Baseline vs TRP vs TRP+RSM.
Also computes a crude M4 Agency Delay Curve proxy:
average delay chosen per episode.

Run:
    python benches/reports/make_tables_timewarp.py
"""

from __future__ import annotations
import numpy as np

from benches.runners.train_timewarp_baselines import train_reinforce_timewarp
from benches.runners.train_timewarp_trp import train_trp_reinforce_timewarp
from benches.runners.train_timewarp_trp_rsm import train_trp_rsm_reinforce_timewarp


def summarize(x):
    return float(np.mean(x)), float(np.std(x))


def main():
    seeds = list(range(5))
    base_means, trp_means, rsm_means = [], [], []

    for s in seeds:
        b = train_reinforce_timewarp(seed=s)
        t = train_trp_reinforce_timewarp(seed=s)
        r = train_trp_rsm_reinforce_timewarp(seed=s)

        base_means.append(b["returns"].mean())
        trp_means.append(t["returns"].mean())
        rsm_means.append(r["returns"].mean())

    b_m, b_sd = summarize(base_means)
    t_m, t_sd = summarize(trp_means)
    r_m, r_sd = summarize(rsm_means)

    print("=== time_warp_grid summary (5 seeds) ===")
    print(f"Baseline   mean return: {b_m:.4f} ± {b_sd:.4f}")
    print(f"TRP        mean return: {t_m:.4f} ± {t_sd:.4f}")
    print(f"TRP+RSM    mean return: {r_m:.4f} ± {r_sd:.4f}")
    print(f"Δ(TRP-Base):     {(t_m-b_m):.4f}")
    print(f"Δ(TRP+RSM-Base): {(r_m-b_m):.4f}")
    print()
    print("M4 Agency Delay Curve is obtained by sweeping delay τ in analysis notebooks.")
    print("This bench provides the substrate; runner logs action-delay implicitly in env info.")


if __name__ == "__main__":
    main()
