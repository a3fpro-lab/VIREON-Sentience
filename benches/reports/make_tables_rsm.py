"""
Comparison table:
Baseline vs TRP vs TRP+RSM on bandits_shift.

Also prints Gate-B signals:
- KL_mirror (should trend down / stabilize)
- G_self self-surprise gap (should be low)

Run:
    python benches/reports/make_tables_rsm.py
"""

from __future__ import annotations
import numpy as np

from benches.runners.train_baselines import train_reinforce_bandit
from benches.runners.train_trp import train_trp_reinforce_bandit
from benches.runners.train_trp_rsm import train_trp_rsm_reinforce_bandit


def summarize(x):
    return float(np.mean(x)), float(np.std(x))


def main():
    seeds = list(range(5))

    base_means, trp_means, rsm_means = [], [], []
    mirror_means, gself_means, pressure_means = [], [], []

    for s in seeds:
        b = train_reinforce_bandit(seed=s)
        t = train_trp_reinforce_bandit(seed=s)
        r = train_trp_rsm_reinforce_bandit(seed=s)

        base_means.append(b["rewards"].mean())
        trp_means.append(t["rewards"].mean())
        rsm_means.append(r["rewards"].mean())

        mirror_means.append(r["kl_mirror"].mean())
        gself_means.append(r["g_self"].mean())
        pressure_means.append(r["pressure"].mean())

    b_m, b_sd = summarize(base_means)
    t_m, t_sd = summarize(trp_means)
    r_m, r_sd = summarize(rsm_means)

    mk_m, mk_sd = summarize(mirror_means)
    gs_m, gs_sd = summarize(gself_means)
    pr_m, pr_sd = summarize(pressure_means)

    print("=== bandits_shift summary (5 seeds) ===")
    print(f"Baseline   mean reward: {b_m:.4f} ± {b_sd:.4f}")
    print(f"TRP        mean reward: {t_m:.4f} ± {t_sd:.4f}")
    print(f"TRP+RSM    mean reward: {r_m:.4f} ± {r_sd:.4f}")
    print(f"Δ(TRP-Base):     {(t_m-b_m):.4f}")
    print(f"Δ(TRP+RSM-Base): {(r_m-b_m):.4f}")
    print()
    print("=== Gate-B signals (TRP+RSM only) ===")
    print(f"Mean KL_mirror: {mk_m:.6f} ± {mk_sd:.6f}")
    print(f"Mean G_self:    {gs_m:.6f} ± {gs_sd:.6f}")
    print(f"Mean pressure:  {pr_m:.6f} ± {pr_sd:.6f}")


if __name__ == "__main__":
    main()
