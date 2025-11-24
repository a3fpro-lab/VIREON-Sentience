"""
Comparison table for MemoryMaze:
Baseline vs TRP vs TRP+RSM.

Run:
    python benches/reports/make_tables_memory.py
"""

from __future__ import annotations
import numpy as np

from benches.runners.train_memory_baselines import train_reinforce_memory_maze
from benches.runners.train_memory_trp import train_trp_reinforce_memory_maze
from benches.runners.train_memory_trp_rsm import train_trp_rsm_reinforce_memory_maze


def summarize(x):
    return float(np.mean(x)), float(np.std(x))


def main():
    seeds = list(range(5))

    base_means, trp_means, rsm_means = [], [], []
    mk_means, gs_means = [], []

    for s in seeds:
        b = train_reinforce_memory_maze(seed=s)
        t = train_trp_reinforce_memory_maze(seed=s)
        r = train_trp_rsm_reinforce_memory_maze(seed=s)

        base_means.append(b["returns"].mean())
        trp_means.append(t["returns"].mean())
        rsm_means.append(r["returns"].mean())

        mk_means.append(r["kl_mirror"].mean())
        gs_means.append(r["g_self"].mean())

    b_m, b_sd = summarize(base_means)
    t_m, t_sd = summarize(trp_means)
    r_m, r_sd = summarize(rsm_means)

    mk_m, mk_sd = summarize(mk_means)
    gs_m, gs_sd = summarize(gs_means)

    print("=== memory_maze summary (5 seeds) ===")
    print(f"Baseline   mean return: {b_m:.4f} ± {b_sd:.4f}")
    print(f"TRP        mean return: {t_m:.4f} ± {t_sd:.4f}")
    print(f"TRP+RSM    mean return: {r_m:.4f} ± {r_sd:.4f}")
    print(f"Δ(TRP-Base):     {(t_m-b_m):.4f}")
    print(f"Δ(TRP+RSM-Base): {(r_m-b_m):.4f}")
    print()
    print("=== Gate-B signals (TRP+RSM) ===")
    print(f"Mean KL_mirror: {mk_m:.6f} ± {mk_sd:.6f}")
    print(f"Mean G_self:    {gs_m:.6f} ± {gs_sd:.6f}")


if __name__ == "__main__":
    main()
