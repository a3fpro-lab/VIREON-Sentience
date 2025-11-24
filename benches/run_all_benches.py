"""
One-button VIREON clinical runner.

Runs:
- bandits_shift
- memory_maze
- mirror_game
- time_warp_grid

Variants (matched budgets):
- baseline
- trp
- trp_rsm

Outputs:
- stdout tables (same as per-bench reports)
- JSONL evidence logs under benches/logs/

Usage:
    python benches/run_all_benches.py

Notes:
- This script is dependency-light (numpy only).
- If running in CI, keep seeds small for speed.
"""

from __future__ import annotations
import os
from pathlib import Path
import numpy as np

from vireon.metrics import JSONLLogger, RunContext, TRPMetrics, RSMMetrics

# --- import runners ---
from benches.runners.train_baselines import train_reinforce_bandit
from benches.runners.train_trp import train_trp_reinforce_bandit
from benches.runners.train_trp_rsm import train_trp_rsm_reinforce_bandit

from benches.runners.train_memory_baselines import train_reinforce_memory_maze
from benches.runners.train_memory_trp import train_trp_reinforce_memory_maze
from benches.runners.train_memory_trp_rsm import train_trp_rsm_reinforce_memory_maze

from benches.runners.train_mirror_baselines import train_reinforce_mirror_game
from benches.runners.train_mirror_trp import train_trp_reinforce_mirror_game
from benches.runners.train_mirror_trp_rsm import train_trp_rsm_reinforce_mirror_game

from benches.runners.train_timewarp_baselines import train_reinforce_timewarp
from benches.runners.train_timewarp_trp import train_trp_reinforce_timewarp
from benches.runners.train_timewarp_trp_rsm import train_trp_rsm_reinforce_timewarp


LOG_DIR = Path("benches/logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)


def summarize(x):
    return float(np.mean(x)), float(np.std(x))


def log_bandits(seed: int, steps: int = 2000):
    bench = "bandits_shift"

    # paths
    base_path = LOG_DIR / f"{bench}_baseline_seed{seed}.jsonl"
    trp_path = LOG_DIR / f"{bench}_trp_seed{seed}.jsonl"
    rsm_path = LOG_DIR / f"{bench}_trp_rsm_seed{seed}.jsonl"

    # --- baseline ---
    rc = RunContext(bench=bench, variant="baseline", seed=seed, steps_or_episodes=steps)
    out_b = train_reinforce_bandit(seed=seed, steps=steps)
    with JSONLLogger(str(base_path)) as L:
        L.log({"type": "run_context", **rc.to_dict()})
        L.log({"type": "summary", "mean_reward": float(out_b["rewards"].mean())})

    # --- TRP ---
    rc = RunContext(bench=bench, variant="trp", seed=seed, steps_or_episodes=steps)
    out_t = train_trp_reinforce_bandit(seed=seed, steps=steps)
    trp_m = TRPMetrics()
    with JSONLLogger(str(trp_path)) as L:
        L.log({"type": "run_context", **rc.to_dict()})
        for i in range(steps):
            trp_m.update_step(
                dt_eff=float(out_t["dt_eff"][i]),
                kl_policy=float(out_t["kl"][i]),
                divergence=float(out_t["divergence"][i]),
                alpha_t=np.nan,  # alpha not traced in runner (ok)
            )
            L.log({
                "type": "step",
                "i": i,
                "reward": float(out_t["rewards"][i]),
                "dt_eff": float(out_t["dt_eff"][i]),
                "kl_policy": float(out_t["kl"][i]),
                "divergence": float(out_t["divergence"][i]),
            })
        L.log({"type": "trp_summary", **trp_m.summary()})
        L.log({"type": "summary", "mean_reward": float(out_t["rewards"].mean())})

    # --- TRP + RSM ---
    rc = RunContext(bench=bench, variant="trp_rsm", seed=seed, steps_or_episodes=steps)
    out_r = train_trp_rsm_reinforce_bandit(seed=seed, steps=steps)
    trp_m = TRPMetrics()
    rsm_m = RSMMetrics()
    with JSONLLogger(str(rsm_path)) as L:
        L.log({"type": "run_context", **rc.to_dict()})
        for i in range(steps):
            trp_m.update_step(
                dt_eff=float(out_r["dt_eff"][i]),
                kl_policy=float(out_r["kl_policy"][i]),
                divergence=float(out_r["pressure"][i]),  # pressure acts as divergence amplification
                alpha_t=np.nan,
            )
            rsm_m.update_step(
                kl_mirror=float(out_r["kl_mirror"][i]),
                g_self=float(out_r["g_self"][i]),
                pressure=float(out_r["pressure"][i]),
            )
            L.log({
                "type": "step",
                "i": i,
                "reward": float(out_r["rewards"][i]),
                "dt_eff": float(out_r["dt_eff"][i]),
                "kl_policy": float(out_r["kl_policy"][i]),
                "kl_mirror": float(out_r["kl_mirror"][i]),
                "g_self": float(out_r["g_self"][i]),
                "pressure": float(out_r["pressure"][i]),
            })
        L.log({"type": "trp_summary", **trp_m.summary()})
        L.log({"type": "rsm_summary", **rsm_m.summary()})
        L.log({"type": "summary", "mean_reward": float(out_r["rewards"].mean())})

    return out_b["rewards"].mean(), out_t["rewards"].mean(), out_r["rewards"].mean()


def log_memory_maze(seed: int, episodes: int = 500):
    bench = "memory_maze"

    base_path = LOG_DIR / f"{bench}_baseline_seed{seed}.jsonl"
    trp_path = LOG_DIR / f"{bench}_trp_seed{seed}.jsonl"
    rsm_path = LOG_DIR / f"{bench}_trp_rsm_seed{seed}.jsonl"

    rc = RunContext(bench=bench, variant="baseline", seed=seed, steps_or_episodes=episodes)
    out_b = train_reinforce_memory_maze(seed=seed, episodes=episodes)
    with JSONLLogger(str(base_path)) as L:
        L.log({"type": "run_context", **rc.to_dict()})
        for i in range(episodes):
            L.log({"type": "episode", "i": i, "ret": float(out_b["returns"][i]),
                   "kl_policy": float(out_b["policy_kl"][i])})
        L.log({"type": "summary", "mean_return": float(out_b["returns"].mean())})

    rc = RunContext(bench=bench, variant="trp", seed=seed, steps_or_episodes=episodes)
    out_t = train_trp_reinforce_memory_maze(seed=seed, episodes=episodes)
    trp_m = TRPMetrics()
    with JSONLLogger(str(trp_path)) as L:
        L.log({"type": "run_context", **rc.to_dict()})
        for i in range(episodes):
            trp_m.update_step(
                dt_eff=float(out_t["dt_eff"][i]),
                kl_policy=float(out_t["policy_kl"][i]),
                divergence=float(out_t["divergence"][i]),
                alpha_t=np.nan,
            )
            L.log({"type": "episode", "i": i,
                   "ret": float(out_t["returns"][i]),
                   "dt_eff": float(out_t["dt_eff"][i]),
                   "kl_policy": float(out_t["policy_kl"][i]),
                   "divergence": float(out_t["divergence"][i])})
        L.log({"type": "trp_summary", **trp_m.summary()})
        L.log({"type": "summary", "mean_return": float(out_t["returns"].mean())})

    rc = RunContext(bench=bench, variant="trp_rsm", seed=seed, steps_or_episodes=episodes)
    out_r = train_trp_rsm_reinforce_memory_maze(seed=seed, episodes=episodes)
    trp_m = TRPMetrics()
    rsm_m = RSMMetrics()
    with JSONLLogger(str(rsm_path)) as L:
        L.log({"type": "run_context", **rc.to_dict()})
        for i in range(episodes):
            trp_m.update_step(
                dt_eff=float(out_r["dt_eff"][i]),
                kl_policy=float(out_r["policy_kl"][i]),
                divergence=float(out_r["pressure"][i]),
                alpha_t=np.nan,
            )
            rsm_m.update_step(
                kl_mirror=float(out_r["kl_mirror"][i]),
                g_self=float(out_r["g_self"][i]),
                pressure=float(out_r["pressure"][i]),
            )
            L.log({"type": "episode", "i": i,
                   "ret": float(out_r["returns"][i]),
                   "dt_eff": float(out_r["dt_eff"][i]),
                   "kl_policy": float(out_r["policy_kl"][i]),
                   "kl_mirror": float(out_r["kl_mirror"][i]),
                   "g_self": float(out_r["g_self"][i]),
                   "pressure": float(out_r["pressure"][i])})
        L.log({"type": "trp_summary", **trp_m.summary()})
        L.log({"type": "rsm_summary", **rsm_m.summary()})
        L.log({"type": "summary", "mean_return": float(out_r["returns"].mean())})

    return out_b["returns"].mean(), out_t["returns"].mean(), out_r["returns"].mean()


def log_mirror_game(seed: int, episodes: int = 500):
    bench = "mirror_game"

    base_path = LOG_DIR / f"{bench}_baseline_seed{seed}.jsonl"
    trp_path = LOG_DIR / f"{bench}_trp_seed{seed}.jsonl"
    rsm_path = LOG_DIR / f"{bench}_trp_rsm_seed{seed}.jsonl"

    rc = RunContext(bench=bench, variant="baseline", seed=seed, steps_or_episodes=episodes)
    out_b = train_reinforce_mirror_game(seed=seed, episodes=episodes)
    with JSONLLogger(str(base_path)) as L:
        L.log({"type": "run_context", **rc.to_dict()})
        for i in range(episodes):
            L.log({"type": "episode", "i": i, "ret": float(out_b["returns"][i]),
                   "kl_policy": float(out_b["policy_kl"][i])})
        L.log({"type": "summary", "mean_return": float(out_b["returns"].mean())})

    rc = RunContext(bench=bench, variant="trp", seed=seed, steps_or_episodes=episodes)
    out_t = train_trp_reinforce_mirror_game(seed=seed, episodes=episodes)
    trp_m = TRPMetrics()
    with JSONLLogger(str(trp_path)) as L:
        L.log({"type": "run_context", **rc.to_dict()})
        for i in range(episodes):
            trp_m.update_step(
                dt_eff=float(out_t["dt_eff"][i]),
                kl_policy=float(out_t["policy_kl"][i]),
                divergence=float(out_t["divergence"][i]),
                alpha_t=np.nan,
            )
            L.log({"type": "episode", "i": i,
                   "ret": float(out_t["returns"][i]),
                   "dt_eff": float(out_t["dt_eff"][i]),
                   "kl_policy": float(out_t["policy_kl"][i]),
                   "divergence": float(out_t["divergence"][i])})
        L.log({"type": "trp_summary", **trp_m.summary()})
        L.log({"type": "summary", "mean_return": float(out_t["returns"].mean())})

    rc = RunContext(bench=bench, variant="trp_rsm", seed=seed, steps_or_episodes=episodes)
    out_r = train_trp_rsm_reinforce_mirror_game(seed=seed, episodes=episodes)
    trp_m = TRPMetrics()
    rsm_m = RSMMetrics()
    with JSONLLogger(str(rsm_path)) as L:
        L.log({"type": "run_context", **rc.to_dict()})
        for i in range(episodes):
            trp_m.update_step(
                dt_eff=float(out_r["dt_eff"][i]),
                kl_policy=float(out_r["policy_kl"][i]),
                divergence=float(out_r["pressure"][i]),
                alpha_t=np.nan,
            )
            rsm_m.update_step(
                kl_mirror=float(out_r["kl_mirror"][i]),
                g_self=float(out_r["g_self"][i]),
                pressure=float(out_r["pressure"][i]),
            )
            L.log({"type": "episode", "i": i,
                   "ret": float(out_r["returns"][i]),
                   "dt_eff": float(out_r["dt_eff"][i]),
                   "kl_policy": float(out_r["policy_kl"][i]),
                   "kl_mirror": float(out_r["kl_mirror"][i]),
                   "g_self": float(out_r["g_self"][i]),
                   "pressure": float(out_r["pressure"][i])})
        L.log({"type": "trp_summary", **trp_m.summary()})
        L.log({"type": "rsm_summary", **rsm_m.summary()})
        L.log({"type": "summary", "mean_return": float(out_r["returns"].mean())})

    return out_b["returns"].mean(), out_t["returns"].mean(), out_r["returns"].mean()


def log_timewarp(seed: int, episodes: int = 600):
    bench = "time_warp_grid"

    base_path = LOG_DIR / f"{bench}_baseline_seed{seed}.jsonl"
    trp_path = LOG_DIR / f"{bench}_trp_seed{seed}.jsonl"
    rsm_path = LOG_DIR / f"{bench}_trp_rsm_seed{seed}.jsonl"

    rc = RunContext(bench=bench, variant="baseline", seed=seed, steps_or_episodes=episodes)
    out_b = train_reinforce_timewarp(seed=seed, episodes=episodes)
    with JSONLLogger(str(base_path)) as L:
        L.log({"type": "run_context", **rc.to_dict()})
        for i in range(episodes):
            L.log({"type": "episode", "i": i, "ret": float(out_b["returns"][i]),
                   "kl_policy": float(out_b["policy_kl"][i])})
        L.log({"type": "summary", "mean_return": float(out_b["returns"].mean())})

    rc = RunContext(bench=bench, variant="trp", seed=seed, steps_or_episodes=episodes)
    out_t = train_trp_reinforce_timewarp(seed=seed, episodes=episodes)
    trp_m = TRPMetrics()
    with JSONLLogger(str(trp_path)) as L:
        L.log({"type": "run_context", **rc.to_dict()})
        for i in range(episodes):
            trp_m.update_step(
                dt_eff=float(out_t["dt_eff"][i]),
                kl_policy=float(out_t["policy_kl"][i]),
                divergence=float(out_t["policy_kl"][i]),
                alpha_t=np.nan,
            )
            L.log({"type": "episode", "i": i,
                   "ret": float(out_t["returns"][i]),
                   "dt_eff": float(out_t["dt_eff"][i]),
                   "kl_policy": float(out_t["policy_kl"][i])})
        L.log({"type": "trp_summary", **trp_m.summary()})
        L.log({"type": "summary", "mean_return": float(out_t["returns"].mean())})

    rc = RunContext(bench=bench, variant="trp_rsm", seed=seed, steps_or_episodes=episodes)
    out_r = train_trp_rsm_reinforce_timewarp(seed=seed, episodes=episodes)
    trp_m = TRPMetrics()
    rsm_m = RSMMetrics()
    with JSONLLogger(str(rsm_path)) as L:
        L.log({"type": "run_context", **rc.to_dict()})
        for i in range(episodes):
            trp_m.update_step(
                dt_eff=float(out_r["dt_eff"][i]),
                kl_policy=float(out_r["policy_kl"][i]),
                divergence=float(out_r["pressure"][i]),
                alpha_t=np.nan,
            )
            rsm_m.update_step(
                kl_mirror=float(out_r["kl_mirror"][i]),
                g_self=float(out_r["g_self"][i]),
                pressure=float(out_r["pressure"][i]),
            )
            L.log({"type": "episode", "i": i,
                   "ret": float(out_r["returns"][i]),
                   "dt_eff": float(out_r["dt_eff"][i]),
                   "kl_policy": float(out_r["policy_kl"][i]),
                   "kl_mirror": float(out_r["kl_mirror"][i]),
                   "g_self": float(out_r["g_self"][i]),
                   "pressure": float(out_r["pressure"][i])})
        L.log({"type": "trp_summary", **trp_m.summary()})
        L.log({"type": "rsm_summary", **rsm_m.summary()})
        L.log({"type": "summary", "mean_return": float(out_r["returns"].mean())})

    return out_b["returns"].mean(), out_t["returns"].mean(), out_r["returns"].mean()


def main():
    seeds = [0, 1, 2, 3, 4]

    # --- bandits_shift ---
    b, t, r = [], [], []
    for s in seeds:
        bb, tt, rr = log_bandits(s)
        b.append(bb); t.append(tt); r.append(rr)
    b_m, b_sd = summarize(b); t_m, t_sd = summarize(t); r_m, r_sd = summarize(r)
    print("\n=== bandits_shift (5 seeds) ===")
    print(f"Baseline  mean reward: {b_m:.4f} ± {b_sd:.4f}")
    print(f"TRP       mean reward: {t_m:.4f} ± {t_sd:.4f}")
    print(f"TRP+RSM   mean reward: {r_m:.4f} ± {r_sd:.4f}")

    # --- memory_maze ---
    b, t, r = [], [], []
    for s in seeds:
        bb, tt, rr = log_memory_maze(s)
        b.append(bb); t.append(tt); r.append(rr)
    b_m, b_sd = summarize(b); t_m, t_sd = summarize(t); r_m, r_sd = summarize(r)
    print("\n=== memory_maze (5 seeds) ===")
    print(f"Baseline  mean return: {b_m:.4f} ± {b_sd:.4f}")
    print(f"TRP       mean return: {t_m:.4f} ± {t_sd:.4f}")
    print(f"TRP+RSM   mean return: {r_m:.4f} ± {r_sd:.4f}")

    # --- mirror_game ---
    b, t, r = [], [], []
    for s in seeds:
        bb, tt, rr = log_mirror_game(s)
        b.append(bb); t.append(tt); r.append(rr)
    b_m, b_sd = summarize(b); t_m, t_sd = summarize(t); r_m, r_sd = summarize(r)
    print("\n=== mirror_game (5 seeds) ===")
    print(f"Baseline  mean return: {b_m:.4f} ± {b_sd:.4f}")
    print(f"TRP       mean return: {t_m:.4f} ± {t_sd:.4f}")
    print(f"TRP+RSM   mean return: {r_m:.4f} ± {r_sd:.4f}")

    # --- time_warp_grid ---
    b, t, r = [], [], []
    for s in seeds:
        bb, tt, rr = log_timewarp(s)
        b.append(bb); t.append(tt); r.append(rr)
    b_m, b_sd = summarize(b); t_m, t_sd = summarize(t); r_m, r_sd = summarize(r)
    print("\n=== time_warp_grid (5 seeds) ===")
    print(f"Baseline  mean return: {b_m:.4f} ± {b_sd:.4f}")
    print(f"TRP       mean return: {t_m:.4f} ± {t_sd:.4f}")
    print(f"TRP+RSM   mean return: {r_m:.4f} ± {r_sd:.4f}")

    print("\nLogs written to benches/logs/*.jsonl")


if __name__ == "__main__":
    main()
