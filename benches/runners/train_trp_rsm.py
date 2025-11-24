"""
TRP + RSM trainer on ShiftingBandit.

Baseline:
- REINFORCE softmax bandit agent

Additions:
- TRP pacing from KL(pi_t+1 || pi_t)
- MirrorModel learning π_hat(a|s)
- SelfForecaster on prediction error of rewards
- CollapseGuard turns self-model signals into extra pressure

Matched-budget: same env steps, same seeds.
"""

from __future__ import annotations
import numpy as np

from benches.envs.bandits_shift import ShiftingBandit
from vireon.trp import TRPPacer
from vireon.rsm import MirrorModel, SelfForecaster, CollapseGuard


def softmax(x):
    x = x - np.max(x)
    e = np.exp(x)
    return e / np.sum(e)


def kl_categorical(p_new, p_old, eps=1e-12):
    p_new = np.clip(p_new, eps, 1.0)
    p_old = np.clip(p_old, eps, 1.0)
    return float(np.sum(p_new * (np.log(p_new) - np.log(p_old))))


def train_trp_rsm_reinforce_bandit(
    K=5,
    steps=2000,
    lr=0.05,
    seed=0,
    shift_every=200,
    shift_scale=0.15,
    kappa=0.02,
    leash_power=1.0,
    # CollapseGuard prereg-weights (can be locked later in configs)
    w_policy=1.0,
    w_mirror=1.0,
    w_self=1.0,
):
    rng = np.random.default_rng(seed)
    env = ShiftingBandit(K=K, shift_every=shift_every, shift_scale=shift_scale, seed=seed)
    env.reset()

    theta = np.zeros(K, dtype=np.float64)

    pacer = TRPPacer(dt=1.0, kappa=kappa, leash_power=leash_power)
    alpha_t = pacer.alpha_schedule.alpha0

    mirror = MirrorModel(n_actions=K, n_buckets=1, ema_beta=0.98)
    forecaster = SelfForecaster(beta=0.98, window=50)
    guard = CollapseGuard(w_policy=w_policy, w_mirror=w_mirror, w_self=w_self)

    rewards = []
    probs_trace = []
    theta_trace = []
    dt_eff_trace = []
    kl_trace = []
    mirror_kl_trace = []
    gself_trace = []
    pressure_trace = []

    baseline = 0.0
    beta = 0.9

    p_old = softmax(theta)

    for t in range(steps):
        obs = np.array([t], dtype=np.float32)

        p_new = softmax(theta)
        a = int(rng.choice(K, p=p_new))
        _, r, _, info = env.step(a)

        # === policy KL ===
        kl_val = kl_categorical(p_new, p_old)

        # === mirror update + KL(pi || old mirror) ===
        kl_mirror = mirror.update(obs, p_new)

        # === self-forecast error ===
        # simple scalar "error": reward surprise vs EMA baseline
        e_t = r - baseline
        forecaster.update(e_t)
        g_self = forecaster.self_surprise_gap()

        # === collapse pressure (>=1) ===
        pressure = guard.pressure(kl_policy=kl_val, kl_mirror=kl_mirror, g_self=g_self)

        # === TRP pacing with pressure amplified divergence ===
        # divergence from KL leash, scaled by pressure
        dt_eff, alpha_next, divergence = pacer.step(kl_val=kl_val * pressure, alpha_t=alpha_t)

        lr_eff = lr * dt_eff

        # REINFORCE update
        grad = -p_new
        grad[a] += 1.0
        advantage = r - baseline
        theta += lr_eff * advantage * grad

        baseline = beta * baseline + (1 - beta) * r

        # traces
        rewards.append(r)
        probs_trace.append(p_new.copy())
        theta_trace.append(theta.copy())
        dt_eff_trace.append(dt_eff)
        kl_trace.append(kl_val)
        mirror_kl_trace.append(kl_mirror)
        gself_trace.append(g_self)
        pressure_trace.append(pressure)

        # advance
        p_old = p_new
        alpha_t = alpha_next

    return {
        "rewards": np.array(rewards),
        "probs": np.array(probs_trace),
        "theta": np.array(theta_trace),
        "dt_eff": np.array(dt_eff_trace),
        "kl_policy": np.array(kl_trace),
        "kl_mirror": np.array(mirror_kl_trace),
        "g_self": np.array(gself_trace),
        "pressure": np.array(pressure_trace),
    }


def main():
    out = train_trp_rsm_reinforce_bandit()
    print("TRP+RSM mean reward:", out["rewards"].mean())
    print("Mean KL_mirror:", out["kl_mirror"].mean())
    print("Mean G_self:", out["g_self"].mean())


if __name__ == "__main__":
    main()
