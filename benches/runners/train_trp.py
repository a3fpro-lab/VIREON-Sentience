"""
TRP trainer on ShiftingBandit.

Same REINFORCE baseline, but updates are TRP-paced:
- KL leash generates divergence pressure
- dt_eff scales the learning rate effectively

Matched-budget: same environment steps, same seeds.
"""

from __future__ import annotations
import numpy as np

from benches.envs.bandits_shift import ShiftingBandit
from vireon.trp import TRPPacer


def softmax(x):
    x = x - np.max(x)
    e = np.exp(x)
    return e / np.sum(e)


def kl_categorical(p_new, p_old, eps=1e-12):
    p_new = np.clip(p_new, eps, 1.0)
    p_old = np.clip(p_old, eps, 1.0)
    return float(np.sum(p_new * (np.log(p_new) - np.log(p_old))))


def train_trp_reinforce_bandit(
    K=5,
    steps=2000,
    lr=0.05,
    seed=0,
    shift_every=200,
    shift_scale=0.15,
    kappa=0.02,
    leash_power=1.0,
):
    rng = np.random.default_rng(seed)
    env = ShiftingBandit(K=K, shift_every=shift_every, shift_scale=shift_scale, seed=seed)
    env.reset()

    theta = np.zeros(K, dtype=np.float64)

    pacer = TRPPacer(dt=1.0, kappa=kappa, leash_power=leash_power)
    alpha_t = pacer.alpha_schedule.alpha0

    rewards = []
    probs_trace = []
    theta_trace = []
    dt_eff_trace = []
    kl_trace = []
    div_trace = []

    baseline = 0.0
    beta = 0.9

    p_old = softmax(theta)

    for t in range(steps):
        p_new = softmax(theta)
        a = int(rng.choice(K, p=p_new))
        _, r, _, info = env.step(a)

        # compute KL between successive policies
        kl_val = kl_categorical(p_new, p_old)

        # TRP pacing
        dt_eff, alpha_next, divergence = pacer.step(kl_val=kl_val, alpha_t=alpha_t)

        # scale learning rate by dt_eff (time dilation)
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
        div_trace.append(divergence)

        # advance
        p_old = p_new
        alpha_t = alpha_next

    return {
        "rewards": np.array(rewards),
        "probs": np.array(probs_trace),
        "theta": np.array(theta_trace),
        "dt_eff": np.array(dt_eff_trace),
        "kl": np.array(kl_trace),
        "divergence": np.array(div_trace),
    }


def main():
    out = train_trp_reinforce_bandit()
    print("TRP mean reward:", out["rewards"].mean())
    print("Mean dt_eff:", out["dt_eff"].mean())


if __name__ == "__main__":
    main()
