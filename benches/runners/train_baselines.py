"""
Baseline trainer on ShiftingBandit.

We use a tiny softmax policy-gradient (REINFORCE) so:
- no big deps
- fully transparent math
- easy matched-budget comparison

Outputs a dict of learning curves.
"""

from __future__ import annotations
import numpy as np

from benches.envs.bandits_shift import ShiftingBandit


def softmax(x):
    x = x - np.max(x)
    e = np.exp(x)
    return e / np.sum(e)


def train_reinforce_bandit(
    K=5,
    steps=2000,
    lr=0.05,
    seed=0,
    shift_every=200,
    shift_scale=0.15,
):
    rng = np.random.default_rng(seed)
    env = ShiftingBandit(K=K, shift_every=shift_every, shift_scale=shift_scale, seed=seed)
    env.reset()

    # policy logits (θ)
    theta = np.zeros(K, dtype=np.float64)

    rewards = []
    probs_trace = []
    theta_trace = []

    baseline = 0.0
    beta = 0.9  # baseline EMA

    for t in range(steps):
        p = softmax(theta)
        a = int(rng.choice(K, p=p))
        _, r, _, info = env.step(a)

        # REINFORCE gradient: ∇ log π(a) * (r - baseline)
        grad = -p
        grad[a] += 1.0
        advantage = r - baseline
        theta += lr * advantage * grad

        baseline = beta * baseline + (1 - beta) * r

        rewards.append(r)
        probs_trace.append(p.copy())
        theta_trace.append(theta.copy())

    return {
        "rewards": np.array(rewards),
        "probs": np.array(probs_trace),
        "theta": np.array(theta_trace),
    }


def main():
    out = train_reinforce_bandit()
    print("Baseline mean reward:", out["rewards"].mean())


if __name__ == "__main__":
    main()
