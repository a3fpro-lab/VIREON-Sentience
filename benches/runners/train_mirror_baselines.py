"""
Baseline trainer for MirrorGame.

REINFORCE with tabular softmax policy.
State for policy:
    (a_prev, o_prev)

No TRP, no RSM.
"""

from __future__ import annotations
import numpy as np
from collections import defaultdict

from benches.envs.mirror_game import MirrorGame


def softmax(logits):
    logits = logits - np.max(logits)
    e = np.exp(logits)
    return e / np.sum(e)


def train_reinforce_mirror_game(
    episodes=500,
    lr=0.1,
    gamma=0.99,
    seed=0,
    p_mirror=0.9,
    max_steps=50,
):
    rng = np.random.default_rng(seed)
    env = MirrorGame(p_mirror=p_mirror, max_steps=max_steps, seed=seed)

    n_actions = 2
    theta = defaultdict(lambda: np.zeros(n_actions, dtype=np.float64))

    ep_returns = []
    policy_kl_trace = []

    baseline = 0.0
    beta = 0.9

    for ep in range(episodes):
        obs = env.reset()
        done = False

        traj_states, traj_actions, traj_rewards, traj_probs = [], [], [], []

        while not done:
            a_prev, o_prev = int(obs[1]), int(obs[2])
            state = (a_prev, o_prev)

            logits = theta[state]
            probs = softmax(logits)
            a = int(rng.choice(n_actions, p=probs))

            next_obs, r, done, info = env.step(a)

            traj_states.append(state)
            traj_actions.append(a)
            traj_rewards.append(r)
            traj_probs.append(probs)

            obs = next_obs

        # returns
        G = 0.0
        returns = []
        for r in reversed(traj_rewards):
            G = r + gamma * G
            returns.append(G)
        returns.reverse()

        # update
        for state, a, Gt, probs in zip(traj_states, traj_actions, returns, traj_probs):
            adv = Gt - baseline
            grad = -probs
            grad[a] += 1.0
            theta[state] += lr * adv * grad
            baseline = beta * baseline + (1 - beta) * Gt

        ep_returns.append(sum(traj_rewards))

        # approx KL between old probs and new probs on visited states
        kl_ep = 0.0
        for state, probs_old in zip(traj_states, traj_probs):
            probs_new = softmax(theta[state])
            probs_new = np.clip(probs_new, 1e-12, 1.0)
            probs_old = np.clip(probs_old, 1e-12, 1.0)
            kl_ep += float(np.sum(probs_new * (np.log(probs_new) - np.log(probs_old))))
        policy_kl_trace.append(kl_ep / max(1, len(traj_states)))

    return {
        "returns": np.array(ep_returns, dtype=np.float64),
        "policy_kl": np.array(policy_kl_trace, dtype=np.float64),
    }


def main():
    out = train_reinforce_mirror_game()
    print("Baseline mean return:", out["returns"].mean())


if __name__ == "__main__":
    main()
