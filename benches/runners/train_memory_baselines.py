"""
Baseline trainer for MemoryMaze.

We use REINFORCE with a tabular softmax policy.
To allow memory, the runner stores the hint seen at t=0
and appends it to the policy state.

State used by agent:
    (x, y, hint_mem)

This is prereg-allowed: memory is agent-side, not env leak.
"""

from __future__ import annotations
import numpy as np
from collections import defaultdict

from benches.envs.memory_maze import MemoryMaze


def softmax(logits):
    logits = logits - np.max(logits)
    e = np.exp(logits)
    return e / np.sum(e)


def train_reinforce_memory_maze(
    size=5,
    episodes=500,
    lr=0.1,
    gamma=0.99,
    seed=0,
    max_steps=50,
):
    rng = np.random.default_rng(seed)
    env = MemoryMaze(size=size, max_steps=max_steps, seed=seed)

    n_actions = len(env.ACTIONS)

    # tabular logits: dict[(x,y,hint_mem)] -> logits
    theta = defaultdict(lambda: np.zeros(n_actions, dtype=np.float64))

    ep_returns = []
    policy_kl_trace = []

    baseline = 0.0
    beta = 0.9

    for ep in range(episodes):
        obs = env.reset()
        hint_mem = int(obs[3])  # hint visible at t=0
        done = False

        traj_states, traj_actions, traj_rewards, traj_probs = [], [], [], []

        while not done:
            x, y = int(obs[0]), int(obs[1])
            state = (x, y, hint_mem)

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

        # policy gradient update
        for state, a, Gt, probs in zip(traj_states, traj_actions, returns, traj_probs):
            adv = Gt - baseline
            grad = -probs
            grad[a] += 1.0
            theta[state] += lr * adv * grad

            baseline = beta * baseline + (1 - beta) * Gt

        ep_returns.append(sum(traj_rewards))

        # KL between last 2 policies on same visited states (approx)
        if ep > 0:
            kl_ep = 0.0
            for state, probs in zip(traj_states, traj_probs):
                probs_new = softmax(theta[state])
                probs_old = probs
                probs_new = np.clip(probs_new, 1e-12, 1.0)
                probs_old = np.clip(probs_old, 1e-12, 1.0)
                kl_ep += float(np.sum(probs_new * (np.log(probs_new) - np.log(probs_old))))
            policy_kl_trace.append(kl_ep / max(1, len(traj_states)))
        else:
            policy_kl_trace.append(0.0)

    return {
        "returns": np.array(ep_returns, dtype=np.float64),
        "policy_kl": np.array(policy_kl_trace, dtype=np.float64),
    }


def main():
    out = train_reinforce_memory_maze()
    print("Baseline mean return:", out["returns"].mean())


if __name__ == "__main__":
    main()
