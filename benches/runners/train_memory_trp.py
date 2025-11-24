"""
TRP trainer for MemoryMaze.

Same tabular REINFORCE as baseline, but:
- compute policy KL per episode
- TRP pacer produces dt_eff
- lr_eff = lr * dt_eff

Matched budgets: same episodes, same max_steps, same seeds.
"""

from __future__ import annotations
import numpy as np
from collections import defaultdict

from benches.envs.memory_maze import MemoryMaze
from vireon.trp import TRPPacer


def softmax(logits):
    logits = logits - np.max(logits)
    e = np.exp(logits)
    return e / np.sum(e)


def kl_categorical(p_new, p_old, eps=1e-12):
    p_new = np.clip(p_new, eps, 1.0)
    p_old = np.clip(p_old, eps, 1.0)
    return float(np.sum(p_new * (np.log(p_new) - np.log(p_old))))


def train_trp_reinforce_memory_maze(
    size=5,
    episodes=500,
    lr=0.1,
    gamma=0.99,
    seed=0,
    max_steps=50,
    kappa=0.02,
    leash_power=1.0,
):
    rng = np.random.default_rng(seed)
    env = MemoryMaze(size=size, max_steps=max_steps, seed=seed)

    n_actions = len(env.ACTIONS)
    theta = defaultdict(lambda: np.zeros(n_actions, dtype=np.float64))

    pacer = TRPPacer(dt=1.0, kappa=kappa, leash_power=leash_power)
    alpha_t = pacer.alpha_schedule.alpha0

    ep_returns = []
    policy_kl_trace = []
    dt_eff_trace = []
    div_trace = []

    baseline = 0.0
    beta = 0.9

    for ep in range(episodes):
        obs = env.reset()
        hint_mem = int(obs[3])
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

        # estimate episode KL (new vs old on visited states)
        kl_ep = 0.0
        for state, probs_old in zip(traj_states, traj_probs):
            probs_new = softmax(theta[state])
            kl_ep += kl_categorical(probs_new, probs_old)
        kl_ep = kl_ep / max(1, len(traj_states))

        # TRP pacing
        dt_eff, alpha_next, divergence = pacer.step(kl_val=kl_ep, alpha_t=alpha_t)
        lr_eff = lr * dt_eff

        # policy update
        for state, a, Gt, probs in zip(traj_states, traj_actions, returns, traj_probs):
            adv = Gt - baseline
            grad = -probs
            grad[a] += 1.0
            theta[state] += lr_eff * adv * grad
            baseline = beta * baseline + (1 - beta) * Gt

        ep_returns.append(sum(traj_rewards))
        policy_kl_trace.append(kl_ep)
        dt_eff_trace.append(dt_eff)
        div_trace.append(divergence)

        alpha_t = alpha_next

    return {
        "returns": np.array(ep_returns, dtype=np.float64),
        "policy_kl": np.array(policy_kl_trace, dtype=np.float64),
        "dt_eff": np.array(dt_eff_trace, dtype=np.float64),
        "divergence": np.array(div_trace, dtype=np.float64),
    }


def main():
    out = train_trp_reinforce_memory_maze()
    print("TRP mean return:", out["returns"].mean())
    print("Mean dt_eff:", out["dt_eff"].mean())


if __name__ == "__main__":
    main()
