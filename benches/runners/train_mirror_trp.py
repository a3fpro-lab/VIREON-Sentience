"""
TRP trainer for MirrorGame.

Same REINFORCE baseline, but:
- estimate episode KL
- TRP pacer -> dt_eff
- lr_eff = lr * dt_eff
"""

from __future__ import annotations
import numpy as np
from collections import defaultdict

from benches.envs.mirror_game import MirrorGame
from vireon.trp import TRPPacer


def softmax(logits):
    logits = logits - np.max(logits)
    e = np.exp(logits)
    return e / np.sum(e)


def kl_categorical(p_new, p_old, eps=1e-12):
    p_new = np.clip(p_new, eps, 1.0)
    p_old = np.clip(p_old, eps, 1.0)
    return float(np.sum(p_new * (np.log(p_new) - np.log(p_old))))


def train_trp_reinforce_mirror_game(
    episodes=500,
    lr=0.1,
    gamma=0.99,
    seed=0,
    p_mirror=0.9,
    max_steps=50,
    kappa=0.02,
    leash_power=1.0,
):
    rng = np.random.default_rng(seed)
    env = MirrorGame(p_mirror=p_mirror, max_steps=max_steps, seed=seed)

    n_actions = 2
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

        # episode KL
        kl_ep = 0.0
        for state, probs_old in zip(traj_states, traj_probs):
            probs_new = softmax(theta[state])
            kl_ep += kl_categorical(probs_new, probs_old)
        kl_ep /= max(1, len(traj_states))

        # TRP pacing
        dt_eff, alpha_next, divergence = pacer.step(kl_val=kl_ep, alpha_t=alpha_t)
        lr_eff = lr * dt_eff

        # update
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
    out = train_trp_reinforce_mirror_game()
    print("TRP mean return:", out["returns"].mean())


if __name__ == "__main__":
    main()
