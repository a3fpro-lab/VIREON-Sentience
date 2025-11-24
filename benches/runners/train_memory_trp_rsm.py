"""
TRP + RSM trainer for MemoryMaze.

Adds:
- MirrorModel on policy π(a|s)
- SelfForecaster on reward surprise
- CollapseGuard pressure multipliers

Then TRP pacing uses KL * pressure.
"""

from __future__ import annotations
import numpy as np
from collections import defaultdict

from benches.envs.memory_maze import MemoryMaze
from vireon.trp import TRPPacer
from vireon.rsm import MirrorModel, SelfForecaster, CollapseGuard


def softmax(logits):
    logits = logits - np.max(logits)
    e = np.exp(logits)
    return e / np.sum(e)


def kl_categorical(p_new, p_old, eps=1e-12):
    p_new = np.clip(p_new, eps, 1.0)
    p_old = np.clip(p_old, eps, 1.0)
    return float(np.sum(p_new * (np.log(p_new) - np.log(p_old))))


def train_trp_rsm_reinforce_memory_maze(
    size=5,
    episodes=500,
    lr=0.1,
    gamma=0.99,
    seed=0,
    max_steps=50,
    kappa=0.02,
    leash_power=1.0,
    w_policy=1.0,
    w_mirror=1.0,
    w_self=1.0,
):
    rng = np.random.default_rng(seed)
    env = MemoryMaze(size=size, max_steps=max_steps, seed=seed)

    n_actions = len(env.ACTIONS)
    theta = defaultdict(lambda: np.zeros(n_actions, dtype=np.float64))

    pacer = TRPPacer(dt=1.0, kappa=kappa, leash_power=leash_power)
    alpha_t = pacer.alpha_schedule.alpha0

    mirror = MirrorModel(n_actions=n_actions, n_buckets=1, ema_beta=0.98)
    forecaster = SelfForecaster(beta=0.98, window=50)
    guard = CollapseGuard(w_policy=w_policy, w_mirror=w_mirror, w_self=w_self)

    ep_returns = []
    policy_kl_trace = []
    mirror_kl_trace = []
    gself_trace = []
    pressure_trace = []
    dt_eff_trace = []

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

        # policy KL episode
        kl_ep = 0.0
        kl_m_ep = 0.0

        # update mirror + compute KLs
        for state, probs_old in zip(traj_states, traj_probs):
            probs_new = softmax(theta[state])
            kl_ep += kl_categorical(probs_new, probs_old)

            # mirror sees same "obs bucket"
            obs_stub = np.array([0.0], dtype=np.float32)
            kl_m_ep += mirror.update(obs_stub, probs_new)

        kl_ep = kl_ep / max(1, len(traj_states))
        kl_m_ep = kl_m_ep / max(1, len(traj_states))

        # self forecast on reward surprise
        ep_ret = sum(traj_rewards)
        e_t = ep_ret - baseline
        forecaster.update(e_t)
        g_self = forecaster.self_surprise_gap()

        # pressure
        pressure = guard.pressure(kl_policy=kl_ep, kl_mirror=kl_m_ep, g_self=g_self)

        # TRP pacing
        dt_eff, alpha_next, divergence = pacer.step(kl_val=kl_ep * pressure, alpha_t=alpha_t)
        lr_eff = lr * dt_eff

        # update policy
        for state, a, Gt, probs in zip(traj_states, traj_actions, returns, traj_probs):
            adv = Gt - baseline
            grad = -probs
            grad[a] += 1.0
            theta[state] += lr_eff * adv * grad

        baseline = beta * baseline + (1 - beta) * ep_ret

        ep_returns.append(ep_ret)
        policy_kl_trace.append(kl_ep)
        mirror_kl_trace.append(kl_m_ep)
        gself_trace.append(g_self)
        pressure_trace.append(pressure)
        dt_eff_trace.append(dt_eff)

        alpha_t = alpha_next

    return {
        "returns": np.array(ep_returns, dtype=np.float64),
        "policy_kl": np.array(policy_kl_trace, dtype=np.float64),
        "kl_mirror": np.array(mirror_kl_trace, dtype=np.float64),
        "g_self": np.array(gself_trace, dtype=np.float64),
        "pressure": np.array(pressure_trace, dtype=np.float64),
        "dt_eff": np.array(dt_eff_trace, dtype=np.float64),
    }


def main():
    out = train_trp_rsm_reinforce_memory_maze()
    print("TRP+RSM mean return:", out["returns"].mean())


if __name__ == "__main__":
    main()
