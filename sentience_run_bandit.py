"""
sentience_run_bandit.py

Structured bandit demo for the TRP–RSM engine + sentience metrics.

- Environment: multi-armed bandit with structured rewards.
- Agent: RSMPTRPAgent from engine_trp_rsm.py
- Metrics: SMC, SII, ICI, IGI → C-vector

This is still simple, but rewards are now from a real task instead of
pure noise. Observations encode arm identity + reward signal.
"""

from __future__ import annotations
from typing import Dict, List

import torch

from engine_trp_rsm import (
    RSMPTRPAgent,
    TRPConfig,
    LossWeights,
    GateThresholds,
    StepLog,
)
from sentience_metrics import (
    compute_smc,
    compute_sii,
    compute_ici,
    compute_igi,
)


# ---------------------------------------------------------------------------
# Structured bandit environment
# ---------------------------------------------------------------------------

class StructuredBanditEnv:
    """
    Simple structured multi-armed bandit.

    - num_arms arms, each with a fixed mean reward.
    - Observations encode chosen arm (one-hot) + last reward.

    This env is stateless w.r.t. dynamics: reward distribution is fixed.
    """

    def __init__(
        self,
        num_arms: int = 4,
        obs_dim: int = 8,
        reward_std: float = 0.5,
        device: str = "cpu",
    ):
        self.num_arms = num_arms
        self.obs_dim = obs_dim
        self.reward_std = reward_std
        self.device = torch.device(device)

        # Structured means: ascending arms
        # e.g. [-0.5, 0.0, 0.5, 1.0]
        base = torch.linspace(-0.5, 1.0, steps=num_arms)
        self.arm_means = base.to(self.device)

    def reset(self, batch_size: int) -> torch.Tensor:
        """
        Returns initial observation x_0: all zeros.
        Shape: (batch_size, obs_dim)
        """
        return torch.zeros(batch_size, self.obs_dim, device=self.device)

    def step(self, actions: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        actions: (batch_size,) int tensor of arm indices.

        Returns:
            obs_next: (batch_size, obs_dim)
            rewards:  (batch_size, 1)
        """
        batch_size = actions.shape[0]

        # Fetch mean reward per sample
        means = self.arm_means[actions]  # (batch_size,)
        noise = self.reward_std * torch.randn(batch_size, device=self.device)
        rewards = means + noise  # (batch_size,)

        # Encode obs: one-hot arm + reward in last dim
        obs_next = torch.zeros(batch_size, self.obs_dim, device=self.device)
        # one-hot for arms in first num_arms dims
        obs_next[torch.arange(batch_size), actions] = 1.0
        # last dim = reward
        obs_next[:, -1] = rewards

        return {
            "x_next": obs_next,
            "rewards": rewards.view(batch_size, 1),
        }


# ---------------------------------------------------------------------------
# Batch builder using bandit env + TRP–RSM engine shapes
# ---------------------------------------------------------------------------

def make_bandit_batch(
    env: StructuredBanditEnv,
    agent: RSMPTRPAgent,
    x_t: torch.Tensor,
    s_dim: int,
    z_dim: int,
    num_actions: int,
) -> Dict[str, torch.Tensor]:
    """
    Builds a batch with real actions/rewards from the bandit, but keeps
    self/world latents synthetic (for now).

    Shapes:
        x_t:      (batch, x_dim)
        s_t:      (batch, s_dim)
        z_t:      (batch, z_dim)
        x_next:   (batch, x_dim)
        s_next:   (batch, s_dim)
        z_next:   (batch, z_dim)
        actions:  (batch,)
        returns:  (batch,1)
        advantages: (batch,)
    """
    device = x_t.device
    batch_size = x_t.size(0)

    # Sample synthetic self-state and world latent
    s_t = torch.randn(batch_size, s_dim, device=device)
    z_t = torch.randn(batch_size, z_dim, device=device)

    # Policy from current self + obs
    h_t = agent.encode(x_t, s_t)
    logits = agent.policy_logits(s_t, h_t)
    probs = torch.softmax(logits, dim=-1)
    dist = torch.distributions.Categorical(probs=probs)
    actions = dist.sample()  # (batch,)

    # Step env with chosen actions
    env_out = env.step(actions)
    x_next = env_out["x_next"]
    rewards = env_out["rewards"]  # (batch,1)

    # Synthetic next self/world latents (could be learned later)
    s_next = torch.randn(batch_size, s_dim, device=device)
    z_next = torch.randn(batch_size, z_dim, device=device)

    # One-step returns = rewards
    returns = rewards
    # Simple advantage baseline = reward - mean(reward)
    advantages = (rewards.view(-1) - rewards.mean())

    batch = {
        "x_t": x_t,
        "s_t": s_t,
        "z_t": z_t,
        "x_next": x_next,
        "s_next": s_next,
        "z_next": z_next,
        "actions": actions,
        "returns": returns,
        "advantages": advantages,
    }
    return batch


# ---------------------------------------------------------------------------
# Bandit demo runner
# ---------------------------------------------------------------------------

def run_bandit_demo(
    T: int = 200,
    batch_size: int = 32,
    x_dim: int = 8,
    s_dim: int = 16,
    z_dim: int = 8,
    num_actions: int = 4,
):
    device = "cpu"

    # Configs
    trp_cfg = TRPConfig(
        eta0=3e-4,
        gamma=1.0,
        eps0=0.1,
        beta=1.0,
        kl_lambda=1.0,
    )
    loss_w = LossWeights(
        alpha_self=1.0,
        alpha_world=1.0,
        alpha_social=0.0,
        alpha_value=0.5,
        alpha_entropy=0.0,
    )
    gates = GateThresholds(
        tau_self=0.1,
        tau_world=0.1,
        tau_social=0.1,
    )

    # Agent
    agent = RSMPTRPAgent(
        x_dim=x_dim,
        s_dim=s_dim,
        z_dim=z_dim,
        num_actions=num_actions,
        trp_cfg=trp_cfg,
        loss_weights=loss_w,
        gate_thresholds=gates,
        device=device,
    )

    # Environment
    env = StructuredBanditEnv(
        num_arms=num_actions,
        obs_dim=x_dim,
        reward_std=0.5,
        device=device,
    )

    # Initial obs and prev obs
    x_t = env.reset(batch_size=batch_size)
    x_prev = x_t.clone()

    step_logs: List[StepLog] = []
    world_errors: List[float] = []
    rewards_history: List[float] = []

    for t in range(T):
        batch = make_bandit_batch(env, agent, x_t, s_dim, z_dim, num_actions)

        # TRP-gated update step
        step_log = agent.update_step(batch, x_prev=x_prev, t=t)
        step_logs.append(step_log)
        world_errors.append(step_log.d_world)

        # Rewards for monitoring
        rewards_history.append(batch["returns"].mean().item())

        # Next obs -> used as current and previous for next step
        x_prev = x_t.detach()
        x_t = batch["x_next"].detach()

    # -----------------------------------------------------------------------
    # Sentience metrics
    # -----------------------------------------------------------------------

    # C1: Self-Model Coherence
    smc = compute_smc(step_logs, tau_self=gates.tau_self)

    # C2: Self Influence Index
    logits_with_self = agent.get_policy_trace_with_self()
    logits_no_self = agent.get_policy_trace_no_self()
    if (logits_with_self is None) or (logits_no_self is None):
        sii = 0.0
    else:
        sii = compute_sii(logits_with_self, logits_no_self)

    # C3: Identity Continuity Index
    state_trace = agent.get_state_trace()
    if state_trace is None:
        ici = 0.0
    else:
        ici = compute_ici(state_trace, delta=max(1, T // 10))

    # C4: Ignition Index
    world_errors_tensor = torch.tensor(world_errors).view(1, T)  # (N_agents=1, T)
    igi = compute_igi(world_errors_tensor, surprise_factor=3.0, window=5)

    # Monitoring reward
    avg_reward = sum(rewards_history) / len(rewards_history)

    # C-vector
    print("=== Sentience Engine Bandit Demo ===")
    print(f"Steps (T):       {T}")
    print(f"Batch size:      {batch_size}")
    print(f"Arms:            {num_actions}")
    print(f"Avg reward:      {avg_reward:.4f}")
    print()
    print("C-vector (C1–C4):")
    print(f"  C1 SMC  (self-model coherence):      {smc:.4f}")
    print(f"  C2 SII  (self influence index):      {sii:.4f}")
    print(f"  C3 ICI  (identity continuity index): {ici:.4f}")
    print(f"  C4 IGI  (ignition index):            {igi:.4f}")
    print()
    print("Note: this demo uses a structured bandit environment.")
    print("      Swap in your own env for richer semantics.")


if __name__ == "__main__":
    run_bandit_demo()
