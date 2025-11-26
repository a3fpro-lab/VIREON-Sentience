"""
engine_trp_rsm.py

RSM+TRP Engine (single- and multi-agent skeleton) for the Sentience repo.

Core ideas:
- TRP clock: T_t = R_t * P_t → controls dt_eff, learning rate eta_t, and KL budget eps_t
- RSM: self-model predicts next internal state s_{t+1}
- World model: predicts next world latent z_{t+1}
- Policy: acts from (s_t, h_t); updates are TRP-gated by a KL trust-region
- Multi-agent consensus: agents negotiate a shared latent via trust-weighted averaging

This file is deliberately modular and minimal:
- Plug in your own Encoder / SelfModel / WorldModel / Policy / ValueNet.
- Plug in your own structure_signal() and perception_gain() for R_t and P_t.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Basic configs and utility structures
# ---------------------------------------------------------------------------

@dataclass
class TRPConfig:
    eta0: float = 3e-4        # base learning rate
    gamma: float = 1.0        # T-scale for dt_eff
    eps0: float = 0.1         # base KL budget
    beta: float = 1.0         # eps_t = eps0 * dt_eff^beta
    kl_lambda: float = 1.0    # weight for KL penalty when using soft constraint


@dataclass
class LossWeights:
    alpha_self: float = 1.0
    alpha_world: float = 1.0
    alpha_social: float = 0.0  # only used for multi-agent
    alpha_value: float = 0.5   # if you plug in value loss
    alpha_entropy: float = 0.0 # optional policy entropy reg


@dataclass
class GateThresholds:
    tau_self: float = 0.1
    tau_world: float = 0.1
    tau_social: float = 0.1  # multi-agent
    # KL threshold is eps_t from TRP; no fixed scalar here


@dataclass
class StepLog:
    t: int
    R_t: float
    P_t: float
    T_t: float
    dt_eff: float
    eta_t: float
    eps_t: float
    d_self: float
    d_world: float
    d_kl: float
    sat: int
    loss_task: float
    loss_self: float
    loss_world: float
    loss_kl: float


# ---------------------------------------------------------------------------
# TRP clock
# ---------------------------------------------------------------------------

class TRPClock:
    """
    TRP clock for a single agent:
        T_t = R_t * P_t
        dt_eff = 1 / (1 + gamma * T_t)
        eta_t = eta0 * dt_eff
        eps_t = eps0 * dt_eff^beta
    """

    def __init__(self, cfg: TRPConfig):
        self.cfg = cfg

    def step(self, R_t: torch.Tensor, P_t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        R_t, P_t: scalar tensors or broadcastable; returns (T_t, eta_t, eps_t)
        """
        T_t = R_t * P_t
        dt_eff = 1.0 / (1.0 + self.cfg.gamma * T_t)
        eta_t = self.cfg.eta0 * dt_eff
        eps_t = self.cfg.eps0 * (dt_eff ** self.cfg.beta)
        return T_t, eta_t, eps_t


# ---------------------------------------------------------------------------
# Placeholders for your models
# Plug in your real implementations from elsewhere in the repo.
# ---------------------------------------------------------------------------

class Encoder(nn.Module):
    def __init__(self, x_dim: int, s_dim: int, h_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(x_dim + s_dim, 128),
            nn.ReLU(),
            nn.Linear(128, h_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([x, s], dim=-1))


class SelfModel(nn.Module):
    def __init__(self, s_dim: int, h_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(s_dim + h_dim, 128),
            nn.ReLU(),
            nn.Linear(128, s_dim),
        )

    def forward(self, s: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([s, h], dim=-1))


class WorldModel(nn.Module):
    def __init__(self, s_dim: int, h_dim: int, z_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(s_dim + h_dim, 128),
            nn.ReLU(),
            nn.Linear(128, z_dim),
        )

    def forward(self, s: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([s, h], dim=-1))


class PolicyNet(nn.Module):
    """
    Simple categorical policy over discrete actions.
    Replace with your real policy architecture (e.g., Gaussian for continuous).
    """
    def __init__(self, s_dim: int, h_dim: int, num_actions: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(s_dim + h_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions),
        )

    def forward(self, s: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        logits = self.net(torch.cat([s, h], dim=-1))
        return logits  # (batch, num_actions)


class ValueNet(nn.Module):
    def __init__(self, s_dim: int, h_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(s_dim + h_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, s: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([s, h], dim=-1))


# ---------------------------------------------------------------------------
# Utility functions for structure/perception signals (R_t, P_t)
# NOTE: replace with your actual TRP signal definitions.
# ---------------------------------------------------------------------------

def structure_signal(x_t: torch.Tensor, x_prev: torch.Tensor) -> torch.Tensor:
    """
    Example structure signal: L2 difference between consecutive obs.
    Replace this with your true R_t definition.
    """
    return (x_t - x_prev).pow(2).mean(dim=-1, keepdim=True)


def perception_gain(logits: torch.Tensor) -> torch.Tensor:
    """
    Example perception gain: inverse of policy entropy.
    Higher when policy is confident (low entropy).
    """
    probs = F.softmax(logits, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)
    entropy = -(probs * log_probs).sum(dim=-1, keepdim=True)  # (batch,1)
    gain = 1.0 / (1.0 + entropy)  # simple bounded transform
    return gain


# ---------------------------------------------------------------------------
# KL utilities
# ---------------------------------------------------------------------------

def kl_categorical(p_logits: torch.Tensor, q_logits: torch.Tensor) -> torch.Tensor:
    """
    KL(q || p) for categorical distributions given logits.
    p_logits: reference (old policy)
    q_logits: candidate (new policy)
    Returns: KL per sample (batch, 1)
    """
    p_log_probs = F.log_softmax(p_logits, dim=-1)
    q_log_probs = F.log_softmax(q_logits, dim=-1)
    q_probs = q_log_probs.exp()
    kl = (q_probs * (q_log_probs - p_log_probs)).sum(dim=-1, keepdim=True)
    return kl


# ---------------------------------------------------------------------------
# Single-agent RSM+TRP Engine
# ---------------------------------------------------------------------------

class RSMPTRPAgent(nn.Module):
    """
    Single-agent engine:
      - encoder, self_model, world_model, policy, value
      - TRP clock gating learning rate & KL budget
      - RSM + world model consistency losses
      - TRP KL-leashed update (prox-like via step scaling)
    """

    def __init__(
        self,
        x_dim: int,
        s_dim: int,
        z_dim: int,
        num_actions: int,
        trp_cfg: TRPConfig,
        loss_weights: LossWeights,
        gate_thresholds: GateThresholds,
        device: str = "cpu",
    ):
        super().__init__()
        self.device = torch.device(device)

        self.encoder = Encoder(x_dim, s_dim, h_dim=64).to(self.device)
        self.self_model = SelfModel(s_dim, h_dim=64).to(self.device)
        self.world_model = WorldModel(s_dim, h_dim=64, z_dim=z_dim).to(self.device)
        self.policy = PolicyNet(s_dim, h_dim=64, num_actions=num_actions).to(self.device)
        self.value_net = ValueNet(s_dim, h_dim=64).to(self.device)

        self.trp_cfg = trp_cfg
        self.trp_clock = TRPClock(trp_cfg)
        self.loss_w = loss_weights
        self.gates = gate_thresholds

        # Optimizer over all params (you can split if desired).
        self.optimizer = torch.optim.Adam(self.parameters(), lr=trp_cfg.eta0)

        # Storage for previous policy logits (for KL)
        self._last_policy_logits: Optional[torch.Tensor] = None

    # ---------------------------------------------------------------------
    # Forward / rollout primitives
    # ---------------------------------------------------------------------

    def encode(self, x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        return self.encoder(x, s)

    def predict_self(self, s: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        return self.self_model(s, h)

    def predict_world(self, s: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        return self.world_model(s, h)

    def policy_logits(self, s: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        return self.policy(s, h)

    def value(self, s: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        return self.value_net(s, h)

    def act(self, s: torch.Tensor, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns sampled actions and logits.
        """
        logits = self.policy_logits(s, h)
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs=probs)
        a = dist.sample()
        return a, logits

    # ---------------------------------------------------------------------
    # Losses
    # ---------------------------------------------------------------------

    def self_loss(self, s_next: torch.Tensor, s_hat_next: torch.Tensor) -> torch.Tensor:
        # MSE-based self-model error
        return (s_next - s_hat_next).pow(2).mean()

    def world_loss(self, z_next: torch.Tensor, z_hat_next: torch.Tensor) -> torch.Tensor:
        # MSE-based world-model error
        return (z_next - z_hat_next).pow(2).mean()

    def policy_loss(
        self,
        logits: torch.Tensor,
        actions: torch.Tensor,
        advantages: torch.Tensor,
    ) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=-1)
        act_log_probs = log_probs.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
        return -(act_log_probs * advantages.detach()).mean()

    def value_loss(self, values: torch.Tensor, returns: torch.Tensor) -> torch.Tensor:
        return 0.5 * (values.squeeze(-1) - returns).pow(2).mean()

    def kl_penalty(
        self,
        old_logits: torch.Tensor,
        new_logits: torch.Tensor,
        eps_t: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Soft KL penalty enforcing KL(new || old) <= eps_t.
        Returns: (loss_kl, d_kl_mean)
        """
        if old_logits is None:
            # First step: no KL constraint
            return torch.tensor(0.0, device=new_logits.device), torch.tensor(0.0, device=new_logits.device)

        d_kl = kl_categorical(old_logits.detach(), new_logits)  # (batch,1)
        d_kl_mean = d_kl.mean()
        # Soft penalty: max(0, KL - eps_t)
        # eps_t may be scalar or tensor; broadcast carefully
        penalty = F.relu(d_kl_mean - eps_t.mean()) * self.trp_cfg.kl_lambda
        return penalty, d_kl_mean

    # ---------------------------------------------------------------------
    # Single-step TRP-gated update
    # ---------------------------------------------------------------------

    def update_step(
        self,
        batch: Dict[str, torch.Tensor],
        x_prev: torch.Tensor,
        log_advantages: bool = False,
    ) -> StepLog:
        """
        Perform one TRP-gated gradient update on a batch of transitions.

        batch keys (minimal):
            'x_t', 's_t', 'z_t', 'x_next', 's_next', 'z_next',
            'actions', 'returns', 'advantages'
        """

        # Move to device
        x_t = batch["x_t"].to(self.device)
        s_t = batch["s_t"].to(self.device)
        z_t = batch["z_t"].to(self.device)
        x_next = batch["x_next"].to(self.device)
        s_next = batch["s_next"].to(self.device)
        z_next = batch["z_next"].to(self.device)
        actions = batch["actions"].to(self.device)
        returns = batch["returns"].to(self.device)
        advantages = batch["advantages"].to(self.device)

        # Encode
        h_t = self.encode(x_t, s_t)

        # Predict self & world
        s_hat_next = self.predict_self(s_t, h_t)
        z_hat_next = self.predict_world(s_t, h_t)

        # Policy + value
        logits = self.policy_logits(s_t, h_t)
        values = self.value(s_t, h_t)

        # Structure + perception signals
        R_t = structure_signal(x_t, x_prev.to(self.device))  # (batch,1)
        P_t = perception_gain(logits)                        # (batch,1)

        # TRP clock
        T_t, eta_t, eps_t = self.trp_clock.step(R_t, P_t)

        # Losses
        L_task = self.policy_loss(logits, actions, advantages)
        L_value = self.value_loss(values, returns)
        L_self = self.self_loss(s_next, s_hat_next)
        L_world = self.world_loss(z_next, z_hat_next)
        L_kl, d_kl_mean = self.kl_penalty(self._last_policy_logits, logits)

        loss_total = (
            L_task
            + self.loss_w.alpha_value * L_value
            + self.loss_w.alpha_self * L_self
            + self.loss_w.alpha_world * L_world
            + L_kl
        )

        # Gradient step with TRP-scaled lr
        self.optimizer.zero_grad()
        loss_total.backward()

        # Override optimizer lr by eta_t mean (simple approach)
        lr = eta_t.mean().item()
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr

        self.optimizer.step()

        # Update stored logits for KL reference
        self._last_policy_logits = logits.detach()

        # Gate diagnostics
        d_self_val = self.self_loss(s_next, s_hat_next).detach().item()
        d_world_val = self.world_loss(z_next, z_hat_next).detach().item()

        # sat_t: 1 if self/world/KL all under thresholds
        sat_t = int(
            (d_self_val <= self.gates.tau_self)
            and (d_world_val <= self.gates.tau_world)
            and (d_kl_mean.item() <= eps_t.mean().item())
        )

        step_log = StepLog(
            t=0,  # you can fill in real t from caller
            R_t=R_t.mean().item(),
            P_t=P_t.mean().item(),
            T_t=T_t.mean().item(),
            dt_eff=float(1.0 / (1.0 + self.trp_cfg.gamma * T_t.mean().item())),
            eta_t=lr,
            eps_t=eps_t.mean().item(),
            d_self=d_self_val,
            d_world=d_world_val,
            d_kl=d_kl_mean.item(),
            sat=sat_t,
            loss_task=L_task.detach().item(),
            loss_self=L_self.detach().item(),
            loss_world=L_world.detach().item(),
            loss_kl=L_kl.detach().item(),
        )
        return step_log


# ---------------------------------------------------------------------------
# Multi-agent consensus + trust skeleton
# ---------------------------------------------------------------------------

class MultiAgentRSMPTRP:
    """
    Thin wrapper for multiple RSMPTRPAgent instances with trust-weighted
    consensus over world latents.

    This is a skeleton: you decide how to feed env transitions in and how
    to store per-agent batches. The key bits:
      - trust weights w_i
      - consensus latent z_bar
      - social loss L_social per agent
      - trust decay based on d_social
    """

    def __init__(
        self,
        agents: List[RSMPTRPAgent],
        gate_thresholds: GateThresholds,
        trust_lambda0: float = 0.1,
        trust_tau_social: float = 0.2,
        device: str = "cpu",
    ):
        self.agents = agents
        self.num_agents = len(agents)
        self.device = torch.device(device)
        self.gates = gate_thresholds
        self.trust_lambda0 = trust_lambda0
        self.trust_tau_social = trust_tau_social

        # Trust weights per agent (normalized)
        self.w = torch.ones(self.num_agents, 1, device=self.device) / self.num_agents
        self.lambda_t = trust_lambda0

    def consensus_latent(self, z_hats: List[torch.Tensor]) -> torch.Tensor:
        """
        z_hats: list of (batch, z_dim) predicted latents from each agent.
        Returns trust-weighted average latent (batch, z_dim).
        """
        # w: (n_agents,1) → (n_agents,batch,1)
        batch_size = z_hats[0].shape[0]
        w_expanded = self.w.view(self.num_agents, 1, 1).expand(-1, batch_size, 1)
        z_stack = torch.stack(z_hats, dim=0)  # (n_agents, batch, z_dim)
        weighted = w_expanded * z_stack
        z_bar = weighted.sum(dim=0) / (w_expanded.sum(dim=0) + 1e-8)
        return z_bar

    def update_trust(
        self,
        z_bar: torch.Tensor,
        z_hats: List[torch.Tensor],
        sat_window: Optional[float] = None,
        noise_sigma: float = 0.0,
    ) -> None:
        """
        Update trust weights based on social disagreement:
            d_social^i = ||z_bar - z_hat^i||^2
        with optional Gaussian noise only for trust.
        """
        n = self.num_agents
        d_social = []
        for i in range(n):
            # MSE distance as social disagreement
            diff = (z_bar.detach() - z_hats[i].detach()).pow(2).mean(dim=-1, keepdim=True)  # (batch,1)
            d_i = diff.mean()  # scalar
            if noise_sigma > 0.0:
                d_i = d_i + noise_sigma * torch.randn_like(d_i)
            d_social.append(d_i)

        d_social_t = torch.stack(d_social, dim=0)  # (n_agents, 1)

        # Optionally adapt lambda_t based on sat_window (global sat%)
        if sat_window is not None:
            # Example: downscale lambda when sat is high
            if sat_window > 0.9:
                self.lambda_t = self.lambda_t * 0.8
            else:
                self.lambda_t = self.trust_lambda0 * (1.0 + 0.5 * (1.0 - sat_window))

        # Exponential decay for disagreements above tau_social
        penalties = torch.clamp(d_social_t - self.trust_tau_social, min=0.0)
        new_w = self.w * torch.exp(-self.lambda_t * penalties)
        # Normalize
        self.w = new_w / (new_w.sum() + 1e-8)

    # You can add a "step" method here that:
    #   - gathers z_hats from all agents
    #   - computes z_bar
    #   - computes social losses for each agent (if you want)
    #   - updates trust
    #   - calls agent.update_step(...) with their own batches


# ---------------------------------------------------------------------------
# END OF FILE
# ---------------------------------------------------------------------------
