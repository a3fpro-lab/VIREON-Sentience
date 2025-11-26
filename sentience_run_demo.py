"""
sentience_run_demo.py

Minimal end-to-end demo for the TRP–RSM engine + sentience metrics.

- Builds a single RSMPTRPAgent
- Runs a toy "environment" with random transitions
- Logs StepLog entries
- Computes SMC, SII, ICI, IGI and prints a C-vector

This is not a real task; it's a wiring test and metrics showcase.
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
# Toy batch generator (fake environment)
# ---------------------------------------------------------------------------

def make_fake_batch(
    batch_size: int,
    x_dim: int,
    s_dim: int,
    z_dim: int,
    num_actions: int,
) -> Dict[str, torch.Tensor]:
    """
    Creates a fake batch of transitions with the right shapes.
    Everything is random; this just exercises the engine plumbing.
    """
    device = "cpu"

    x_t = torch.randn(batch_size, x_dim, device=device)
    s_t = torch.randn(batch_size, s_dim, device=device)
    z_t = torch.randn(batch_size, z_dim, device=device)

    # Next-step states/latents (just random for now)
    x_next = torch.randn(batch_size, x_dim, device=device)
    s_next = torch.randn(batch_size, s_dim, device=device)
    z_next = torch.randn(batch_size, z_dim, device=device)

    # Actions, returns, advantages are random
    actions = torch.randint(low=0, high=num_actions, size=(batch_size,), device=device)
    returns = torch.randn(batch_size, 1, device=device)
    advantages = torch.randn(batch_size, device=device)

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
# Demo runner
# ---------------------------------------------------------------------------

def run_demo(
    T: int = 100,
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

    # Initial previous observation for structure_signal
    x_prev = torch.zeros(batch_size, x_dim, device=device)

    step_logs: List[StepLog] = []
    world_errors: List[float] = []

    for t in range(T):
        batch = make_fake_batch(batch_size, x_dim, s_dim, z_dim, num_actions)
        x_t = batch["x_t"]

        # Single TRP-gated update step
        step_log = agent.update_step(batch, x_prev=x_prev, t=t)
        step_logs.append(step_log)
        world_errors.append(step_log.d_world)

        # For next iteration's R_t
        x_prev = x_t.detach()

    # -----------------------------------------------------------------------
    # Sentience metrics
    # -----------------------------------------------------------------------

    # C1: Self-Model Coherence
    smc = compute_smc(step_logs, tau_self=gates.tau_self)

    # C2: Self Influence Index (needs logits_with_self and logits_no_self traces)
    logits_with_self = agent.get_policy_trace_with_self()
    logits_no_self = agent.get_policy_trace_no_self()
    if (logits_with_self is None) or (logits_no_self is None):
        sii = 0.0
    else:
        sii = compute_sii(logits_with_self, logits_no_self)

    # C3: Identity Continuity Index (needs state trace)
    state_trace = agent.get_state_trace()
    if state_trace is None:
        ici = 0.0
    else:
        # state_trace: (T, dim)
        ici = compute_ici(state_trace, delta=max(1, T // 10))

    # C4: Ignition Index (global broadcast over agents; here just 1 agent)
    world_errors_tensor = torch.tensor(world_errors).view(1, T)  # (N_agents=1, T)
    igi = compute_igi(world_errors_tensor, surprise_factor=3.0, window=5)

    # C-vector
    print("=== Sentience Engine Demo ===")
    print(f"Steps (T):       {T}")
    print(f"Batch size:      {batch_size}")
    print()
    print("C-vector (C1–C4):")
    print(f"  C1 SMC  (self-model coherence):      {smc:.4f}")
    print(f"  C2 SII  (self influence index):      {sii:.4f}")
    print(f"  C3 ICI  (identity continuity index): {ici:.4f}")
    print(f"  C4 IGI  (ignition index):            {igi:.4f}")
    print()
    print("Note: this run uses a random toy environment.")
    print("      Plug in a real env to give these numbers semantic bite.")


if __name__ == "__main__":
    run_demo()
