"""
sentience_metrics.py

Consciousness-flavored metrics for the Sentience repo.

C1: SMC  (Self-Model Coherence)
C2: SII  (Self Influence Index)
C3: ICI  (Identity Continuity Index)
C4: IGI  (Ignition Index; optional)

These are *diagnostics*, not proofs of consciousness.
"""

from __future__ import annotations
from typing import List, Dict, Tuple
import torch
import torch.nn.functional as F

from engine_trp_rsm import StepLog


# ---------------------------------------------------------------------------
# C1: Self-Model Coherence (SMC)
# ---------------------------------------------------------------------------

def compute_smc(step_logs: List[StepLog], tau_self: float) -> float:
    """
    Self-Model Coherence:
        SMC = 1 - mean(clip(d_self / tau_self, 0, 1))
    Higher SMC means the self-model consistently predicts self-state
    within the gate threshold.
    """
    if not step_logs:
        return 0.0

    import math
    ratios = [min(1.0, sl.d_self / max(tau_self, 1e-8)) for sl in step_logs]
    mean_ratio = sum(ratios) / len(ratios)
    smc = 1.0 - mean_ratio
    return max(0.0, min(1.0, smc))


# ---------------------------------------------------------------------------
# C2: Self Influence Index (SII)
# ---------------------------------------------------------------------------

def compute_sii(
    logits_with_self: torch.Tensor,
    logits_no_self: torch.Tensor,
) -> float:
    """
    Self Influence Index:
        SII = E_t[ KL( pi_with_self || pi_no_self ) ]

    logits_with_self:  (T, batch, A)
    logits_no_self:    (T, batch, A)

    Returns scalar float (mean over time and batch).
    """
    # Flatten time+batch
    w = logits_with_self.view(-1, logits_with_self.size(-1))
    ns = logits_no_self.view(-1, logits_no_self.size(-1))

    log_p = F.log_softmax(w, dim=-1)
    log_q = F.log_softmax(ns, dim=-1)
    p = log_p.exp()

    kl = (p * (log_p - log_q)).sum(dim=-1)  # (T*batch,)
    return float(kl.mean().item())


# ---------------------------------------------------------------------------
# C3: Identity Continuity Index (ICI)
# ---------------------------------------------------------------------------

def compute_ici(
    states: torch.Tensor,
    delta: int = 10,
) -> float:
    """
    Identity Continuity Index for one agent:

    states: (T, dim) tensor of self-states s_t
    delta:  temporal offset

    We compute:
        Δ_s(Δ) = mean_t || s_{t+Δ} - s_t ||^2
        ICI = 1 / (1 + Δ_s(Δ))

    Higher ICI = more continuous identity over Δ steps.
    """
    T = states.size(0)
    if T <= delta:
        return 0.0

    diffs = states[delta:] - states[:-delta]
    d2 = (diffs ** 2).mean().item()
    ici = 1.0 / (1.0 + d2)
    return ici


# ---------------------------------------------------------------------------
# C4: Ignition Index (IGI) – optional global broadcast metric
# ---------------------------------------------------------------------------

def compute_igi(
    world_errors_per_agent: torch.Tensor,
    surprise_factor: float = 3.0,
    window: int = 5,
) -> float:
    """
    Ignition Index:

    world_errors_per_agent: (N_agents, T) tensor, e.g. per-step world error.
    surprise_factor: spike threshold = surprise_factor * median error.
    window: how many steps after a spike we look for "ignition".

    For each spike event (agent i0 at time t0), we ask:
        - How many agents show a large response in [t0, t0+window]?

    IGI = average fraction of agents that respond to each spike.
    """

    N, T = world_errors_per_agent.shape
    if T <= 1:
        return 0.0

    med = torch.median(world_errors_per_agent)
    spike_thresh = surprise_factor * med

    # Find (agent, time) pairs where error spikes
    spike_mask = world_errors_per_agent > spike_thresh  # (N,T)
    spike_idx = spike_mask.nonzero(as_tuple=False)  # (num_spikes, 2)

    if spike_idx.numel() == 0:
        return 0.0

    responses = []
    for i in range(spike_idx.size(0)):
        a0, t0 = spike_idx[i].tolist()
        t1 = min(T, t0 + window + 1)

        # Response = error above median in the window
        window_err = world_errors_per_agent[:, t0:t1]  # (N, w)
        response_mask = (window_err > med).any(dim=1)  # (N,)
        frac = response_mask.float().mean().item()
        responses.append(frac)

    return sum(responses) / len(responses)
