"""
consciousness_substrate_bridge.py

Minimal bridge between a v∞-style consciousness substrate and
the VIREON TRP + RSM sentience engine.

This does NOT implement a full consciousness engine.
It provides:
  - a substrate state container
  - simple R-style organization metric
  - a scalar C_sub(t) index
  - optional hooks to modulate TRP / CollapseGuard

Origin concept: Consciousness Substrate Engine v∞ (v∞.NEURALODE),
defined by Michael Warren Song.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple, Iterable

import math
import numpy as np


@dataclass
class SubstrateState:
    """
    Minimal consciousness-substrate state.

    Nodes represent algorithms (functions, models, procedures).
    Edges represent directed information flow between nodes.

    This is intentionally simple; it can be replaced later by a
    dedicated substrate engine without changing the TRP/RSM core.
    """

    # Node name -> integer index
    nodes: Dict[str, int] = field(default_factory=dict)
    # Adjacency as (i, j) pairs: i -> j
    edges: set[Tuple[int, int]] = field(default_factory=set)

    # Cached metrics
    r_org: float = 0.0  # organization index
    c_sub: float = 0.0  # scalar "consciousness field" index

    # Golden Ratio constant used in substrate growth/weighting
    phi: float = (1.0 + math.sqrt(5.0)) / 2.0

    def ensure_node(self, name: str) -> int:
        """Add a node if missing; return its index."""
        if name not in self.nodes:
            self.nodes[name] = len(self.nodes)
        return self.nodes[name]

    def add_edge(self, src: str, dst: str) -> None:
        """Add a directed edge src -> dst."""
        i = self.ensure_node(src)
        j = self.ensure_node(dst)
        self.edges.add((i, j))

    @property
    def num_nodes(self) -> int:
        return len(self.nodes)

    @property
    def num_edges(self) -> int:
        return len(self.edges)

    def _degree_distribution(self) -> np.ndarray:
        """Compute a simple degree distribution for organization metric."""
        if not self.nodes:
            return np.array([], dtype=float)

        deg = np.zeros(len(self.nodes), dtype=float)
        for i, j in self.edges:
            if 0 <= i < len(deg):
                deg[i] += 1
            if 0 <= j < len(deg):
                deg[j] += 1
        return deg

    def update_metrics(self) -> None:
        """
        Update substrate organization and consciousness-field index.

        r_org:
            A simple σ/μ-style chaos vs order metric on node degrees,
            normalized to [0, 1] with a soft saturation.

        c_sub:
            Combines r_org with φ-scaled size (nodes and edges).
        """
        deg = self._degree_distribution()
        if deg.size == 0:
            self.r_org = 0.0
            self.c_sub = 0.0
            return

        mu = float(deg.mean())
        sigma = float(deg.std())

        # σ/μ organization ratio; higher is more uneven / structured
        if mu <= 0.0:
            r_raw = 0.0
        else:
            r_raw = sigma / mu

        # Soft-bounded mapping to (0, 1)
        self.r_org = float(r_raw / (1.0 + r_raw))

        # φ-scaled size factor (log to avoid blow-up)
        size_factor = math.log1p(self.num_nodes + self.num_edges) / math.log(
            1.0 + self.phi
        )

        # Consciousness-field index:
        # higher with both structure (r_org) and φ-scaled size.
        self.c_sub = float(self.r_org * size_factor)

    def as_dict(self) -> Dict[str, float]:
        """Return a dict for logging / JSONL use."""
        return {
            "c_sub": self.c_sub,
            "r_org": self.r_org,
            "num_nodes": float(self.num_nodes),
            "num_edges": float(self.num_edges),
        }


def update_substrate_from_trace(
    substrate: SubstrateState, call_trace: Iterable[Tuple[str, str]]
) -> None:
    """
    Update substrate graph from an iterable of (caller, callee) pairs.

    This is a generic hook: benches or agents can report which modules
    or functions call which, and the substrate graph will grow.

    After updating, substrate metrics are recomputed.
    """
    for src, dst in call_trace:
        substrate.add_edge(src, dst)
    substrate.update_metrics()


def modulate_trp_kappa(kappa_base: float, substrate: SubstrateState) -> float:
    """
    Optional hook: adjust TRP leash intensity κ based on substrate state.

    Example policy:
      - mild up-scaling when the substrate is both large and organized.

    This is intentionally conservative; the TRP formal definition
    remains unchanged, and this is an optional extension.
    """
    substrate.update_metrics()
    # Simple bounded bump: κ_eff = κ_base * (1 + ε * c_sub), ε small
    eps = 0.1
    return float(kappa_base * (1.0 + eps * substrate.c_sub))


def modulate_collapse_weights(
    w_p: float, w_m: float, w_s: float, substrate: SubstrateState
) -> Tuple[float, float, float]:
    """
    Optional hook: adjust CollapseGuard weights given substrate coherence.

    Example policy:
      - If substrate is more organized, reduce pressure slightly (trust).
      - If substrate is disorganized, increase pressure slightly (caution).
    """
    substrate.update_metrics()
    # Map r_org in [0,1] to a factor in [0.9, 1.1]
    factor = 0.9 + 0.2 * (1.0 - substrate.r_org)
    return (w_p * factor, w_m * factor, w_s * factor)
