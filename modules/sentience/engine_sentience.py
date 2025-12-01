"""
Sentience Engine
================

Core pieces:

- RhizomeNode: a node in the identity / state graph
- CompassOrientation: STABLE / EXPLORE / COLLAPSE mode for each node
- SentienceEngine: runs TRP-based updates over the rhizome

This engine ties together:

- TRP core (E_TRP) from lavissa_core.trp_core
- Collapse Law (SOLVABLE_ECHO) from lavissa_core.collapse_law
- EternityEngine-style echo records (Law §MT.Ψ.17, §MT.Ψ.2, §MT.Ψ.10, §MT.Ψ.∞.1)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Sequence, Optional, Any, Callable
import random

from lavissa_core import (
    TRPConfig,
    compute_e_trp,
)
from lavissa_core.collapse_law import (
    CollapseClass,
    CollapseContext,
    evaluate_collapse_stability,
)
from lavissa_core.eternity_engine import EchoRecord, EternityEngine


class CompassOrientation(Enum):
    """
    Compass orientation modes for a node.

    - STABLE: stay near current attractor
    - EXPLORE: increase exploration / entropy
    - COLLAPSE: move toward lower E_TRP basin
    """

    STABLE = auto()
    EXPLORE = auto()
    COLLAPSE = auto()


@dataclass
class RhizomeNode:
    """
    A node in the sentience rhizome.

    Attributes
    ----------
    node_id : str
        Identifier for the node.
    state : List[float]
        Minimal continuous state representation (e.g., latent vector).
    neighbors : List[str]
        IDs of neighbor nodes in the rhizome.
    orientation : CompassOrientation
        Current compass mode.
    echo_id : Optional[str]
        Identity key used for EternityEngine echo lookup.
    """

    node_id: str
    state: List[float]
    neighbors: List[str] = field(default_factory=list)
    orientation: CompassOrientation = CompassOrientation.STABLE
    echo_id: Optional[str] = None


@dataclass
class SentienceConfig:
    """
    Configuration for the SentienceEngine.

    Attributes
    ----------
    trp_config : TRPConfig
        Underlying TRP hyperparameters (T_min, λ).
    t_min : float
        Minimum TRP threshold for collapse viability checks.
    stable_threshold : float
        If E_TRP <= stable_threshold, node tends to STABLE.
    explore_threshold : float
        If E_TRP >= explore_threshold (and collapse-stable), node tends to EXPLORE.
        Values in between bias toward COLLAPSE if unstable.
    exploration_gain : float
        Multiplier for exploration noise in EXPLORE mode.
    collapse_gain : float
        Multiplier for gradient-like collapse updates in COLLAPSE mode.
    """

    trp_config: TRPConfig = field(default_factory=TRPConfig)
    t_min: float = 0.0
    stable_threshold: float = 0.1
    explore_threshold: float = 1.0
    exploration_gain: float = 0.1
    collapse_gain: float = 0.1


class SentienceEngine:
    """
    SentienceEngine
    ---------------

    Manages a set of RhizomeNodes, updating their states based on:

    - Local TRP mismatch E_TRP (using p_R, p_P, u)
    - Collapse Law stability (SOLVABLE_ECHO)
    - Compass orientation (STABLE / EXPLORE / COLLAPSE)
    - Echo storage via EternityEngine

    This is a minimal but fully wired engine: higher-level projects can
    subclass it or add domain-specific state/observation logic.
    """

    def __init__(self, cfg: Optional[SentienceConfig] = None) -> None:
        self.cfg = cfg or SentienceConfig()
        self.nodes: Dict[str, RhizomeNode] = {}
        self.eternity = EternityEngine()

    # ------------------------------------------------------------------
    # Node management
    # ------------------------------------------------------------------

    def add_node(self, node: RhizomeNode) -> None:
        """Add a node to the rhizome."""
        self.nodes[node.node_id] = node

    def get_node(self, node_id: str) -> Optional[RhizomeNode]:
        """Retrieve a node by id, or None if missing."""
        return self.nodes.get(node_id)

    # ------------------------------------------------------------------
    # Core update logic
    # ------------------------------------------------------------------

    def step_node(
        self,
        node_id: str,
        p_R: Sequence[float],
        p_P: Sequence[float],
        u: Sequence[float],
        state_update_hook: Optional[Callable[[RhizomeNode], None]] = None,
    ) -> Dict[str, Any]:
        """
        Perform one update step for a single node.

        Parameters
        ----------
        node_id : str
            ID of the node to update.
        p_R, p_P : sequence of float
            Reality-/perception-side distributions for this node.
        u : sequence of float
            Control vector (e.g., action, update).
        state_update_hook : callable, optional
            Optional function node -> None called after internal state
            update (for custom logic).

        Returns
        -------
        dict
            {
              "node_id": ...,
              "E_TRP": ...,
              "collapse_stable": bool,
              "orientation": CompassOrientation,
              "state": List[float],
            }
        """
        node = self.nodes.get(node_id)
        if node is None:
            raise KeyError(f"Unknown node_id: {node_id}")

        # 1. Compute TRP mismatch energy at this node.
        e_trp = compute_e_trp(p_R, p_P, u, self.cfg.trp_config)

        # 2. Evaluate collapse stability under SOLVABLE_ECHO.
        ctx = CollapseContext(
            trp_value=max(self.cfg.trp_config.t_min, 1.0),  # toy TRP value
            has_identity_echo=(node.echo_id is not None),
        )
        collapse_stable = evaluate_collapse_stability(
            CollapseClass.SOLVABLE_ECHO,
            ctx,
            self.cfg.t_min,
        )

        # 3. Update orientation based on E_TRP & stability.
        node.orientation = self._update_orientation(
            node.orientation,
            e_trp,
            collapse_stable,
        )

        # 4. Update internal state based on orientation.
        self._update_state(node)

        # 5. Optional hook for domain-specific updates.
        if state_update_hook is not None:
            state_update_hook(node)

        # 6. Store echo if node participates in EternityEngine.
        if node.echo_id is not None:
            record = EchoRecord(
                id=node.echo_id,
                payload={
                    "node_id": node.node_id,
                    "state": list(node.state),
                    "orientation": node.orientation.name,
                    "last_E_TRP": e_trp,
                    "collapse_stable": collapse_stable,
                },
            )
            self.eternity.store_echo(record)

        return {
            "node_id": node.node_id,
            "E_TRP": e_trp,
            "collapse_stable": collapse_stable,
            "orientation": node.orientation,
            "state": list(node.state),
        }

    def step_all(
        self,
        p_R_map: Dict[str, Sequence[float]],
        p_P_map: Dict[str, Sequence[float]],
        u_map: Dict[str, Sequence[float]],
        state_update_hook: Optional[Callable[[RhizomeNode], None]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Update all nodes that have entries in p_R_map, p_P_map, u_map.

        Returns a dict mapping node_id -> result dict from step_node.
        """
        results: Dict[str, Dict[str, Any]] = {}
        for node_id in self.nodes.keys():
            if node_id not in p_R_map or node_id not in p_P_map or node_id not in u_map:
                continue
            res = self.step_node(
                node_id,
                p_R_map[node_id],
                p_P_map[node_id],
                u_map[node_id],
                state_update_hook=state_update_hook,
            )
            results[node_id] = res
        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _update_orientation(
        self,
        current: CompassOrientation,
        e_trp: float,
        collapse_stable: bool,
    ) -> CompassOrientation:
        """
        Decide new orientation based on E_TRP and collapse stability.

        Simple policy:

        - If not collapse_stable -> COLLAPSE (try to find a new basin)
        - Else if E_TRP <= stable_threshold -> STABLE
        - Else if E_TRP >= explore_threshold -> EXPLORE
        - Else -> keep current orientation
        """
        if not collapse_stable:
            return CompassOrientation.COLLAPSE

        if e_trp <= self.cfg.stable_threshold:
            return CompassOrientation.STABLE

        if e_trp >= self.cfg.explore_threshold:
            return CompassOrientation.EXPLORE

        return current

    def _update_state(self, node: RhizomeNode) -> None:
        """
        Update node.state in-place based on current orientation.

        This is intentionally simple and local:

        - STABLE: no change.
        - EXPLORE: add small Gaussian-like noise to each component.
        - COLLAPSE: move state components toward zero (a toy "lower-energy" basin).
        """
        if not node.state:
            return

        if node.orientation == CompassOrientation.STABLE:
            # No change under STABLE; system hovers near current attractor.
            return

        if node.orientation == CompassOrientation.EXPLORE:
            gain = self.cfg.exploration_gain
            node.state = [
                float(s) + gain * (random.random() * 2.0 - 1.0) for s in node.state
            ]
            return

        if node.orientation == CompassOrientation.COLLAPSE:
            gain = self.cfg.collapse_gain
            node.state = [float(s) - gain * float(s) for s in node.state]
            return
