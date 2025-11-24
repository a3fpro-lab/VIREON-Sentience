"""
MirrorGame bench.

Two-player iterative game:
- Agent chooses action a_t ∈ {0,1}.
- Opponent mirrors agent's previous action with probability p_mirror,
  otherwise flips it.
- Reward to agent: +1 if a_t matches opponent action o_t, else -1.

To do well, agent must infer the opponent is a "mirror of self"
and stabilize behavior accordingly.
"""

from .env import MirrorGame

__all__ = ["MirrorGame"]
