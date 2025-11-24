"""
RSM = Recursive Self-Modeling modules.

Pieces:
- MirrorModel: learns a predictive model of the agent’s own policy.
- SelfForecaster: forecasts upcoming error/loss spikes.
- CollapseGuard: combines self-signals into a stability pressure.
"""

from .mirror import MirrorModel
from .self_forecast import SelfForecaster, kl_gaussian_1d
from .collapse_guard import CollapseGuard

__all__ = ["MirrorModel", "SelfForecaster", "kl_gaussian_1d", "CollapseGuard"]
