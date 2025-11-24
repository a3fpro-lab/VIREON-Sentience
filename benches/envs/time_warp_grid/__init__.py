"""
time_warp_grid bench.

Grid task where delaying action can be beneficial.
Used to measure M4 Agency Delay Curve.

Agent chooses:
- move action (up/down/left/right/stay)
- optional delay τ in {0,1,2,3}

Reward favors reaching goal while minimizing penalties;
delay can reduce hazard penalties if timed correctly.
"""
from .env import TimeWarpGrid

__all__ = ["TimeWarpGrid"]
