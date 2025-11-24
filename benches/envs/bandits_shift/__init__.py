"""
Non-stationary multi-armed bandit bench.

The reward distribution shifts at fixed intervals.
Used to test TRP time-dilation stability + adaptation.
"""
from .env import ShiftingBandit

__all__ = ["ShiftingBandit"]
