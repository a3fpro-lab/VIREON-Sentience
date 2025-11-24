"""
MirrorGame environment.

State:
- last agent action a_{t-1}
- last opponent action o_{t-1}

Observation to agent:
    (t, a_{t-1}, o_{t-1})

Opponent policy:
    o_t = a_{t-1} with prob p_mirror else 1-a_{t-1}

Reward:
    r_t = +1 if a_t == o_t else -1

Episode length fixed at max_steps.
"""

from __future__ import annotations
import numpy as np


class MirrorGame:
    def __init__(
        self,
        p_mirror: float = 0.9,
        max_steps: int = 50,
        seed: int = 0,
    ):
        if not (0.0 <= p_mirror <= 1.0):
            raise ValueError("p_mirror must be in [0,1].")
        if max_steps < 5:
            raise ValueError("max_steps must be >= 5.")
        self.p_mirror = p_mirror
        self.max_steps = max_steps
        self.rng = np.random.default_rng(seed)

        self.reset()

    def reset(self):
        self.t = 0
        self.a_prev = int(self.rng.integers(0, 2))
        self.o_prev = int(self.rng.integers(0, 2))
        self.done = False
        return self._obs()

    def step(self, action: int):
        if self.done:
            raise RuntimeError("Episode done. Call reset().")
        if action not in (0, 1):
            raise ValueError("action must be 0 or 1.")

        # opponent mirrors previous agent action with prob p_mirror
        if self.rng.random() < self.p_mirror:
            o_t = self.a_prev
        else:
            o_t = 1 - self.a_prev

        # reward for matching opponent
        r = 1.0 if action == o_t else -1.0

        # advance
        self.t += 1
        self.a_prev = action
        self.o_prev = o_t

        if self.t >= self.max_steps:
            self.done = True

        return self._obs(), float(r), self.done, {"t": self.t, "o_t": o_t}

    def _obs(self):
        return np.array([self.t, self.a_prev, self.o_prev], dtype=np.float32)
