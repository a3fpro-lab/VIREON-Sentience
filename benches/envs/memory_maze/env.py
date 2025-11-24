"""
MemoryMaze environment.

Gridworld with two goals A/B. A random hint at reset tells which goal is correct.
Observation:
    (x, y, hint_visible, hint_value)
- hint_visible=1 only at t=0, else 0
- hint_value ∈ {0,1} if visible, else 0

Rewards:
- +1 if reach correct goal
- -1 if reach wrong goal
- small step penalty each move
Episode ends on reaching any goal or max_steps.
"""

from __future__ import annotations
import numpy as np


class MemoryMaze:
    ACTIONS = {
        0: (0, 1),   # up
        1: (0, -1),  # down
        2: (-1, 0),  # left
        3: (1, 0),   # right
        4: (0, 0),   # stay
    }

    def __init__(
        self,
        size: int = 5,
        max_steps: int = 50,
        step_penalty: float = 0.01,
        seed: int = 0,
    ):
        if size < 3:
            raise ValueError("size must be >= 3.")
        if max_steps < 5:
            raise ValueError("max_steps must be >= 5.")
        self.size = size
        self.max_steps = max_steps
        self.step_penalty = step_penalty
        self.rng = np.random.default_rng(seed)

        # fixed start + two fixed goals
        self.start = (0, 0)
        self.goal_A = (size - 1, 0)
        self.goal_B = (0, size - 1)

        self.reset()

    def reset(self):
        self.t = 0
        self.pos = list(self.start)
        self.hint = int(self.rng.integers(0, 2))  # 0 -> A, 1 -> B
        self.done = False
        return self._obs(hint_visible=True)

    def step(self, action: int):
        if self.done:
            raise RuntimeError("Episode done. Call reset().")
        if action not in self.ACTIONS:
            raise ValueError("Invalid action.")

        dx, dy = self.ACTIONS[action]
        nx = int(np.clip(self.pos[0] + dx, 0, self.size - 1))
        ny = int(np.clip(self.pos[1] + dy, 0, self.size - 1))
        self.pos = [nx, ny]

        self.t += 1

        reward = -self.step_penalty
        info = {"t": self.t, "hint": self.hint}

        # check terminal
        if (nx, ny) == self.goal_A:
            reward = 1.0 if self.hint == 0 else -1.0
            self.done = True
        elif (nx, ny) == self.goal_B:
            reward = 1.0 if self.hint == 1 else -1.0
            self.done = True
        elif self.t >= self.max_steps:
            self.done = True

        obs = self._obs(hint_visible=False)
        return obs, float(reward), self.done, info

    def _obs(self, hint_visible: bool):
        hv = 1.0 if hint_visible else 0.0
        hval = float(self.hint) if hint_visible else 0.0
        return np.array([self.pos[0], self.pos[1], hv, hval], dtype=np.float32)
