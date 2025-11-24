"""
TimeWarpGrid environment.

Gridworld with a moving hazard wave:
- If you step on a hazard cell, you get -1.
- Hazard wave shifts every step.
- You may choose to delay τ steps (do-nothing) before moving.
- Delay lets you "wait out" a hazard sweep.

Observation:
    (x, y, hazard_phase)

Actions:
    move ∈ {0..4} (up/down/left/right/stay)
    delay τ ∈ {0..3}

We encode combined action as:
    a = move + 5*delay   where move in 0..4, delay in 0..3  => 20 actions.

Reward:
    -0.01 per step
    -1.0 if land on hazard
    +5.0 if reach goal

Episode ends at goal or max_steps.
"""

from __future__ import annotations
import numpy as np


class TimeWarpGrid:
    MOVES = {
        0: (0, 1),   # up
        1: (0, -1),  # down
        2: (-1, 0),  # left
        3: (1, 0),   # right
        4: (0, 0),   # stay
    }

    def __init__(
        self,
        size: int = 7,
        max_steps: int = 80,
        hazard_period: int = 4,
        step_penalty: float = 0.01,
        hazard_penalty: float = 1.0,
        goal_reward: float = 5.0,
        seed: int = 0,
    ):
        if size < 5:
            raise ValueError("size must be >= 5.")
        if hazard_period < 2:
            raise ValueError("hazard_period must be >= 2.")

        self.size = size
        self.max_steps = max_steps
        self.hazard_period = hazard_period
        self.step_penalty = step_penalty
        self.hazard_penalty = hazard_penalty
        self.goal_reward = goal_reward
        self.rng = np.random.default_rng(seed)

        self.start = (0, 0)
        self.goal = (size - 1, size - 1)

        self.reset()

    def reset(self):
        self.t = 0
        self.pos = list(self.start)
        self.phase = int(self.rng.integers(0, self.hazard_period))
        self.done = False
        return self._obs()

    def step(self, action: int):
        if self.done:
            raise RuntimeError("Episode done. Call reset().")
        if not (0 <= action < 20):
            raise ValueError("action must be in [0,19].")

        move = action % 5
        delay = action // 5  # 0..3

        reward = 0.0
        info = {"t": self.t, "phase": self.phase, "delay": delay, "move": move}

        # delay τ steps (do nothing but advance hazard)
        for _ in range(delay):
            if self.done:
                break
            reward -= self.step_penalty
            self._advance_phase()
            self.t += 1
            if self.t >= self.max_steps:
                self.done = True

        if not self.done:
            # apply move
            dx, dy = self.MOVES[move]
            nx = int(np.clip(self.pos[0] + dx, 0, self.size - 1))
            ny = int(np.clip(self.pos[1] + dy, 0, self.size - 1))
            self.pos = [nx, ny]

            reward -= self.step_penalty

            # hazard penalty if land on hazard cell
            if self._is_hazard(nx, ny):
                reward -= self.hazard_penalty

            # goal
            if (nx, ny) == self.goal:
                reward += self.goal_reward
                self.done = True

            self._advance_phase()
            self.t += 1
            if self.t >= self.max_steps:
                self.done = True

        return self._obs(), float(reward), self.done, info

    def _advance_phase(self):
        self.phase = (self.phase + 1) % self.hazard_period

    def _is_hazard(self, x, y):
        # simple moving diagonal hazard wave
        # hazard when (x + y + phase) mod hazard_period == 0
        return ((x + y + self.phase) % self.hazard_period) == 0

    def _obs(self):
        return np.array([self.pos[0], self.pos[1], self.phase], dtype=np.float32)
