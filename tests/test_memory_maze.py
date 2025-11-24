import numpy as np
from benches.envs.memory_maze import MemoryMaze


def test_memory_maze_reset_and_step():
    env = MemoryMaze(size=5, max_steps=10, seed=0)
    obs = env.reset()
    assert obs.shape == (4,)
    assert obs[2] == 1.0  # hint visible at reset

    obs2, r, done, info = env.step(4)  # stay
    assert obs2.shape == (4,)
    assert obs2[2] == 0.0  # hint hidden after reset
