"""
MemoryMaze bench.

A tiny POMDP-style maze:
- At t=0 you see a hint (0 or 1).
- After t=0 the hint disappears from observation.
- There are two goals; only the hinted goal gives +1.
- Wrong goal gives -1.

Agents must store the hint internally to solve.
"""
from .env import MemoryMaze

__all__ = ["MemoryMaze"]
