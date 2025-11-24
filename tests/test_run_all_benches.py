"""
Smoke test for the one-button runner.

We don't run full budgets in CI.
We just import and call one small slice.
"""

from benches.run_all_benches import log_bandits


def test_log_bandits_small():
    b, t, r = log_bandits(seed=0, steps=50)
    assert isinstance(b, float)
    assert isinstance(t, float)
    assert isinstance(r, float)p
