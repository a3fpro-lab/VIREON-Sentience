from benches.envs.time_warp_grid import TimeWarpGrid


def test_time_warp_grid_runs():
    env = TimeWarpGrid(size=7, max_steps=10, seed=0)
    obs = env.reset()
    assert obs.shape == (3,)

    obs2, r, done, info = env.step(0)
    assert obs2.shape == (3,)
    assert isinstance(r, float)
    assert "delay" in info
