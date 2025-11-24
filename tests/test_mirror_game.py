from benches.envs.mirror_game import MirrorGame


def test_mirror_game_runs():
    env = MirrorGame(p_mirror=0.9, max_steps=10, seed=0)
    obs = env.reset()
    assert obs.shape == (3,)

    obs2, r, done, info = env.step(1)
    assert obs2.shape == (3,)
    assert isinstance(r, float)
