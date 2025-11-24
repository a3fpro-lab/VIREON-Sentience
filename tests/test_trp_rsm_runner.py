from benches.runners.train_trp_rsm import train_trp_rsm_reinforce_bandit


def test_trp_rsm_runner_shapes():
    out = train_trp_rsm_reinforce_bandit(steps=200, seed=0)
    assert out["rewards"].shape == (200,)
    assert out["kl_policy"].shape == (200,)
    assert out["kl_mirror"].shape == (200,)
    assert out["g_self"].shape == (200,)
    assert out["pressure"].shape == (200,)
