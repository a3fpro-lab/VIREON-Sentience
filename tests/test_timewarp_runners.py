from benches.runners.train_timewarp_baselines import train_reinforce_timewarp
from benches.runners.train_timewarp_trp import train_trp_reinforce_timewarp
from benches.runners.train_timewarp_trp_rsm import train_trp_rsm_reinforce_timewarp


def test_timewarp_baseline_shapes():
    out = train_reinforce_timewarp(episodes=20, seed=0)
    assert out["returns"].shape == (20,)
    assert out["policy_kl"].shape == (20,)

def test_timewarp_trp_shapes():
    out = train_trp_reinforce_timewarp(episodes=20, seed=0)
    assert out["returns"].shape == (20,)
    assert out["policy_kl"].shape == (20,)
    assert out["dt_eff"].shape == (20,)

def test_timewarp_trp_rsm_shapes():
    out = train_trp_rsm_reinforce_timewarp(episodes=20, seed=0)
    assert out["returns"].shape == (20,)
    assert out["policy_kl"].shape == (20,)
    assert out["kl_mirror"].shape == (20,)
    assert out["g_self"].shape == (20,)
    assert out["pressure"].shape == (20,)
    assert out["dt_eff"].shape == (20,)
