import numpy as np
import math

from vireon.rsm.mirror import MirrorModel
from vireon.rsm.self_forecast import SelfForecaster, kl_gaussian_1d
from vireon.rsm.collapse_guard import CollapseGuard


def test_mirror_updates_reduce_kl_on_repeat():
    mm = MirrorModel(n_actions=3, n_buckets=1, ema_beta=0.5)
    obs = np.array([0.0])

    pi = np.array([0.7, 0.2, 0.1])
    kl1 = mm.update(obs, pi)
    kl2 = mm.update(obs, pi)
    assert kl2 <= kl1 + 1e-9

def test_self_forecaster_gap_nonnegative():
    sf = SelfForecaster(beta=0.5, window=10)
    for e in [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]:
        sf.update(e)
    g = sf.self_surprise_gap()
    assert g >= 0.0

def test_kl_gaussian_zero_when_equal():
    k = kl_gaussian_1d(0.0, 1.0, 0.0, 1.0)
    assert math.isclose(k, 0.0, abs_tol=1e-12)

def test_collapse_guard_pressure():
    cg = CollapseGuard(w_policy=2.0, w_mirror=3.0, w_self=4.0)
    p = cg.pressure(0.1, 0.2, 0.3)
    assert math.isclose(p, 1.0 + 2*0.1 + 3*0.2 + 4*0.3)
