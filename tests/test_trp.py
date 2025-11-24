import math
import pytest

from vireon.trp.warp import dt_eff_from_divergence
from vireon.trp.leash import kl_leash_ok, kl_leash_scale
from vireon.trp.alpha import AlphaSchedule, F_alpha_k, C_k_prime_weight
from vireon.trp.pacing import TRPPacer


def test_dt_eff_basic():
    dt = 1.0
    divergence = 2.0
    alpha = 0.5
    out = dt_eff_from_divergence(dt, divergence, alpha)
    assert math.isclose(out, math.exp(-1.0), rel_tol=1e-9)

def test_kl_leash():
    assert kl_leash_ok(0.01, 0.02)
    assert not kl_leash_ok(0.03, 0.02)

def test_kl_scale():
    assert kl_leash_scale(0.01, 0.02) == 1.0
    s = kl_leash_scale(0.04, 0.02, power=1.0)
    assert math.isclose(s, 2.0)

def test_alpha_schedule_update_monotone():
    sched = AlphaSchedule(alpha0=1.0, decay=1.0, gain=0.1)
    a1 = sched.update(1.0, 0.0)
    a2 = sched.update(1.0, 1.0)
    assert a2 > a1

def test_F_alpha_k():
    val = F_alpha_k(k=1, mean_spacing=1.0, T=math.e**2)
    # alpha = (1/1)*(2π/log T)=2π/2=π; F=e^{-π}
    assert math.isclose(val, math.exp(-math.pi), rel_tol=1e-9)

def test_C_k_prime_weight():
    assert C_k_prime_weight(1) == 1.0
    assert C_k_prime_weight(3) > 1.0  # (3-1)/(3-2)=2

def test_trp_pacer_step():
    pacer = TRPPacer(dt=1.0, kappa=0.02, leash_power=1.0, dt_min=1e-9)
    dt_eff, a_next, div = pacer.step(kl_val=0.04, alpha_t=1.0)
    assert div == 2.0
    assert dt_eff <= 1.0
    assert a_next > 0
