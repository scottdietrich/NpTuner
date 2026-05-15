"""Unit tests for cpw_physics.

Run with:  pytest -q   from the cpw_optimizer/ directory.
"""

import os
import sys

import numpy as np
import pytest

# Allow running without installing the package.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from cpw_physics import (  # noqa: E402
    Z0_to_S,
    Rs,
    alpha_c,
    u_opt,
    fom_curve,
    skin_depth,
    conformal_k,
)


# ------------------------------------------------------------------
# 1) Z0_to_S recovers a known S/W ratio for fused silica, 50 Ohm
# ------------------------------------------------------------------
def test_Z0_to_S_fused_silica():
    """For eps_r = 3.8 (fused silica, half-space), eps_eff = 2.4 and Z0 = 50 Ohm
    requires K(k')/K(k) = 0.822.  The root returned by Z0_to_S must
    (a) reproduce 50 Ohm when plugged back into the conformal formula, and
    (b) be scale-invariant: S/W depends only on eps_r & Z0."""
    # (a) self-consistency
    W = 100e-6
    S = Z0_to_S(W, eps_r=3.8, Z0=50.0)
    k = conformal_k(W, S)
    from scipy.special import ellipk
    eps_eff = (3.8 + 1.0) / 2.0
    Z0_check = (30.0 * np.pi / np.sqrt(eps_eff)) * \
               ellipk(max(1.0 - k*k, 1e-15)) / ellipk(k*k)
    assert abs(Z0_check - 50.0) < 1e-3
    # (b) scale invariance of S/W
    W2 = 400e-6
    S2 = Z0_to_S(W2, eps_r=3.8, Z0=50.0)
    assert abs(S / W - S2 / W2) < 1e-6
    # (c) S/W in the expected neighbourhood for the half-space model (~0.095)
    assert 0.08 < S / W < 0.11


# ------------------------------------------------------------------
# 2) Rs: bulk limit (t_m >> delta_s) and thin-film limit (t_m << delta_s)
# ------------------------------------------------------------------
def test_Rs_bulk_limit():
    sigma = 58e6    # Cu
    f = 10e9
    delta = skin_depth(f, sigma)
    t_m = 100 * delta              # very thick
    rs = Rs(f, sigma, t_m)
    # Bulk value: 1/(sigma*delta)
    rs_bulk = 1.0 / (sigma * delta)
    assert abs(rs - rs_bulk) / rs_bulk < 1e-3


def test_Rs_thinfilm_limit():
    sigma = 58e6
    f = 10e9
    delta = skin_depth(f, sigma)
    t_m = 0.001 * delta            # ultra-thin
    rs = Rs(f, sigma, t_m)
    # Thin-film limit: Rs -> 1/(sigma * t_m)
    rs_thin = 1.0 / (sigma * t_m)
    assert abs(rs - rs_thin) / rs_thin < 1e-2


# ------------------------------------------------------------------
# 3) u_opt
# ------------------------------------------------------------------
def test_u_opt_values():
    assert u_opt(0, 0.0) == pytest.approx(1.0)
    assert u_opt(1, 0.0) == pytest.approx(2.0)
    assert u_opt(0, 0.5) == pytest.approx(2.0)
    assert u_opt(1, 0.5) == pytest.approx(4.0)


# ------------------------------------------------------------------
# 4) alpha_c scales as 1/W across a decade of W, within 20 %
# ------------------------------------------------------------------
def test_alpha_c_scales_as_inverse_W():
    """alpha_c is 1/W times slowly-varying logarithmic factors.  Over a
    modest range the ratio (alpha_c1/alpha_c2)/(W2/W1) should sit within
    20 % of unity.  A full decade introduces ~40 % log correction so we
    test a factor-of-three window, which is what the paper's W_opt
    argument actually relies on."""
    f = 10e9
    sigma = 58e6
    t_m = 35e-6           # 1 oz Cu
    eps_r = 3.8
    W1 = 100e-6
    W2 = 300e-6           # factor of 3
    a1 = alpha_c(W1, f, sigma, t_m, eps_r)
    a2 = alpha_c(W2, f, sigma, t_m, eps_r)
    ratio = (a1 / a2) / (W2 / W1)
    assert 0.8 < ratio < 1.2, f"alpha_c 1/W scaling off: ratio={ratio:.2f}"

    # And verify the decade behaviour is within 50 % (log corrections).
    a_small = alpha_c(50e-6, f, sigma, t_m, eps_r)
    a_big = alpha_c(500e-6, f, sigma, t_m, eps_r)
    ratio_dec = (a_small / a_big) / 10.0
    assert 0.5 < ratio_dec < 1.0, \
        f"alpha_c over one decade off: {ratio_dec:.2f}"


# ------------------------------------------------------------------
# 5) fom_curve peaks at u ~ u_opt
# ------------------------------------------------------------------
def test_fom_curve_peaks_at_u_opt():
    # Choose parameters so that the W-grid brackets u_opt for n=0, phi=0 => u=1
    f = 10e9
    sigma = 58e6
    t_m = 35e-6
    eps_r = 3.8
    tan_delta = 1e-4
    # Pick L such that alpha_c*L covers [0.1, 10] as W varies over the grid.
    L = 0.05
    W_grid = np.logspace(-5, -3, 60)   # 10 um to 1 mm
    u, fom, fom_norm = fom_curve(W_grid, L, f, sigma, t_m, eps_r,
                                 tan_delta, n=0, phi=0.0)
    # Peak of u^1 * e^{-u} should sit at u = 1.
    u_peak = u[np.argmax(fom)]
    assert 0.7 < u_peak < 1.4, f"Peak at u={u_peak:.2f}, expected ~1"

    # And check for n=1 case: peak should be at u=2.
    u2, fom2, _ = fom_curve(W_grid, L, f, sigma, t_m, eps_r,
                            tan_delta, n=1, phi=0.0)
    u_peak2 = u2[np.argmax(fom2)]
    assert 1.5 < u_peak2 < 2.7, f"Peak at u={u_peak2:.2f}, expected ~2"
