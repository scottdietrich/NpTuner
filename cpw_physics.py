"""Pure physics routines for coplanar waveguide (CPW) sensitivity optimization.

All functions are SI-unit based unless otherwise stated in the docstring.
References to equations follow the companion paper on CPW design optimization
for broadband magnetic resonance spectroscopy.
"""

from __future__ import annotations

import numpy as np
from scipy.special import ellipk
from scipy.optimize import brentq

# --- Physical constants (SI) ---------------------------------------------
MU_0 = 4.0e-7 * np.pi          # vacuum permeability [H/m]
C_LIGHT = 2.99792458e8         # speed of light [m/s]


# --- Substrate and metal presets ----------------------------------------
# Values follow Dietrich & Solodovnyk, "CPW Transmission-Line Survey" v10,
# Table 2 (Substrate Dielectric Loss Reference) at 10 GHz.
SUBSTRATE_PRESETS = {
    "Fused silica":      {"eps_r": 3.8,  "tan_delta": 1.0e-4},
    "Rogers RO4003C":    {"eps_r": 3.6,  "tan_delta": 2.7e-3},
    "FR4":               {"eps_r": 4.4,  "tan_delta": 2.0e-2},
    "Sapphire (Al2O3)":  {"eps_r": 9.4,  "tan_delta": 1.0e-4},
    "RT/Duroid 6010":    {"eps_r": 10.2, "tan_delta": 2.3e-3},
    "HR-Si":             {"eps_r": 11.7, "tan_delta": 2.0e-3},
    "HR-GaAs":           {"eps_r": 12.9, "tan_delta": 4.0e-3},
}

# Conductivities in S/m.  "Cu (PCB-grade)" reflects the ~50 % reduction in
# σ typical of electrodeposited copper laminates (paper Sec. 5: PCB-grade
# Cu has σ ≈ 0.5 σ_ideal, giving R_s a √2 boost over drawn-wire Cu).
METAL_PRESETS = {
    "Au":             41.0e6,
    "Cu":             58.0e6,
    "Cu (PCB-grade)": 29.0e6,
    "Al":             37.7e6,
}


# ------------------------------------------------------------------------
# Geometry / conformal mapping
# ------------------------------------------------------------------------
def conformal_k(W: float, S: float) -> float:
    """Elliptic modulus for a CPW gap.

    k = W / (W + 2S)

    Parameters
    ----------
    W : float
        Centre-conductor width [m].
    S : float
        Slot (gap) width on each side [m].
    """
    return W / (W + 2.0 * S)


def _K_ratio(k: float) -> float:
    """Return K(k)/K(k') for 0 < k < 1."""
    k = float(k)
    kp = np.sqrt(max(1.0 - k * k, 1e-15))
    return ellipk(k * k) / ellipk(kp * kp)   # scipy uses m = k^2


def eps_eff(W: float, S: float, eps_r: float) -> float:
    """Effective dielectric constant for a CPW on a semi-infinite substrate.

    For a CPW sitting on a half-space substrate the standard conformal-mapping
    result is
        eps_eff = (eps_r + 1) / 2
    independent of geometry.  W, S are accepted for API symmetry.
    """
    return 0.5 * (eps_r + 1.0)


def Z0_to_S(W: float, eps_r: float, Z0: float = 50.0) -> float:
    """Solve for the CPW slot width S given centre conductor W, substrate and Z0.

    Uses the conformal-mapping relation
        Z0 = (30 pi / sqrt(eps_eff)) * K(k') / K(k)
    with k = W/(W+2S) and eps_eff = (eps_r+1)/2.
    Solved numerically with brentq over S in [1e-4*W, 1e3*W].
    """
    eps_e = eps_eff(W, 0.0, eps_r)
    target = Z0 * np.sqrt(eps_e) / (30.0 * np.pi)   # = K(k')/K(k)

    def f(S):
        k = conformal_k(W, S)
        # K(k')/K(k) - target.  Use m = k^2 convention of scipy.
        kp2 = max(1.0 - k * k, 1e-15)
        return ellipk(kp2) / ellipk(k * k) - target

    # Bracket: as S -> 0 the ratio -> 0 (k->1),
    #         as S -> inf it -> +inf (k->0). So root exists for any target>0.
    return brentq(f, 1e-6 * W, 1e4 * W, xtol=1e-12 * W)


# ------------------------------------------------------------------------
# Losses
# ------------------------------------------------------------------------
def skin_depth(f: float, sigma: float) -> float:
    """Classical skin depth delta_s = 1/sqrt(pi f mu_0 sigma) [m]."""
    return 1.0 / np.sqrt(np.pi * f * MU_0 * sigma)


def Rs(f: float, sigma: float, t_m: float) -> float:
    """Surface resistance of a finite-thickness conductor [Ohm/sq].

    Uses the coth interpolation between thin-film and bulk limits:

        Rs = (1 / (sigma * delta_s)) * coth(t_m / delta_s)

    where delta_s = 1/sqrt(pi f mu_0 sigma).
    """
    delta = skin_depth(f, sigma)
    x = t_m / delta
    # coth(x) = cosh/sinh; for very large x this -> 1 (bulk),
    # for small x -> 1/x so Rs -> 1/(sigma*t_m) (thin-film DC sheet R).
    coth = 1.0 / np.tanh(x)
    return coth / (sigma * delta)


def alpha_c(W: float, f: float, sigma: float, t_m: float,
            eps_r: float, Z0: float = 50.0) -> float:
    """Conductor loss per unit length [Np/m] from the Pucel formula.

    alpha_c = Rs / (4 Z0 [K(k)]^2) *
              [ (pi + ln(4 pi W / t_m)) / W
              + (pi + ln(4 pi S / t_m)) / S ]

    S is obtained from Z0_to_S for the given W and substrate.
    """
    S = Z0_to_S(W, eps_r, Z0)
    k = conformal_k(W, S)
    Kk = ellipk(k * k)
    rs = Rs(f, sigma, t_m)

    term_W = (np.pi + np.log(4.0 * np.pi * W / t_m)) / W
    term_S = (np.pi + np.log(4.0 * np.pi * S / t_m)) / S
    return rs / (4.0 * Z0 * Kk * Kk) * (term_W + term_S)


def alpha_d(f: float, eps_r: float, tan_delta: float) -> float:
    """Dielectric loss per unit length [Np/m].

    alpha_d = (pi f sqrt(eps_eff) / c) * q * tan_delta
    with filling factor q = eps_r / (eps_r + 1).
    """
    eps_e = 0.5 * (eps_r + 1.0)
    q = eps_r / (eps_r + 1.0)
    return np.pi * f * np.sqrt(eps_e) / C_LIGHT * q * tan_delta


# ------------------------------------------------------------------------
# Sample filling factor (thin-sample regime)
# ------------------------------------------------------------------------
def filling_factor(W: float, t: float, eps_r: float = 1.0,
                   Z0: float = 50.0, C_eta: float = 1.0):
    """Transverse microwave-field filling factor for a thin sample.

    In the thin-sample regime (t/W < 0.1) the transverse H1 averaged over the
    sample scales linearly with thickness:

        eta ~ C_eta * t / W

    Parameters
    ----------
    W : float      centre conductor width [m]
    t : float      sample thickness       [m]
    eps_r : float  substrate permittivity (unused in leading order; kept for API)
    Z0 : float     characteristic impedance (unused; kept for API)
    C_eta : float  geometric prefactor (default 1.0)

    Returns
    -------
    eta : float
    thin_film_valid : bool   True iff t/W < 0.1
    """
    ratio = t / W
    eta = C_eta * ratio
    return eta, bool(ratio < 0.1)


# ------------------------------------------------------------------------
# Optimization: meander / straight, full / fixed footprint
# ------------------------------------------------------------------------
def _Ac(W: float, f: float, sigma: float, t_m: float,
        eps_r: float, Z0: float = 50.0) -> float:
    """Slowly-varying geometric prefactor: A_c = alpha_c * W / Rs.

    alpha_c ~ (Rs/W) * A_c(W), so A_c is nearly constant in W on a log scale
    (weak logarithmic corrections from ln(W/t_m)).
    """
    rs = Rs(f, sigma, t_m)
    ac = alpha_c(W, f, sigma, t_m, eps_r, Z0)
    return ac * W / rs


def W_opt_meander(w_s: float, l_s: float, f: float, sigma: float,
                  t_m: float, eps_r: float, Z0: float = 50.0,
                  n_iter: int = 3) -> float:
    """Optimal centre-conductor width for a meander filling a fixed footprint.

        W_opt = sqrt( A_c * Rs * w_s * l_s / 2 )

    A_c is extracted self-consistently from alpha_c at the current W estimate.
    """
    rs = Rs(f, sigma, t_m)
    # Initial guess: use W = w_s/10 to evaluate A_c.
    W = max(1e-6, w_s / 10.0)
    for _ in range(n_iter):
        Ac = _Ac(W, f, sigma, t_m, eps_r, Z0)
        W = np.sqrt(Ac * rs * w_s * l_s / 2.0)
    return W


def u_opt(n: int, phi: float) -> float:
    """Optimal conductor loss in nepers.

        u_opt = (n + 1) / (1 - phi)

    n = 0 for full-coverage sample, n = 1 for partial coverage.
    phi is the noise-mixing parameter (0 = additive, 1 = multiplicative).
    """
    if phi >= 1.0:
        return float("inf")
    return (n + 1) / (1.0 - phi)


def fom_curve(W_values, L: float, f: float, sigma: float, t_m: float,
              eps_r: float, tan_delta: float, n: int, phi: float,
              Z0: float = 50.0):
    """Figure-of-merit curve FOM(u) = u^(n+1) * exp(-u), normalized to its max.

    Parameters
    ----------
    W_values : array_like   grid of conductor widths [m]
    L : float               line length [m]
    f, sigma, t_m, eps_r, tan_delta : standard CPW parameters
    n, phi : sensitivity-regime parameters.

    Returns
    -------
    u_values   : ndarray  conductor loss u = alpha_c * L [Np] at each W
    fom_values : ndarray  u^(n+1) * exp(-u)  (un-normalized)
    fom_norm   : ndarray  FOM normalized so that max = 1
    """
    W_values = np.atleast_1d(np.asarray(W_values, dtype=float))
    u = np.array([alpha_c(W, f, sigma, t_m, eps_r, Z0) * L for W in W_values])
    fom = np.power(u, n + 1) * np.exp(-u)
    peak = np.max(fom) if np.any(fom > 0) else 1.0
    return u, fom, fom / peak


def regime_classify(W_min: float, w_s: float, W_opt: float) -> str:
    """Classify fabrication regime relative to the optimal meander width.

    Returns one of:
        "meander" : W_min <= W_opt_meander  (meander is achievable & optimal)
        "B1"      : W_opt_meander < W_min <= w_s  (straight line inside footprint)
        "B2"      : W_min > w_s              (line wider than sample — sub-optimal)
    """
    if W_min <= W_opt:
        return "meander"
    if W_min <= w_s:
        return "B1"
    return "B2"
