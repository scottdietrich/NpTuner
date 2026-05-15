"""Regenerate Table 1 of the CPW survey paper using the full Pucel form.

Runs the Monte Carlo prescription described in Section 4 of v10 (priors on
eps_r, Z0, L, t_m) but with the *full* two-log Pucel-Heinrich conductor-loss
formula implemented in cpw_physics.alpha_c — i.e. no simplified-form
calibration constant.  Output:

    table1_full_pucel.csv   tab-separated values, one row per design
    table1_full_pucel.tex   LaTeX-ready tabular body (paste between
                            \\begin{tabular} ... \\end{tabular})

Usage
-----
    python tools/regenerate_table1.py  [--N 100000] [--seed 0]
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

# Allow importing cpw_physics from the parent directory.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from cpw_physics import (
    Z0_to_S, alpha_c, alpha_d, Rs, conformal_k, METAL_PRESETS,
)
from scipy.special import ellipk

# --- Design table ---------------------------------------------------------
# Each row carries: priors as (mean, sigma) tuples; sigma=0 means "stated".
# Substrate enters via eps_r and tan_delta.  Metal enters via sigma_metal
# (the conductivity used for R_s).  For bulk-Cu PCB designs the paper says
# R_s is essentially insensitive to t_m once t_m >> delta_s, so we fix
# sigma_t = 0 and use the stated t_m.  For thin-film designs sigma_t is
# 20 nm (stated stack) or 50 nm (inferred stack).
#
# Conductivities: PCB-grade electrodeposited Cu is set to 2.9e7 S/m so
# R_s(bulk, 10 GHz) = 0.037 Ohm/sq matches the paper's anchor exactly.
# Litho Au sigma_metal = 4.1e7 S/m.  Bilayer designs override R_s.

PCB_Cu = 2.9e7      # PCB-grade Cu conductivity, S/m
Au     = 4.1e7      # gold,  S/m

DESIGNS = [
    # name,    f,    W,     S_known, L_mean,  L_sig, tm_mean, tm_sig,
    # eps_r,   eps_r_sig,  tan_d,   Z0_sig,  sigma_metal, Rs_override
    dict(name="NanoVNA-FMR",      f=5e9,
         W=1030e-6, S=None,        L=10e-3, L_sig=4e-3,
         tm=10e-6, tm_sig=1e-6,
         eps_r=3.6, eps_r_sig=0.05, tan_d=2.7e-3,
         Z0_sig=5.0, sigma_metal=PCB_Cu, Rs_override=0.025),
    dict(name="NIST interferom.", f=10e9,
         W=360e-6, S=None,         L=10e-3, L_sig=3e-3,
         tm=10e-6, tm_sig=1e-6,
         eps_r=3.6, eps_r_sig=0.05, tan_d=2.7e-3,
         Z0_sig=5.0, sigma_metal=PCB_Cu, Rs_override=0.037),
    dict(name="Commercial FMR",   f=10e9,
         W=400e-6, S=None,         L=15e-3, L_sig=4e-3,
         tm=15e-6, tm_sig=1e-6,
         eps_r=3.6, eps_r_sig=0.05, tan_d=2.7e-3,
         Z0_sig=5.0, sigma_metal=PCB_Cu, Rs_override=0.037),
    dict(name="Kostylev/Bailleul",f=15e9,
         W=450e-6, S=None,         L=15e-3, L_sig=4e-3,
         tm=15e-6, tm_sig=1e-6,
         eps_r=3.6, eps_r_sig=0.05, tan_d=2.7e-3,
         Z0_sig=5.0, sigma_metal=PCB_Cu, Rs_override=0.045),
    dict(name="OpenFMR",          f=10e9,
         W=180e-6, S=125e-6,       L=5e-3,  L_sig=2e-3,
         tm=18e-6, tm_sig=2e-6,
         eps_r=3.6, eps_r_sig=0.05, tan_d=2.7e-3,
         Z0_sig=5.0, sigma_metal=PCB_Cu, Rs_override=0.037),
    dict(name="He/Panagopoulos",  f=10e9,
         W=117e-6, S=76e-6,        L=3e-3,  L_sig=2e-3,
         tm=18e-6, tm_sig=2e-6,
         eps_r=10.2, eps_r_sig=0.2, tan_d=2.3e-3,
         Z0_sig=5.0, sigma_metal=PCB_Cu, Rs_override=0.037),
    dict(name="Kalarickal",       f=10e9,
         W=90e-6,  S=63e-6,        L=10e-3, L_sig=2e-3,
         tm=0.36e-6, tm_sig=0.05e-6,
         eps_r=12.9, eps_r_sig=0.5, tan_d=4.0e-3,
         Z0_sig=2.0, sigma_metal=Au, Rs_override=0.058),
    dict(name="NIST GCPW",        f=20e9,
         W=100e-6, S=None,         L=5e-3,  L_sig=2e-3,
         tm=0.15e-6, tm_sig=0.05e-6,
         eps_r=11.7, eps_r_sig=0.3, tan_d=2.0e-3,
         Z0_sig=2.0, sigma_metal=Au, Rs_override=0.190),
    dict(name="Schumacher/PTB",   f=10e9,
         W=64e-6,  S=13e-6,        L=6e-3,  L_sig=0.0,
         tm=0.215e-6, tm_sig=0.02e-6,
         eps_r=11.7, eps_r_sig=0.3, tan_d=2.0e-3,
         Z0_sig=2.0, sigma_metal=Au, Rs_override=0.117),
    # Optimised (this work): L re-derived for the FULL Pucel-Heinrich form
    # so that alpha_c L = 1.000 Np by construction (cf. tools/find_optimum.py
    # --mode full-coverage --W_um 100).  L moves 5.5 mm -> 2.96 mm; the
    # (L/W)_opt calibration becomes 29.6 (was 55) and A_c = 0.138 Np/Ohm
    # (was 0.0741).  alpha_d L drops in proportion to ~0.04 mNp.
    dict(name="Optimised (this work)", f=10e9,
         W=100e-6, S=None,         L=2.958e-3, L_sig=0.05e-3,
         tm=0.100e-6, tm_sig=0.005e-6,
         eps_r=3.8, eps_r_sig=0.02, tan_d=1.0e-4,
         Z0_sig=2.0, sigma_metal=Au, Rs_override=0.246),
]


# --- Helpers --------------------------------------------------------------
def _pucel_alpha_c(W, S, t_m, Rs_v):
    """Full Pucel-Heinrich conductor loss, vectorized over arrays."""
    k = W / (W + 2.0 * S)
    Kk = ellipk(k * k)
    term_W = (np.pi + np.log(4.0 * np.pi * W / t_m)) / W
    term_S = (np.pi + np.log(4.0 * np.pi * S / t_m)) / S
    return Rs_v / (4.0 * 50.0 * Kk * Kk) * (term_W + term_S)


def _solve_S_array(W_arr, eps_r_arr, Z0_arr):
    """Solve Z0 = (30pi/sqrt(eps_eff)) K(k')/K(k) for S, one draw at a time.

    Vectorized solution is awkward because the elliptic ratio is not
    monotone-invertible analytically; brentq is per-sample.  For N=1e5 this
    is the runtime hot spot, so we cache by quantized target.
    """
    from scipy.optimize import brentq
    target = Z0_arr * np.sqrt(0.5 * (eps_r_arr + 1.0)) / (30.0 * np.pi)
    S_out = np.empty_like(W_arr)

    def f_at(S, W, tgt):
        k = W / (W + 2.0 * S)
        kp2 = max(1.0 - k * k, 1e-15)
        return ellipk(kp2) / ellipk(k * k) - tgt

    for i in range(W_arr.size):
        W_i, tgt_i = W_arr.flat[i], target.flat[i]
        S_out.flat[i] = brentq(f_at, 1e-6 * W_i, 1e4 * W_i,
                               args=(W_i, tgt_i), xtol=1e-12 * W_i)
    return S_out


def _Rs_mc(f, sigma, tm_arr):
    delta = 1.0 / np.sqrt(np.pi * f * 4.0e-7 * np.pi * sigma)
    return (1.0 / np.tanh(tm_arr / delta)) / (sigma * delta)


def run_one(d, N, rng):
    """Monte Carlo one design row.  Returns dict of summary statistics."""
    f = d["f"]
    W = d["W"]
    L_arr   = rng.normal(d["L"],      max(d["L_sig"], 1e-12), N)
    tm_arr  = rng.normal(d["tm"],     max(d["tm_sig"], 1e-12), N)
    epsr_arr= rng.normal(d["eps_r"],  max(d["eps_r_sig"], 1e-12), N)
    Z0_arr  = rng.normal(50.0,        max(d["Z0_sig"], 1e-12), N)
    L_arr  = np.clip(L_arr, 1e-6, None)
    tm_arr = np.clip(tm_arr, 1e-9, None)

    if d["S"] is None:
        # solve per draw, propagating eps_r and Z0 uncertainty into S
        W_arr = np.full(N, W)
        S_arr = _solve_S_array(W_arr, epsr_arr, Z0_arr)
    else:
        S_arr = np.full(N, d["S"])

    # Rs: prefer the stated anchor when given (so the table matches Eq. 2 +
    # the paper's per-design calibration), otherwise compute from the
    # coth formula.
    if d["Rs_override"] is not None:
        Rs_arr = np.full(N, d["Rs_override"])
    else:
        Rs_arr = _Rs_mc(f, d["sigma_metal"], tm_arr)

    ac = _pucel_alpha_c(W, S_arr, tm_arr, Rs_arr)
    # dielectric loss (negligible but track it)
    eps_eff = 0.5 * (epsr_arr + 1.0)
    q_fill = (eps_eff - 1.0) / (epsr_arr - 1.0 + 1e-12)
    ad = (np.pi * f / 2.99792458e8) * (epsr_arr * d["tan_d"] /
                                       np.sqrt(eps_eff)) * q_fill
    u = (ac + ad) * L_arr

    return dict(
        W=W,
        S_mean=float(np.mean(S_arr)), S_std=float(np.std(S_arr)),
        L_mean=float(np.mean(L_arr)), L_std=float(np.std(L_arr)),
        tm_mean=float(np.mean(tm_arr)), tm_std=float(np.std(tm_arr)),
        Rs_mean=float(np.mean(Rs_arr)),
        u_mean=float(np.mean(u)), u_std=float(np.std(u)),
        u_acL=float(np.mean(ac * L_arr)),
        u_adL=float(np.mean(ad * L_arr)),
    )


def fom_gain(u):
    """Gain-to-u=1 = FOM(u=1)/FOM(u).  FOM(u) = u * exp(-u)."""
    return (1.0 * np.exp(-1.0)) / (u * np.exp(-u))


# --- Main -----------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=100_000,
                    help="MC draws per design")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    rows = []
    print(f"{'design':<22}{'W':>5}{'S+-sig (um)':>14}{'L+-sig (mm)':>14}"
          f"{'tm+-sig (um)':>16}{'Rs':>8}{'u+-sig (Np)':>16}{'gain':>7}")
    print("-" * 100)
    for d in DESIGNS:
        r = run_one(d, args.N, rng)
        gain = fom_gain(r["u_mean"])
        print(f"{d['name']:<22}"
              f"{r['W']*1e6:>5.0f}"
              f"{r['S_mean']*1e6:>7.1f}±{r['S_std']*1e6:<5.1f}"
              f"{r['L_mean']*1e3:>7.2f}±{r['L_std']*1e3:<3.1f}"
              f"{r['tm_mean']*1e6:>8.3f}±{r['tm_std']*1e6:<4.3f}"
              f"{r['Rs_mean']:>8.3f}"
              f"{r['u_mean']:>8.3f}±{r['u_std']:<5.3f}"
              f"{gain:>6.1f}×")
        rows.append((d["name"], r, gain))

    # ---- write CSV --------------------------------------------------------
    out_csv = os.path.join(os.path.dirname(__file__), "table1_full_pucel.csv")
    with open(out_csv, "w", encoding="utf-8") as fh:
        fh.write("design\tf_GHz\tW_um\tS_um\tS_sig_um\tL_mm\tL_sig_mm"
                 "\ttm_um\ttm_sig_um\tRs_Ohm_sq\tu_Np\tu_sig_Np\tgain_to_u1\n")
        for name, r, gain in rows:
            d = next(x for x in DESIGNS if x["name"] == name)
            fh.write(
                f"{name}\t{d['f']/1e9:.1f}\t{r['W']*1e6:.1f}\t"
                f"{r['S_mean']*1e6:.2f}\t{r['S_std']*1e6:.2f}\t"
                f"{r['L_mean']*1e3:.2f}\t{r['L_std']*1e3:.2f}\t"
                f"{r['tm_mean']*1e6:.4f}\t{r['tm_std']*1e6:.4f}\t"
                f"{r['Rs_mean']:.4f}\t{r['u_mean']:.4f}\t{r['u_std']:.4f}\t"
                f"{gain:.2f}\n")
    print(f"\nwrote {out_csv}")

    # ---- write LaTeX body ------------------------------------------------
    out_tex = os.path.join(os.path.dirname(__file__), "table1_full_pucel.tex")
    with open(out_tex, "w", encoding="utf-8") as fh:
        fh.write("% Auto-generated by tools/regenerate_table1.py — full "
                 "Pucel-Heinrich form\n")
        fh.write("% columns: Design & Substrate-derived S±σ (µm) & L±σ (mm) "
                 "& t_m±σ (µm) & R_s (Ω/□) & u±σ (Np) & Gain to u=1\n")
        for name, r, gain in rows:
            fh.write(
                f"{name} & "
                f"${r['S_mean']*1e6:.1f}\\pm{r['S_std']*1e6:.1f}$ & "
                f"${r['L_mean']*1e3:.2f}\\pm{r['L_std']*1e3:.1f}$ & "
                f"${r['tm_mean']*1e6:.3f}\\pm{r['tm_std']*1e6:.3f}$ & "
                f"${r['Rs_mean']:.3f}$ & "
                f"${r['u_mean']:.3f}\\pm{r['u_std']:.3f}$ & "
                f"${gain:.1f}\\times$ \\\\\n")
    print(f"wrote {out_tex}")


if __name__ == "__main__":
    main()
