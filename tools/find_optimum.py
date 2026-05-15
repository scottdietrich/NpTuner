"""Find the (W, L) that put "Optimised (this work)" at u = 1.000 Np exactly,
using the full Pucel-Heinrich form (no calibration constant).

Two modes:
  --mode full-coverage   (default)   hold W fixed, solve for L such that u = 1.
  --mode fixed-footprint              find W_opt for a w_s × l_s sample
                                      meander (n=1 case, u_opt = 2 by default).

Defaults match the manuscript: fused silica, 100 nm Au, 10 GHz.

Outputs the design parameters to stdout and writes a small machine-readable
JSON the MC regenerator can pick up if you want to chain them.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from scipy.optimize import brentq

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from cpw_physics import (
    Rs, Z0_to_S, alpha_c, alpha_d, conformal_k, W_opt_meander,
    SUBSTRATE_PRESETS, METAL_PRESETS,
)


def report(label, **kw):
    print(f"--- {label} ---")
    for k, v in kw.items():
        if isinstance(v, float):
            if 1e-9 < abs(v) < 1e-3:
                print(f"  {k:<22} = {v*1e6:.4g} µm" if "S" in k or "W" in k
                      else f"  {k:<22} = {v*1e6:.4g} × 10⁻⁶ {kw.get('_unit_' + k, '')}")
            else:
                print(f"  {k:<22} = {v:.6g}")
        else:
            print(f"  {k:<22} = {v}")
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode",
                    choices=["full-coverage", "fixed-footprint"],
                    default="full-coverage")
    ap.add_argument("--W_um", type=float, default=100.0,
                    help="conductor width (full-coverage mode)")
    ap.add_argument("--w_s_um", type=float, default=3000.0,
                    help="sample width (fixed-footprint mode)")
    ap.add_argument("--l_s_um", type=float, default=3000.0,
                    help="sample length (fixed-footprint mode)")
    ap.add_argument("--f_GHz", type=float, default=10.0)
    ap.add_argument("--tm_nm", type=float, default=100.0,
                    help="metal thickness")
    ap.add_argument("--substrate", default="Fused silica",
                    choices=list(SUBSTRATE_PRESETS.keys()))
    ap.add_argument("--metal", default="Au",
                    choices=list(METAL_PRESETS.keys()))
    ap.add_argument("--u_target", type=float, default=None,
                    help="override u_opt (default 1 for full-coverage, "
                         "2 for fixed-footprint)")
    args = ap.parse_args()

    f = args.f_GHz * 1e9
    t_m = args.tm_nm * 1e-9
    sub = SUBSTRATE_PRESETS[args.substrate]
    eps_r = sub["eps_r"]
    tan_d = sub["tan_delta"]
    sigma = METAL_PRESETS[args.metal]
    rs = Rs(f, sigma, t_m)

    u_opt = (args.u_target if args.u_target is not None
             else (1.0 if args.mode == "full-coverage" else 2.0))

    # ---- common derived quantities (printed for context) -----------------
    delta = 1.0 / np.sqrt(np.pi * f * 4e-7 * np.pi * sigma)

    print(f"\nConstraints")
    print(f"  substrate              = {args.substrate} "
          f"(eps_r = {eps_r}, tan delta = {tan_d:.1e})")
    print(f"  metal                  = {args.metal} "
          f"(sigma = {sigma:.2e} S/m)")
    print(f"  frequency              = {args.f_GHz} GHz")
    print(f"  metal thickness t_m    = {args.tm_nm} nm")
    print(f"  skin depth delta_s     = {delta*1e6:.3f} um")
    print(f"  R_s (coth)             = {rs:.4f} Ohm/sq")
    print(f"  target u_opt           = {u_opt}")
    print()

    # ---- mode dispatch ---------------------------------------------------
    if args.mode == "full-coverage":
        W = args.W_um * 1e-6
        S = Z0_to_S(W, eps_r)
        ac = alpha_c(W, f, sigma, t_m, eps_r)
        ad = alpha_d(f, eps_r, tan_d)
        # u = (alpha_c + alpha_d) * L = u_opt  ==>  L = u_opt / (ac + ad)
        L = u_opt / (ac + ad)
        AcL = ac * L
        AdL = ad * L
        A_c_eff = ac * W / rs            # geometric prefactor in Np/Ohm

        print("Full-coverage optimum (n = 0, u_opt = 1)")
        print("----------------------------------------")
        print(f"  W                      = {W*1e6:.3f} um  (fixed)")
        print(f"  S (Z0_to_S)            = {S*1e6:.3f} um")
        print(f"  L                      = {L*1e3:.3f} mm")
        print(f"  L / W                  = {L/W:.2f}")
        print(f"  alpha_c                = {ac:.2f} Np/m")
        print(f"  alpha_d                = {ad*1e3:.4f} mNp/m")
        print(f"  alpha_c * L            = {AcL:.4f} Np  ({AcL*8.686:.2f} dB)")
        print(f"  alpha_d * L            = {AdL*1e3:.4f} mNp")
        print(f"  total u                = {AcL+AdL:.4f} Np")
        print(f"  A_c (full Pucel)       = {A_c_eff:.4f} Np/Ohm  "
              f"(simplified-form anchor was 0.0741)")
        out = dict(mode="full-coverage", W=W, S=S, L=L, t_m=t_m,
                   f=f, eps_r=eps_r, tan_d=tan_d, sigma=sigma,
                   Rs=rs, alpha_c=ac, alpha_d=ad, u=AcL+AdL,
                   substrate=args.substrate, metal=args.metal)

    else:  # fixed-footprint
        w_s = args.w_s_um * 1e-6
        l_s = args.l_s_um * 1e-6
        # First-pass: built-in W_opt_meander uses u_opt = 2 implicitly
        # (n = 1 case).  Iterate self-consistently for the user's u_opt.
        W = W_opt_meander(w_s, l_s, f, sigma, t_m, eps_r)
        for _ in range(5):
            S = Z0_to_S(W, eps_r)
            pitch = W + 2 * S
            n_pass = max(1, int(np.floor(w_s / pitch)))
            L = n_pass * l_s
            ac = alpha_c(W, f, sigma, t_m, eps_r)
            ad = alpha_d(f, eps_r, tan_d)
            u_now = (ac + ad) * L
            # Adjust W so that u_now -> u_opt; alpha_c ~ 1/W so
            # u_now ~ A_c * R_s * (n_pass*l_s) / W.  Iterate.
            if u_now <= 0:
                break
            W = W * (u_now / u_opt)
        S = Z0_to_S(W, eps_r)
        pitch = W + 2 * S
        n_pass = max(1, int(np.floor(w_s / pitch)))
        L = n_pass * l_s
        ac = alpha_c(W, f, sigma, t_m, eps_r)
        ad = alpha_d(f, eps_r, tan_d)
        u_total = (ac + ad) * L

        print(f"Fixed-footprint optimum (n = 1, u_opt = {u_opt})")
        print( "------------------------------------------------")
        print(f"  sample footprint       = {w_s*1e6:.0f} x {l_s*1e6:.0f} um")
        print(f"  W                      = {W*1e6:.3f} um")
        print(f"  S                      = {S*1e6:.3f} um  "
              f"(pitch = {pitch*1e6:.2f} um)")
        print(f"  n_passes               = {n_pass}")
        print(f"  L                      = {L*1e3:.3f} mm  "
              f"(= {n_pass} x {l_s*1e3:.2f} mm)")
        print(f"  alpha_c * L            = {ac*L:.4f} Np")
        print(f"  alpha_d * L            = {ad*L*1e3:.4f} mNp")
        print(f"  total u                = {u_total:.4f} Np  (target {u_opt})")
        out = dict(mode="fixed-footprint", W=W, S=S, L=L, t_m=t_m,
                   f=f, w_s=w_s, l_s=l_s, n_pass=n_pass,
                   eps_r=eps_r, tan_d=tan_d, sigma=sigma,
                   Rs=rs, alpha_c=ac, alpha_d=ad, u=u_total,
                   substrate=args.substrate, metal=args.metal)

    # ---- write JSON for MC chaining --------------------------------------
    out_path = os.path.join(os.path.dirname(__file__),
                            "optimised_design.json")
    with open(out_path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
