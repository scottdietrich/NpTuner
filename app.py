"""Streamlit app: CPW Sensitivity Optimizer.

Companion tool to the CPW design-optimization paper for broadband magnetic
resonance spectroscopy.  Select a sample, a noise regime and a fabrication
process, and the app solves the conformal-mapping + loss physics to pick the
best centre-conductor width W and line length L.
"""

from __future__ import annotations

import os
import sys

# Make sure cpw_physics.py (sitting next to this file) is importable no matter
# what directory streamlit was launched from.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import streamlit as st

from cpw_physics import (
    SUBSTRATE_PRESETS,
    METAL_PRESETS,
    Z0_to_S,
    Rs,
    alpha_c,
    alpha_d,
    u_opt,
    fom_curve,
    W_opt_meander,
    regime_classify,
    filling_factor,
)

# --------------------------------------------------------------------------
# App config
# --------------------------------------------------------------------------
st.set_page_config(page_title="CPW Sensitivity Optimizer",
                   layout="wide",
                   page_icon=None)

st.title("CPW Sensitivity Optimizer")
st.caption(
    "Broadband magnetic-resonance spectroscopy — companion app to the "
    "paper on coplanar-waveguide design optimization."
)

# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
with st.sidebar:
    st.header("Sample")
    sample_type = st.radio(
        "Sample type",
        ["Full coverage (powder/coating)", "Fixed footprint (crystal/flake)"],
        index=1,
    )

    if sample_type.startswith("Fixed"):
        w_s_um = st.number_input("Sample width w_s (µm)",
                                 min_value=1.0, max_value=50000.0,
                                 value=500.0, step=10.0)
        l_s_um = st.number_input("Sample length l_s (µm)",
                                 min_value=1.0, max_value=50000.0,
                                 value=500.0, step=10.0)
        w_s = w_s_um * 1e-6
        l_s = l_s_um * 1e-6
        # Sample thickness in nm (thin films / crystals)
        t_nm = st.number_input("Sample thickness t (nm)",
                               min_value=0.1, max_value=10000.0,
                               value=100.0, step=10.0)
        t_sample = t_nm * 1e-9
        n_regime = 1  # partial coverage
    else:
        w_s = np.inf
        l_s = np.inf
        t_um = st.number_input("Sample thickness t (µm)",
                               min_value=0.1, max_value=1000.0,
                               value=50.0, step=1.0)
        t_sample = t_um * 1e-6
        n_regime = 0  # full coverage

    f_ghz = st.slider("Frequency f (GHz)", 0.5, 20.0, 10.0, 0.1)
    f = f_ghz * 1e9

    st.header("Noise regime")
    phi = st.slider("Noise mixing parameter φ", 0.0, 1.0, 0.0, 0.05)
    st.caption(
        "φ = 0: field-subtracted EPR/FMR measurement — additive noise floor "
        "dominates, finite optimum exists.  φ → 1: baseline-limited VNA scan "
        "— multiplicative ripple dominates, optimum diverges (longer is "
        "always better)."
    )

    st.header("Fabrication")
    sub_key = st.selectbox("Substrate", list(SUBSTRATE_PRESETS.keys()), index=2)
    sub = SUBSTRATE_PRESETS[sub_key]
    metal_key = st.selectbox("Metal", list(METAL_PRESETS.keys()), index=0)
    sigma = METAL_PRESETS[metal_key]
    t_m_nm = st.slider("Metal thickness t_m (nm)", 50, 500, 200, 10)
    t_m = t_m_nm * 1e-9

    fab_options = {
        "PCB mill (W_min = 150 µm)":   150e-6,
        "Optical litho (W_min = 10 µm)": 10e-6,
        "E-beam litho (W_min = 1 µm)":    1e-6,
        "Custom":                         None,
    }
    fab_choice = st.selectbox("Fabrication process",
                              list(fab_options.keys()), index=1)
    W_min_preset = fab_options[fab_choice]
    if W_min_preset is None:
        W_min_um = st.number_input("Custom W_min (µm)",
                                   min_value=0.1, max_value=5000.0,
                                   value=50.0, step=1.0)
        W_min = W_min_um * 1e-6
    else:
        W_min = W_min_preset

# --------------------------------------------------------------------------
# Calculations for the four comparison cards
# --------------------------------------------------------------------------
eps_r = sub["eps_r"]
tan_d = sub["tan_delta"]

def _pick_W(is_meander: bool, W_min_case: float) -> float:
    """Choose W for a given case.

    - Meander + fixed footprint: use W_opt_meander (clipped at W_min).
    - Meander + full coverage: the optimum is not set by footprint; use W_min
      as a proxy (best resolution available).
    - Straight + fixed footprint: W forced to sample width w_s (if > W_min)
      else W_min.
    - Straight + full coverage: W_min gives the highest Rs/W and therefore
      the fastest route to u_opt → use W_min.
    """
    if sample_type.startswith("Fixed"):
        if is_meander:
            W_opt = W_opt_meander(w_s, l_s, f, sigma, t_m, eps_r)
            return max(W_min_case, W_opt)
        # straight line in a fixed footprint fills the sample width
        return max(W_min_case, min(w_s, w_s))   # = w_s clipped to W_min below
    # full coverage
    return W_min_case


def _build_case(label, is_meander, fab_W_min, metal_sigma, metal_t_m):
    """Assemble a dict of all numbers needed to render one comparison card."""
    if sample_type.startswith("Fixed"):
        if is_meander:
            W_candidate = W_opt_meander(w_s, l_s, f, metal_sigma,
                                        metal_t_m, eps_r)
            W = max(fab_W_min, W_candidate)
            # length of line that meanders the sample footprint:
            S = Z0_to_S(W, eps_r)
            pitch = W + 2 * S
            n_passes = max(1, int(np.floor(w_s / pitch)))
            L = n_passes * l_s
            fits_meander = (W_candidate >= fab_W_min) and (pitch <= w_s)
        else:
            # straight line: W set by sample width, L = sample length
            W = max(fab_W_min, min(w_s, w_s))
            L = l_s
            fits_meander = False
    else:  # full coverage — no footprint constraint
        W = fab_W_min
        # length required to hit u = u_opt exactly
        ac = alpha_c(W, f, metal_sigma, metal_t_m, eps_r)
        ad = alpha_d(f, eps_r, tan_d)
        L = u_opt(n_regime, phi) / (ac + ad)
        fits_meander = True

    ac = alpha_c(W, f, metal_sigma, metal_t_m, eps_r)
    ad = alpha_d(f, eps_r, tan_d)
    u_c = ac * L
    u_d = ad * L
    u_total = u_c + u_d
    u_star = u_opt(n_regime, phi)

    # FOM achieved: (u/u*)^(n+1) * exp(-(u - u*))
    if np.isfinite(u_star):
        fom_here = (u_total / u_star) ** (n_regime + 1) * \
                   np.exp(-(u_total - u_star))
    else:
        fom_here = np.nan
    fom_here = float(np.clip(fom_here, 0.0, 1.0))

    return {
        "label":   label,
        "W":       W,
        "L":       L,
        "u_c":     u_c,
        "u_d":     u_d,
        "u":       u_total,
        "u_star":  u_star,
        "fom_frac": fom_here,
        "fits":    fits_meander,
        "meander": is_meander,
    }


# PCB = Cu, 35 µm default for PCB trace.
# litho = user metal selection (default Au thin film).
pcb_t_m = 35e-6                      # 1 oz Cu
pcb_sigma = METAL_PRESETS["Cu"]
litho_t_m = t_m                      # sidebar value
litho_sigma = sigma                  # sidebar value

cases = [
    _build_case("A  Straight — PCB",      False, 150e-6, pcb_sigma, pcb_t_m),
    _build_case("B  Meander — PCB",       True,  150e-6, pcb_sigma, pcb_t_m),
    _build_case("C  Straight — litho",    False, W_min,  litho_sigma, litho_t_m),
    _build_case("D  Meander — litho",     True,  W_min,  litho_sigma, litho_t_m),
]

# --------------------------------------------------------------------------
# Regime callout
# --------------------------------------------------------------------------
if sample_type.startswith("Fixed"):
    W_opt_m = W_opt_meander(w_s, l_s, f, litho_sigma, litho_t_m, eps_r)
    regime = regime_classify(W_min, w_s, W_opt_m)

    if regime == "meander":
        st.success(
            f"**Regime — meander achievable.**  "
            f"Your fabrication W_min = {W_min*1e6:.1f} µm ≤ W_opt = "
            f"{W_opt_m*1e6:.1f} µm, so a meander fills the {w_s*1e6:.0f} × "
            f"{l_s*1e6:.0f} µm² footprint at its sensitivity optimum."
        )
    elif regime == "B1":
        # Improvement factor available if user dropped to optical/e-beam
        st.warning(
            f"**Regime B1 — straight line, inside footprint.**  "
            f"W_min = {W_min*1e6:.1f} µm is larger than W_opt = "
            f"{W_opt_m*1e6:.2f} µm but still ≤ w_s = {w_s*1e6:.0f} µm.  "
            f"Upgrading to a process that can reach W ≤ W_opt would enable "
            f"meandering and improve sensitivity by roughly "
            f"√(w_s · l_s / (2 · W_opt²)) ≈ "
            f"{np.sqrt(w_s*l_s/(2*W_opt_m**2)):.1f}×."
        )
    else:  # B2
        st.error(
            f"**Regime B2 — line wider than sample.**  "
            f"W_min = {W_min*1e6:.1f} µm exceeds w_s = {w_s*1e6:.0f} µm. "
            f"Most of the line's microwave field does not overlap the "
            f"sample.  Consider a finer-resolution process (optical or "
            f"e-beam litho) or a larger sample."
        )
else:
    with st.expander("No fixed footprint? (Regime A)", expanded=False):
        st.markdown(
            """
            For full-coverage samples any width *W* is usable.  The line
            length is then chosen to hit the conductor-loss optimum
            $(\\alpha_c + \\alpha_d)\\, L = u_\\mathrm{opt}$, which on a
            low-loss substrate reduces to

            $$\\frac{L}{W} = \\frac{u_\\mathrm{opt}}{A_c\\,R_s}$$

            i.e. the aspect ratio is set entirely by the conductor-loss
            prefactor $A_c$ and the surface resistance $R_s$.  Narrower
            *W* just scales *L* down proportionally.
            """
        )

# --------------------------------------------------------------------------
# 2 x 2 comparison cards
# --------------------------------------------------------------------------
st.subheader("Four-way comparison")

W_scan = np.logspace(np.log10(max(W_min, 1e-6)) - 0.5,
                     np.log10(max(W_min * 1000, 5e-3)), 80)

def _render_card(col, case, i_curve):
    with col:
        W_um = case["W"] * 1e6
        L_mm = case["L"] * 1e3
        L_um = case["L"] * 1e6
        lw = f"{L_mm:.2f} mm" if L_mm >= 1 else f"{L_um:.0f} µm"
        st.markdown(f"**{case['label']}**")
        st.markdown(
            f"- W = {W_um:.1f} µm, L = {lw}  \n"
            f"- L/W = {case['L']/case['W']:.1f}  \n"
            f"- u = {case['u']:.2f} Np = {case['u']*8.686:.1f} dB  \n"
            f"- u★ = {case['u_star']:.2f} Np  "
            f"(FOM = {100*case['fom_frac']:.1f}%)  \n"
            f"- α_c·L = {case['u_c']*8.686:.2f} dB, "
            f"α_d·L = {case['u_d']*8.686:.2f} dB"
        )
        # compact plotly chart of the FOM curve
        u_arr, _, fom_norm = fom_curve(W_scan, case["L"], f,
                                       (pcb_sigma if "PCB" in case["label"]
                                        else litho_sigma),
                                       (pcb_t_m if "PCB" in case["label"]
                                        else litho_t_m),
                                       eps_r, tan_d, n_regime, phi)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=u_arr, y=fom_norm,
                                 mode="lines", name="FOM(u)"))
        fig.add_vline(x=case["u"], line_dash="dash",
                      line_color="crimson",
                      annotation_text=f"u = {case['u']:.2f}",
                      annotation_position="top")
        if np.isfinite(case["u_star"]):
            fig.add_vline(x=case["u_star"], line_dash="dot",
                          line_color="green",
                          annotation_text=f"u★ = {case['u_star']:.2f}",
                          annotation_position="bottom")
        fig.update_layout(
            height=220, margin=dict(l=10, r=10, t=10, b=10),
            xaxis_title="u = α_c·L (Np)",
            yaxis_title="FOM / FOM_max",
            xaxis_type="log",
        )
        st.plotly_chart(fig, use_container_width=True,
                        key=f"fom_{i_curve}")


row1 = st.columns(2)
row2 = st.columns(2)
_render_card(row1[0], cases[0], 0)
_render_card(row1[1], cases[1], 1)
_render_card(row2[0], cases[2], 2)
_render_card(row2[1], cases[3], 3)


# --------------------------------------------------------------------------
# Meander diagram
# --------------------------------------------------------------------------
st.subheader("Meander layout")

def _draw_meander(W, w_s_plot, l_s_plot):
    fig, ax = plt.subplots(figsize=(6, 2.8))
    if not np.isfinite(w_s_plot):
        # show a single straight line as schematic
        ax.add_patch(plt.Rectangle((0, -W*5e5), 1, W*1e6,
                                   color="#c27a00"))
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-W*1e6, W*1e6)
        ax.set_xlabel("Straight line — full coverage (Regime A)")
        ax.set_yticks([])
    else:
        S = Z0_to_S(W, eps_r)
        pitch = W + 2 * S
        n_pass = max(1, int(np.floor(w_s_plot / pitch)))
        # draw the sample footprint
        ax.add_patch(plt.Rectangle((0, 0), w_s_plot*1e6, l_s_plot*1e6,
                                   fill=True, color="#f2f2f2",
                                   edgecolor="black", lw=0.8))
        # draw meander traces
        for i in range(n_pass):
            x0 = i * pitch * 1e6 + S * 1e6
            ax.add_patch(plt.Rectangle((x0, 0), W*1e6, l_s_plot*1e6,
                                       color="#c27a00"))
            # connector at alternating ends
            if i < n_pass - 1:
                y_conn = 0 if i % 2 else l_s_plot*1e6
                ax.add_patch(plt.Rectangle((x0, y_conn - W*1e6/2),
                                           pitch*1e6, W*1e6,
                                           color="#c27a00"))
        # annotation of W and S
        ax.annotate("", xy=(S*1e6, l_s_plot*1e6*0.3),
                    xytext=((S+W)*1e6, l_s_plot*1e6*0.3),
                    arrowprops=dict(arrowstyle="<->", color="black"))
        ax.text((S+W/2)*1e6, l_s_plot*1e6*0.35,
                f"W = {W*1e6:.1f} µm", ha="center", fontsize=8)
        ax.annotate("", xy=(0, l_s_plot*1e6*0.6),
                    xytext=(S*1e6, l_s_plot*1e6*0.6),
                    arrowprops=dict(arrowstyle="<->", color="black"))
        ax.text(S*1e6/2, l_s_plot*1e6*0.65,
                f"S = {S*1e6:.1f} µm", ha="center", fontsize=8)
        ax.set_xlim(-pitch*1e6*0.1, w_s_plot*1e6*1.02)
        ax.set_ylim(-l_s_plot*1e6*0.05, l_s_plot*1e6*1.05)
        ax.set_xlabel(f"sample footprint {w_s_plot*1e6:.0f} × "
                      f"{l_s_plot*1e6:.0f} µm² "
                      f"({n_pass} pass{'es' if n_pass != 1 else ''})")
    ax.set_aspect("equal")
    ax.set_yticks([])
    return fig


# show the meander diagram for the litho-meander case (case D) since it is
# usually the headline result
W_show = cases[3]["W"] if sample_type.startswith("Fixed") else W_min
st.pyplot(_draw_meander(W_show, w_s, l_s), clear_figure=True)


# --------------------------------------------------------------------------
# Footer: surface resistance and filling-factor sanity info
# --------------------------------------------------------------------------
st.subheader("Your design")
rs_user = Rs(f, sigma, t_m)
eta, thin_ok = filling_factor(cases[3]["W"] if sample_type.startswith("Fixed")
                              else W_min, t_sample)
cc1, cc2, cc3 = st.columns(3)
cc1.metric("R_s (sidebar metal/thickness)", f"{rs_user*1e3:.2f} mΩ/□")
cc2.metric("Skin depth δ_s",
           f"{1e6/np.sqrt(np.pi*f*4e-7*np.pi*sigma):.2f} µm")
cc3.metric("Sample η (thin-film)", f"{eta:.3g}",
           delta=("thin-film ok" if thin_ok else "t/W ≥ 0.1 — not thin"))
