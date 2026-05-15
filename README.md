# CPW Sensitivity Optimizer

Streamlit companion app for the paper on coplanar-waveguide (CPW) design
optimization for broadband magnetic resonance spectroscopy (EPR / FMR /
broadband VNA).  Given a sample geometry, a noise regime and a fabrication
process, the app picks the centre-conductor width *W* and line length *L*
that maximise the spin-signal sensitivity, and shows how the four
canonical fabrication choices (PCB-mill vs litho × straight vs meander)
compare against the absolute optimum.

## Install

```bash
pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

Then run the test suite with:

```bash
pytest -q
```

## Optimization framework

The signal that emerges from a CPW spin-resonance line follows a
**Beer–Lambert** form: the microwave probe is attenuated as it travels,
so the absorbed power along the line is `P_in (1 − e^{−2αL})`, while the
spin-signal contribution from a thin sample on top of the line scales as
`η · α_c · L · e^{−αL}`.  The product `u^{n+1} · e^{−u}` of dimensionless
loss `u = α_c L` therefore has a finite maximum at
`u_opt = (n + 1)/(1 − φ)`: too-short lines waste sample volume, too-long
lines absorb the probe before it reaches the far end.  The exponent `n`
distinguishes full-coverage (`n = 0`) from fixed-footprint (`n = 1`)
samples, while the noise-mixing parameter `φ` interpolates between
field-subtracted EPR/FMR spectroscopy (`φ = 0`, additive noise) and
baseline-limited VNA scans (`φ → 1`, multiplicative ripple).  All
geometric and material trade-offs in the app are derived from this single
optimization principle.
