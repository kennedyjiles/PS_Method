# Parker–Sochacki Power Series Method: Charged-Particle Motion in Magnetic Fields

## Overview

This repository contains the simulation codes and analysis tools that compare the
**Parker–Sochacki power series (PS) integration method** against several
**Runge–Kutta-based solvers**: fixed-step fourth-order (RK4), adaptive Dormand–Prince
(RK45; a compiled, step-for-step port of SciPy's `RK45`), and the symplectic two-stage
Gauss–Legendre Runge–Kutta (RKG), all for relativistic charged-particle motion in a static
magnetic dipole field. Over integrations of up to ~10⁹ gyroperiods (~10 years), the PS
method conserves kinetic energy **at least 10 orders of magnitude** better than RK4 and
**4–7 orders** better than RKG, conserves canonical momentum 4–11 orders better than RK4
and comparably to RKG, and matches RK45 at its strictest tolerances with roughly ten
times less runtime.

The work was developed in the **Space Weather Lab, Department of Physics and Astronomy,
George Mason University**. This repository accompanies:

> H. L. Jiles and R. S. Weigel, *"High-Precision Long-Time Numerical Solutions of
> Charged-Particle Motion in a Static Magnetic Dipole Field,"* 2026.
> [arXiv:2604.20876](https://doi.org/10.48550/arXiv.2604.20876) ·
> software archive [doi:10.5281/zenodo.21026049](https://doi.org/10.5281/zenodo.21026049)

Three field drivers are included. **Only the dipole is part of the paper**; the two simpler
fields were used to develop and validate the method and appear in the thesis only:

- **`dipoleb.py`** — Dipole field (Earth's dipole analog) — **the paper**:  $\mathbf{B(r)}=\dfrac{3(\mathbf{m}\cdot\hat{\mathbf{r}})\hat{\mathbf{r}}-\mathbf{m}}{r^3}$
- **`constb.py`** — Uniform field (thesis):  $\mathbf{B}=B_0\,\hat{\mathbf{z}}$
- **`hyperb.py`** — Hyperbolic-tangent field (Harris current-sheet analog; thesis):  $\mathbf{B}=B_0\tanh(y/\delta)\,\hat{\mathbf{z}}$

---

## ⚠️ What is in the paper vs. ongoing research

This codebase contains **both** the paper material **and** unpublished thesis work.

| | In the paper | Thesis / ongoing research (NOT in the paper) |
|---|---|---|
| **Scope** | PS vs RK4 / RK45 / RKG accuracy, stability and runtime in the static dipole field | Uniform- and hyperbolic-field benchmarks; bounce/drift periods; applications of full-orbit PS trajectories to radiation-belt physics |
| **Drivers / configs** | `dipoleb.py` with the `paper1`–`paper3` configs (fixed step) | `constb.py`, `hyperb.py`; `dipoleb.py` adaptive mode + Dragt diagnostics |
| **Batch reproductions** | `scripts/batch_runner_eandp.py` (Appendix B tables; Figs. 3 and 6) | `scripts/batch_runner_walt.py` (bounce/drift period tables)<br>`scripts/batch_runner_fluxmap.py` (dwell-occupancy sweeps) |
| **Analysis / plots** | kinetic-energy and canonical-momentum error, trajectories, runtime-vs-error scatter | magnetic-moment μ diagnostics, bounce/drift periods, dwell-time maps, Dragt–Störmer / Dragt–Finn chaos, Poincaré sections, Walt ε–L stability |

The paper notes (Sec. 1) that *"in a subsequent work, we will address applications of
high-accuracy long-time solutions."* The **constb / hyperb** drivers, the **bounce/drift**
tables, the **dwell-time map** and the **Dragt–Störmer/Finn** tools fall under that thesis
work — they are usable and documented, but their interfaces, defaults, and outputs may
change. See [Thesis and Ongoing Research](#thesis-and-ongoing-research-not-in-the-paper)
at the end.

---

## Repository Layout

```
.
├── run.py                       # Unified entry point (dispatches to the right driver)
├── constb.py                    # Driver — uniform field (thesis)
├── hyperb.py                    # Driver — hyperbolic / current-sheet field (thesis)
├── dipoleb.py                   # Driver — dipole field (fixed + adaptive step) — THE PAPER
│
├── configs/                     # YAML run configurations (the "run modes")
│   ├── config_loader.py         #   loads a run yml, merges with base.yml, validates
│   ├── constb/                  #   base.yml, demo.yml, paper.yml, ...
│   ├── hyperb/                  #   base.yml, demo.yml, paper1..paper4.yml, ...
│   └── dipoleb/                 #   base.yml, demo.yml, paper1..paper3.yml, electrons/, protons/, ...
│
├── ps_method/                   # Core library (importable package)
│   ├── constants.py             #   shared physical constants (q_e, m_e, m_p, RE, B_0)
│   ├── constb_physics.py        #   PS / analytical / Lorentz kernels — uniform field
│   ├── hyperb_physics.py        #   PS / Lorentz kernels — hyperbolic field
│   ├── dipoleb_physics.py       #   PS dipole integrator (fixed step, streaming) + RKG
│   ├── dipoleb_adaptive.py      #   adaptive-step PS dipole integrator (streaming)
│   ├── constb_hyperb_energy_analysis.py  # KE drift (constb/hyperb)
│   ├── dipoleb_energy_analysis.py        # KE, P_phi and μ drift (dipole, one chunked pass)
│   ├── dipoleb_moment_analysis.py        # magnetic-moment μ diagnostics (dipole)
│   ├── dipoleb_bouncedrift_analysis.py   # bounce/drift period detection (dipole)
│   ├── dipoleb_dragt_analysis.py         # Dragt (1965) diagnostics — RESEARCH
│   ├── constb_hyperb_plots.py / dipoleb_plots.py   # plotting
│   ├── utils.py                 #   shared numerical + plotting utilities, RK4 + compiled RK45
│   └── writers.py               #   HDF5 I/O, run hashing, summaries, master CSV
│
├── scripts/                     # Stand-alone tools (not imported by the drivers)
│   ├── batch_runner_eandp.py    #   PAPER: proton/electron method-comparison sweeps
│   ├── batch_runner_walt.py     #   THESIS: electron/proton bounce–drift period sweeps
│   ├── batch_runner_fluxmap.py  #   RESEARCH: dwell-occupancy parameter sweeps
│   ├── batch_runner_trappedplots.py  # RESEARCH: trapped-particle L sweeps with phased refinement
│   ├── dragt.py / si_to_dragt.py            # RESEARCH: Dragt-unit conversions
│   ├── test_dragt_roundtrip.py / test_dragt_thorough.py  # Dragt-unit validation tests
│   ├── inspect_hdf5.py / trim_h5.py / benchmark_compression.py  # h5 utilities
│   ├── build_vds.py / compare_runs.py                          # segment stitching + bit-compare
│   ├── run_at_night.py          #   pause/resume a long run within a time window
│   └── plots/                   #   post-processing figure scripts
│       ├── scatterplot.py / build_summary_results.py  # PAPER: Appendix B tables, Figs. 3 / 6
│       ├── trappedbands.py / flux_map_builder.py / launch_fate_map.py  # RESEARCH
│       ├── epsilon_map.py / fma_map.py                # RESEARCH: (L, E) μ-verdict maps, frequency maps
│       ├── waltplot.py / timestepplots.py             # THESIS: bounce/drift and step-size plots
│       ├── animate_merged.py                          # trajectory animation (animate.py is legacy)
│       └── animate_section.py / animate_mu_kicks.py   # explainer videos (section_demo runs)
│
├── data/                        # Created on first run — all outputs land here
├── ps_method.yml                # Conda environment specification
└── README.md
```

> **Note:** This layout reflects the current, refactored codebase (lowercase drivers,
> YAML configs, an importable `ps_method/` package, and `data/` outputs). Earlier
> versions used a different structure (`functions/`, `utility_scripts/`,
> `test_particles/*.py`, `outputs/`); those names are obsolete.

---

## Installation

### Conda (recommended)

```bash
git clone https://github.com/kennedyjiles/PS_Method.git
cd PS_Method
conda env create -f ps_method.yml
conda activate ps_method
```

### Key dependencies

| Package | Version (`ps_method.yml`) |
|---------|---------------------------|
| Python | 3.12 |
| NumPy | 2.1 |
| SciPy | 1.16 |
| Matplotlib | 3.10 |
| pandas | 3.0 |
| h5py | 3.14 |
| Numba | 0.61 |
| PyYAML | 6.0 |
| psutil | (used only by `scripts/run_at_night.py`) |

### pip alternative

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install numpy scipy matplotlib pandas h5py numba pyyaml psutil
```

> **Precision note:** Plots that probe errors near machine precision, and runs that use
> `float128` / `longdouble`, can differ slightly across library versions and hardware.
> The relative ordering of the methods (PS ≪ RK) is robust; the exact floor of a
> near-machine-precision curve may shift.

---

## Running Simulations

The unified entry point is **`run.py`**. A run is defined by a **YAML config** under
`configs/<field>/`. The two convenient invocation styles:

```bash
# Shorthand:  python run.py <field> <config-name>
python run.py constb demo
python run.py hyperb demo
python run.py dipoleb demo

# Explicit config path
python run.py configs/dipoleb/demo.yml
```

You can also call a driver directly (`python dipoleb.py demo`) — it resolves
`configs/dipoleb/demo.yml` the same way.

### Run modes (configs)

Every field has a `base.yml` (full default parameter set) plus thin override configs that
merge on top of it:

| Config | Purpose |
|--------|---------|
| `demo` | Quick verification — a short run (seconds) to check setup and visualize trajectories. **Default.** |
| `paper1`–`paper3` (dipoleb) | Full-scale reproductions of the paper figures (≈15–20 min each). |
| `paper`, `paper1`–`paper4` (constb, hyperb) | Thesis benchmark runs for the simpler fields (seconds to minutes). |
| `manual_template` | Template for replotting a cached `.h5` without re-running solvers. |

To create a custom run, copy a config (e.g. `configs/dipoleb/demo.yml`) and edit the
physics fields (`energy_eV`, `pitch_deg`, `x_initial`, `particle`, `gyroperiods`, …).
See [`configs/dipoleb/base.yml`](configs/dipoleb/base.yml) for the full set of options and
inline documentation.

### Replotting from cached data

Once a run has written its `.h5`, you can regenerate plots without re-integrating. Every
run folder holds a copy of its config; passing any yml that lives under
`data/` replots that run:

```bash
python run.py data/dipoleb/demo/<hash>/demo.yml   # yml under data/ → replot mode
python run.py configs/dipoleb/manual_template.yml --replot
```

---

## Reproducing the Paper Figures

All paper figures come from `dipoleb.py`. Every fixed-step run uses 65 steps per
equatorial gyroperiod (Δτ = 12.1) at L = 5 R_E over a common interval of τ/T = 1.5×10⁵
unless noted; RK45 runs with rtol = 10⁻¹⁸, atol = 10⁻²⁰. Each config takes ≈15–20 min in
`float64`.

| Command | Figure(s) | Physics |
|---------|-----------|---------|
| `python run.py dipoleb paper1` | Figs. 1, 2 | 100 keV proton, α=30°: PS13 / RK45 / RKG trajectory and invariant errors |
| `python run.py dipoleb paper2` | Figs. 4, 5 | 100 MeV electron, α=60°: PS13 / RK4 / RK45 / RKG |
| `python run.py dipoleb paper3` | Fig. 7 | target-accuracy case study (Sec. 5.2): PS22 (Δτ=54) vs RK4 (Δτ=0.5) vs RKG (Δτ=7.9) vs RK45 (rtol 10⁻¹³, atol 10⁻¹⁵) |
| `scripts/batch_runner_eandp.py --phase 1` + plot scripts | Fig. 3, Table B.1 | proton grid: runtime-vs-error scatter (see [Appendices](#reproducing-the-paper-appendices)) |
| `scripts/batch_runner_eandp.py --phase 2` + plot scripts | Fig. 6, Table B.2 | electron grid: runtime-vs-error scatter |

> The extended curves in Figs. 2 and 5 (PS13 to τ/T = 10⁹, RKG to 2×10⁷) are the same
> configs with `gyroperiods` raised; set `ps_segment_gyroperiods` so the multi-GB run is
> checkpointed. Each run writes trajectory plots, relative kinetic-energy, P_φ and μ error
> plots, and a summary `.txt` into its run folder (see [Output](#output-and-post-processing)).

---

## Reproducing the Paper Appendices

Appendix B is a parameter sweep. `scripts/batch_runner_eandp.py` reproduces it by
launching many `dipoleb.py` runs in parallel and consolidating the results into a single
`master_simulation_log.csv` per group.

### Method-comparison tables (Appendix B, Tables B.1 / B.2; Figs. 3, 6)

`scripts/batch_runner_eandp.py` runs the proton and electron four-method (PS/RK4/RK45/RKG)
comparison grid at L=5 used for the accuracy/runtime tables and bubble plots.

```bash
# Phase 1 = protons (10 keV, 100 keV, 1 MeV, 10 MeV at 90° and 30°)
python scripts/batch_runner_eandp.py --phase 1

# Phase 2 = electrons (50 keV, 1 MeV, 100 MeV, 150 MeV at 90° and 60°)
python scripts/batch_runner_eandp.py --phase 2

python scripts/batch_runner_eandp.py --phase 1 --dry-run   # preview the run plan
python scripts/batch_runner_eandp.py --phase 1 --resume    # skip already-completed cells
```

Outputs land in `data/dipoleb/protons/` and `data/dipoleb/electrons/`. Then build the
summary tables (B.1 / B.2) and the runtime-vs-error scatter plots (Figs. 3 / 6):

```bash
python scripts/plots/build_summary_results.py              # → *_summary_results.csv
python scripts/plots/scatterplot.py                        # proton plot (default)
SUMMARY_CSV=electron_summary_results.csv USE_ELECTRON=1 python scripts/plots/scatterplot.py
```

> **Note:** `batch_runner_eandp.py` shares its progress file (`scripts/batch_progress.json`)
> with the thesis runner `batch_runner_walt.py`. Run one sweep at a time, or clear the file
> between unrelated sweeps.

---

## Output and Post-Processing

On first run a `data/` directory is created. Outputs are organized by field and config:

```
data/
└── <field>/                     # constb | hyperb | dipoleb
    └── <config-or-group>/       # demo, paper1, protons, electrons, fluxmap_10mev, ...
        ├── master_simulation_log.csv   # (dipole) parameters + Dragt diagnostics per run
        ├── <hash>/                     # one folder per run
        │   ├── figures/<hash>_*.png    # trajectory, slice, and error plots
        │   ├── <hash>_summary.txt      # initial conditions, per-method runtimes, error stats
        │   └── <config>.yml            # provenance copy of the config (replot with it)
        └── _rawdata/
            ├── <hash>_full.h5          # streamed trajectory (a virtual dataset when segmented)
            └── <hash>_segNNN.h5        # checkpoint segments (dipole, ps_segment_gyroperiods > 0)
```

Each run's identity is a hash of its physics parameters, so re-running with the same
physics reuses the cached `.h5` (set `read_data: true`). Caution: a single long dipole
run can produce a multi-GB `.h5`.

### HDF5 storage format

The dipole integrators stream to disk during integration. The `ps/y` dataset stores **9
rows** per time step:

| Rows | Contents |
|------|----------|
| 0–2 | Position (x, y, z) in R_E |
| 3–5 | Velocity (vx, vy, vz), dimensionless |
| 6–8 | Magnetic field (Bx, By, Bz), dimensionless |

The 8 internal PS auxiliary variables (r², a, b, c, d, e, f, g) are **not** saved — they
are only needed during integration. PS orders used per step are stored in `ps/orders`.
Inspect any file with:

```bash
python scripts/inspect_hdf5.py data/dipoleb/demo/_rawdata/<hash>_full.h5
```

---

## Precision and Truncation Control

- **Floating point.** Runs default to `float64`. Set `use_float128: true` in a config to
  engage extended precision (`float128` / `longdouble`) — useful only for studying the
  error floor, and much slower. For the dipole it also disables the Numba JIT and the RKG
  solver, so it is practical only for short runs. (`float128` requires platform support;
  Windows typically maps it to `longdouble`.)
- **Adaptive PS order.** The PS series is truncated dynamically: each step adds orders
  until the largest relative term contribution falls below machine epsilon for the working
  precision. This gives efficient convergence without sacrificing accuracy.

---

## Thesis and Ongoing Research (NOT in the paper)

> Everything in this section is **thesis work**, not part of the paper: the two simpler
> field benchmarks, the bounce/drift period tables, and tools that apply the full-orbit
> PS dipole integrator to radiation-belt physics. The research tools are under
> development — interfaces, defaults, and output formats may change. They are documented
> here for collaborators; happy to discuss.

### Uniform and hyperbolic fields (`constb.py`, `hyperb.py`)

The two simpler fields were used to develop and validate the PS method before the dipole
work. Each has a `demo` config plus full-scale `paper*` configs (the name is historical;
these runs appear in the thesis only):

| Command | Physics |
|---------|---------|
| `python run.py constb paper` | 100 eV electron in B = B₀ ẑ, 10⁵ gyroperiods |
| `python run.py hyperb paper1` | 10 keV e⁻, α=75°, δ=500 km (mild gradient) |
| `python run.py hyperb paper2` | 10 keV e⁻, α=75°, δ=50 km (strong gradient) |
| `python run.py hyperb paper3` | 100 keV proton, α=−15°, δ=200 km |
| `python run.py hyperb paper4` | same as paper1 with a 10× larger PS step (coarse-step convergence) |

`use_float128: true` reproduces the extended-precision error-floor curves for these
fields (much slower).

### Bounce / drift period tables

`scripts/batch_runner_walt.py` runs the long PS integrations used to measure bounce and
drift periods across energy and L-shell against the analytical guiding-centre
approximations (Walt 1994); `scripts/plots/waltplot.py` plots them.

```bash
python scripts/batch_runner_walt.py --phase 1
python scripts/batch_runner_walt.py --phase 1 --dry-run
```

### Dragt–Störmer / Dragt–Finn full-orbit trapping

Dragt (1965) and Dragt & Finn (1976) showed that dipole trapping is governed by two exact
invariants — energy *W₀²* and canonical angular momentum *P_φ* — with a trapping boundary
*W₀²* < *P_φ⁴/16*, valid even when the adiabaticity parameter ε = r_g·|∇B|/B exceeds unity
and the magnetic moment μ is no longer conserved.

- **`dipoleb.py` adaptive mode** (`ps_adaptive`/`solvers.adaptive` in the config) — hybrid
  fixed/adaptive PS stepping that subdivides by local |B| in steep-gradient regions, with
  NaN/Inf rollback and atmosphere-impact detection (`ps_method/dipoleb_adaptive.py`).
- **Dragt diagnostics** (`ps_method/dipoleb_dragt_analysis.py`) — computes *W₀²*, *P_φ*,
  trapping boundary (CLOSED/OPEN), orbit character (REGULAR/CHAOTIC, Dragt 1965 eq. 6.1),
  adiabaticity ε(t), Poincaré surfaces of section, and meridian-plane projections. Written
  to each run's summary and the `master_simulation_log.csv`.
- **Dragt-unit conversions** — `scripts/dragt.py` (Dragt → physical) and
  `scripts/si_to_dragt.py` (physical → Dragt), validated by
  `scripts/test_dragt_roundtrip.py` and `scripts/test_dragt_thorough.py`.
- **Walt ε–L stability diagrams** — `scripts/plots/trappedbands.py` plots ε vs L-shell with
  trapped / atmosphere / untrapped zones and REGULAR/CHAOTIC classification, from a
  `master_simulation_log.csv`.

### Dwell-time (occupancy) maps

A meridian-plane map of where trapped orbits spend their time — a step toward
AP-8-style belt-occupancy maps. **This is a dwell-time occupancy map, not a calibrated
omnidirectional flux** (the toroidal Jacobian and a true pitch-angle integration are not
applied).

```bash
# 1. Sweep a fine L grid at one or more energies (writes h5 trajectories)
python scripts/batch_runner_fluxmap.py 10mev
python scripts/batch_runner_fluxmap.py all --resume

# 2. Build the meridian dwell map from the h5 trajectories
python scripts/plots/flux_map_builder.py --group fluxmap_10mev
python scripts/plots/flux_map_builder.py --group fluxmap_all --per-energy

# Optional: launch-fate map (trapped / lost / escaped) from a master CSV
python scripts/plots/launch_fate_map.py data/dipoleb/fluxmap_all/master_simulation_log.csv --pitch 90
```

---

## Citation

If you use this code, please cite the paper:

> H. L. Jiles and R. S. Weigel, *"High-Precision Long-Time Numerical Solutions of
> Charged-Particle Motion in a Static Magnetic Dipole Field,"* 2026.
> [arXiv:2604.20876](https://doi.org/10.48550/arXiv.2604.20876)

and the archived software (Zenodo mints a new version DOI for each release; this DOI
is the v1.0.0 release cited in the paper):

> H. L. Jiles and R. S. Weigel (2026). *Jiles PSM Initial Release* (v1.0.0). Zenodo.
> [doi:10.5281/zenodo.21026049](https://doi.org/10.5281/zenodo.21026049)

### Key references

- Dragt, A. J. (1965). Trapped orbits in a magnetic dipole field. *Reviews of Geophysics*, 3(2), 255–298.
- Dragt, A. J., & Finn, J. M. (1976). Insolubility of trapped particle motion in a magnetic dipole field. *J. Geophys. Res.*, 81(13), 2327–2340.
- Parker, G., & Sochacki, J. (1996). Implementing the Picard iteration. *Neural, Parallel, and Scientific Computations*, 4, 97–112.
- Northrop, T. G. (1963). Adiabatic charged-particle motion. *Reviews of Geophysics*, 1(3), 283–304.
- Yugo, H., & Iyemori, T. (2007). Symplectic tracing of high-energy charged particles in the inner magnetosphere. *J. Geophys. Res.*, 112, A06243.
