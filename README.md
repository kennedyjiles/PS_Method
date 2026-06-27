# Parker–Sochacki Power Series Method: Charged-Particle Motion in Magnetic Fields

## Overview

This repository contains the simulation codes and analysis tools that compare the
**Parker–Sochacki power series (PS) integration method** against several
**Runge–Kutta-based solvers** — fixed-step fourth-order (RK4), adaptive Dormand–Prince
(RK45), and the symplectic two-stage Gauss–Legendre Runge–Kutta (RKG) — for
charged-particle motion in static magnetic fields. Across three field configurations of
increasing complexity, the PS method achieves **4–13 orders of magnitude** better
kinetic-energy conservation than the RK methods over long integrations.

The work was developed in the **Space Weather Lab, Department of Physics and Astronomy,
George Mason University**. This repository accompanies:

> H. Jiles and R. Weigel, *"High-Accuracy Numerical Solutions of Particle Motion in
> Static Magnetic Fields,"* 2026.
> [arXiv:2604.20876](https://doi.org/10.48550/arXiv.2604.20876)

Three benchmark problems are included (in order of increasing complexity):

- **`constb.py`** — Uniform field:  $\mathbf{B}=B_0\,\hat{\mathbf{z}}$
- **`hyperb.py`** — Hyperbolic-tangent field (Harris current-sheet analog):  $\mathbf{B}=B_0\tanh(y/\delta)\,\hat{\mathbf{z}}$
- **`dipoleb.py`** — Dipole field (Earth's dipole analog):  $\mathbf{B(r)}=\dfrac{3(\mathbf{m}\cdot\hat{\mathbf{r}})\hat{\mathbf{r}}-\mathbf{m}}{r^3}$

---

## ⚠️ What is in the paper vs. ongoing research

This codebase contains **both** the published paper material **and** active, unpublished
thesis research. Please be clear about the distinction:

| | In the published paper | Ongoing research (NOT in the paper) |
|---|---|---|
| **Scope** | PS vs RK accuracy/stability benchmarks in three static fields | Applications of full-orbit PS trajectories to radiation-belt physics |
| **Drivers / configs** | `constb.py`, `hyperb.py`, `dipoleb.py` with the `paper*` configs | `dipoleb.py` adaptive mode + Dragt diagnostics |
| **Batch reproductions** | `scripts/batch_runner_eandp.py` (method-comparison tables)<br>`scripts/batch_runner_walt.py` (bounce/drift period tables) | `scripts/batch_runner_fluxmap.py` (dwell-occupancy sweeps) |
| **Analysis / plots** | energy & momentum error, trajectories, bounce/drift periods | dwell-time maps, Dragt–Störmer / Dragt–Finn chaos, Poincaré sections, Walt ε–L stability |

The paper itself notes (Sec. 6.2.1) that *"although we do not consider applications of
improved numerical methods for particle trajectory calculations in this work, this is an
active area of research."* The **dwell-time map** and **Dragt–Störmer/Finn** tools below
fall under that ongoing work — they are usable and documented, but their interfaces,
defaults, and outputs may change. See [Ongoing Research](#ongoing-research-not-in-the-paper)
at the end.

---

## Repository Layout

```
.
├── run.py                       # Unified entry point (dispatches to the right driver)
├── constb.py                    # Driver — uniform field
├── hyperb.py                    # Driver — hyperbolic / current-sheet field
├── dipoleb.py                   # Driver — dipole field (fixed + adaptive step)
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
│   ├── dipoleb_energy_analysis.py        # KE + P_phi drift (dipole, chunked)
│   ├── dipoleb_moment_analysis.py        # magnetic-moment μ diagnostics (dipole)
│   ├── dipoleb_bouncedrift_analysis.py   # bounce/drift period detection (dipole)
│   ├── dipoleb_dragt_analysis.py         # Dragt (1965) diagnostics — RESEARCH
│   ├── constb_hyperb_plots.py / dipoleb_plots.py   # plotting
│   ├── utils.py                 #   shared numerical + plotting utilities, RK4
│   └── writers.py               #   HDF5 I/O, run hashing, summaries, master CSV
│
├── scripts/                     # Stand-alone tools (not imported by the drivers)
│   ├── batch_runner_eandp.py    #   PAPER: proton/electron method-comparison sweeps
│   ├── batch_runner_walt.py     #   PAPER: electron/proton bounce–drift period sweeps
│   ├── batch_runner_fluxmap.py  #   RESEARCH: dwell-occupancy parameter sweeps
│   ├── dragt.py / si_to_dragt.py            # RESEARCH: Dragt-unit conversions
│   ├── test_dragt_roundtrip.py / test_dragt_thorough.py  # Dragt-unit validation tests
│   ├── inspect_hdf5.py / trim_h5.py / benchmark_compression.py  # h5 utilities
│   ├── run_at_night.py          #   pause/resume a long run within a time window
│   └── plots/                   #   post-processing figure scripts
│       ├── scatterplot.py / build_summary_results.py  # method-comparison plots (paper)
│       ├── trappedbands.py / flux_map_builder.py / launch_fate_map.py  # RESEARCH
│       ├── waltplot.py / timestepplots.py             # supplementary plots
│       └── animate.py / animate_merged.py             # trajectory animations
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
| `paper`, `paper1`, `paper2`, … | Full-scale reproductions of specific paper figures (minutes to hours). |
| `manual_template` | Template for replotting a cached `.h5` without re-running solvers. |

To create a custom run, copy a config (e.g. `configs/dipoleb/demo.yml`) and edit the
physics fields (`energy_eV`, `pitch_deg`, `x_initial`, `particle`, `gyroperiods`, …).
See [`configs/dipoleb/base.yml`](configs/dipoleb/base.yml) for the full set of options and
inline documentation.

### Replotting from cached data

Once a run has written its `.h5`, you can regenerate plots without re-integrating:

```bash
python run.py data/dipoleb/demo/demo.yml          # data/ path auto-triggers replot mode
python run.py configs/dipoleb/manual_template.yml --replot
```

---

## Reproducing the Paper Figures

After setting up the environment, run the commands below. Approximate `float64` runtimes
are noted; `float128` (`use_float128: true` in the config) reproduces the
extended-precision curves and takes substantially longer.

### Uniform field — Section 4

| Command | Figure | Physics |
|---------|--------|---------|
| `python run.py constb paper` | Fig. 1 | 100 eV electron, 10⁵ gyroperiods |

### Hyperbolic / current-sheet field — Section 5

| Command | Figure | Physics |
|---------|--------|---------|
| `python run.py hyperb paper1` | Fig. 2 | 10 keV e⁻, α=75°, δ=500 km (mild gradient) |
| `python run.py hyperb paper2` | Fig. 3 | 10 keV e⁻, α=75°, δ=50 km (strong gradient) |
| `python run.py hyperb paper3` | Fig. 4 | 100 keV proton, α=−15°, δ=200 km |
| `python run.py hyperb paper4` | Fig. 5 | same as paper1 but PS uses a 10× larger step (coarse-step convergence demo) |

### Dipole field — Section 6

| Command | Figure(s) | Physics |
|---------|-----------|---------|
| `python run.py dipoleb paper1` | Figs. 6, 7, 8 | 100 keV proton, α=30°, L=5 R_E (PS/RK45/RKG) |
| `python run.py dipoleb paper2` | Figs. 10, 11, 12 | 100 MeV electron, α=60°, L=5 R_E |
| `python run.py dipoleb paper3` | Fig. 14 | error-matched comparison: PS (order 16, Δτ=54) vs RK4 (Δτ=0.5) vs RKG |

> Each paper run writes trajectory plots, relative kinetic-energy (and, for the dipole,
> P_φ and μ) error plots, and a summary `.txt` into its output folder
> (see [Output](#output-and-post-processing)). The dipole runs are the longest (≈15–20 min
> each in `float64`).

---

## Reproducing the Paper Appendices

The appendix datasets are large parameter sweeps. Two batch runners reproduce them; each
launches many `dipoleb.py` runs in parallel and consolidates the results into a single
`master_simulation_log.csv` per group.

### Method-comparison tables (Appendices A.2 / A.3; Figs. 8, 12)

`scripts/batch_runner_eandp.py` runs the proton and electron four-method (PS/RK4/RK45/RKG)
comparison grid at L=5 used for the dipole accuracy/runtime tables and bubble plots.

```bash
# Phase 1 = protons (10 keV, 100 keV, 1 MeV, 10 MeV at 90° and 30°)
python scripts/batch_runner_eandp.py --phase 1

# Phase 2 = electrons (50 keV, 1 MeV, 100 MeV, 150 MeV at 90° and 60°)
python scripts/batch_runner_eandp.py --phase 2

python scripts/batch_runner_eandp.py --phase 1 --dry-run   # preview the run plan
python scripts/batch_runner_eandp.py --phase 1 --resume    # skip already-completed cells
```

Outputs land in `data/dipoleb/protons/` and `data/dipoleb/electrons/`. Then build the
summary tables and the runtime-vs-error scatter plots (Figs. 8 / 12):

```bash
python scripts/plots/build_summary_results.py              # → *_summary_results.csv
python scripts/plots/scatterplot.py                        # proton plot (default)
SUMMARY_CSV=electron_summary_results.csv USE_ELECTRON=1 python scripts/plots/scatterplot.py
```

### Bounce / drift period tables (Appendices A.4 / A.5; Figs. 9, 13)

`scripts/batch_runner_walt.py` runs the long integrations used to measure characteristic
bounce and drift periods across energy and L-shell, compared against the analytical
guiding-center approximations (Walt). These are PS-derived and appear in the paper.

```bash
python scripts/batch_runner_walt.py --phase 1
python scripts/batch_runner_walt.py --phase 1 --dry-run
```

> **Note:** `batch_runner_eandp.py` and `batch_runner_walt.py` currently share one
> progress file (`scripts/batch_progress.json`). Run them one group at a time, or clear
> the file between unrelated sweeps.

---

## Output and Post-Processing

On first run a `data/` directory is created. Outputs are organized by field and config:

```
data/
└── <field>/                     # constb | hyperb | dipoleb
    └── <config-or-group>/       # demo, paper1, protons, electrons, fluxmap_10mev, ...
        ├── <hash>_*.png         # trajectory, slice, and error plots
        ├── <hash>_summary.txt   # initial conditions, per-method runtimes, error stats
        ├── master_simulation_log.csv   # (dipole) parameters + Dragt diagnostics per run
        └── _rawdata/
            └── <hash>_full.h5   # raw streamed trajectory (if write_data: true)
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
  engage extended precision (`float128` / `longdouble`) for the `constb` and `hyperb`
  cases — useful only for studying the error floor, and much slower. **The dipole
  integrator runs in `float64` only.** (`float128` requires platform support; Windows
  typically maps it to `longdouble`.)
- **Adaptive PS order.** The PS series is truncated dynamically: each step adds orders
  until the largest relative term contribution falls below a tolerance (machine epsilon by
  default, or a user value). This gives efficient convergence without sacrificing accuracy.

---

## Ongoing Research (NOT in the paper)

> These tools apply the full-orbit PS dipole integrator to radiation-belt physics. They are
> part of **active thesis work** and are under development — interfaces, defaults, and
> output formats may change. They are documented here for collaborators; happy to discuss.

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

If you use this code, please cite:

> H. Jiles and R. Weigel, *"High-Accuracy Numerical Solutions of Particle Motion in Static
> Magnetic Fields,"* 2026. [arXiv:2604.20876](https://doi.org/10.48550/arXiv.2604.20876)

### Key references

- Dragt, A. J. (1965). Trapped orbits in a magnetic dipole field. *Reviews of Geophysics*, 3(2), 255–298.
- Dragt, A. J., & Finn, J. M. (1976). Insolubility of trapped particle motion in a magnetic dipole field. *J. Geophys. Res.*, 81(13), 2327–2340.
- Parker, G., & Sochacki, J. (1996). Implementing the Picard iteration. *Neural, Parallel, and Scientific Computations*, 4, 97–112.
- Northrop, T. G. (1963). Adiabatic charged-particle motion. *Reviews of Geophysics*, 1(3), 283–304.
