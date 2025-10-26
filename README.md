# Parker–Sochacki Method: Charged-Particle Motion in Magnetic Fields

## Overview
This repository contains the full suite of Python codes developed to compare the **Parker–Sochacki (PS) power-series integration method** against several **Runge–Kutta-based solvers** (fixed-step fourth order (RK4), adaptive Dormand-Prince (RK45), and the symplectic application of the Gauss-Lagrange Runge-Kutta) for charged-particle motion in various magnetic-field configurations.

The project was developed as part of graduate research at **George Mason University**, focused on energy and magnetic-moment conservation, adaptive truncation, and numerical performance across field geometries representative of magnetospheric environments.

Three benchmark problems are included:
- **`constB.py`** — Uniform magnetic field
- **`hyperB.py`** — Hyperbolic tangent field $B_z = B_0 \tanh(\alpha y)$ (current-sheet analog)
- **`dipoleB.py`** — Dipole magnetic field (Earth's dipole representation) 

Each driver can be run in **demo** or **paper** mode, depending on whether a fast diagnostic or full-scale reproduction of published results is desired.

---

## Repository Layout

```
.
├── constB.py                   # Main simulation driver (uniform field)
├── hyperB.py                   # Main simulation driver (hyperbolic/current-sheet field)
├── dipoleB.py                  # Main simulation driver (dipole field)
│
├── functions/
│   ├── functions_library_constB.py
│   ├── functions_library_hyper.py
│   ├── functions_library_dipole.py
│   └── functions_library_universal.py     # Shared numerical + plotting utilities
│
├── test_particles/
│   ├── constB_testparticles.py
│   ├── hyperB_testparticles.py
│   └── dipoleB_testparticles.py
│
├── misc_plots/                 # Example figures and post-processing scripts
├── master_simulation_log.csv   # Optional CSV record of parameter sets
├── ps_method.yml               # Conda environment specification
└── README.md
```

---

## Installation and Environment Setup

### Option 1 — Conda (recommended)
Create the exact environment used for the paper:

```bash
conda env create -f ps_method.yml
conda activate ps_method
```

### Option 2 — Virtual environment with pip
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install numpy scipy matplotlib pandas h5py numba
```

### Key dependencies
- Python = 3.9.13 
- NumPy = 1.21.5  
- SciPy = 1.9.1  
- Matplotlib = 3.5.2  
- Pandas = 1.4.4  
- h5py = 3.7.0  
- Numba = 0.56.3  

---


## Running Simulations

Each of the three main simulation drivers (`constB.py`, `hyperB.py`, `dipoleB.py`) can be executed directly from the repository root:

```bash
python constB.py
python hyperB.py
python dipoleB.py
```

By default, **all scripts run in `demo` mode** unless otherwise specified. The default settings produce short, lightweight runs suitable for quick verification and visualization.

At the top of each main driver file, a small configuration block defines the **run mode**, with a note on available modes. To switch between modes, update this block or supply a command-line argument if supported. The main driver file (e.g., `dipoleB.py`) determines which configuration to execute—`demo` or `paper`—and calls the corresponding setup and integration routines.

Each **test particle file** (e.g., `constB_testparticles.py`, `dipoleB_testparticles.py`) defines the initial particle parameters (position, velocity, charge-to-mass ratio, kinetic, etc.)  and integration settings (time step, number of steps, maximum PS order, tolerances, etc.). These parameters are passed to the solver functions in the corresponding field-specific library. 

### Run modes
| Mode | Purpose | Description |
|------|----------|-------------|
| `demo` | Quick verification | Short, lightweight run for testing setup and viewing basic trajectory plots (typically seconds). |
| `paper` | Full simulation | High-order PS expansion with small time steps, reproducing results from the paper (can take several minutes). |

Example (inside a driver or test script):
```python
run = "demo"      # or "paper"
```

---

## Example Command-Line Usage

### 1. Constant Magnetic Field
Quick demo:
```bash
python constB.py
```
Produces a circular trajectory and energy conservation plot in `constB_outputs_demo/`.

Reproduce paper results:
```bash
python constB.py --mode paper
```
Generates high-order PS runs with kinetic energy error plots (10⁻¹¹–10⁻¹⁸ range).

### 2. Hyperbolic Magnetic Field
Quick demo:
```bash
python hyperB.py
```
Visualizes the figure-eight motion near the current sheet with adaptive PS truncation.

### 3. Dipole Magnetic Field
Quick demo:
```bash
python dipoleB.py
```
Produces a 3-D trajectory in Earth-like dipole geometry and logs drift/bounce statistics.

Reproduce full paper datasets:
```bash
python dipoleB.py --mode paper
```
Expected outputs are saved under `dipoleB_outputs_paper/` with trajectory and error summaries.

---

## Precision and Truncation Control

### Floating-point precision
Precision can be toggled globally using the `npfloat` alias inside the test particle files:
- `npfloat = np.float64` (default, fast and sufficient for most demo runs)
- `npfloat = np.float128` (extended precision for paper-quality energy conservation)

When using `float128`, make sure your platform supports it (Linux/macOS only; Windows typically maps it to long double).

### Adaptive PS-order truncation
The Parker–Sochacki expansion is truncated dynamically based on term magnitude:
- The **PS order \(M\)** increases until consecutive term contributions drop below a chosen tolerance.
- Tolerances can be set to machine epsilon (`np.finfo(npfloat).eps`) or a user-defined threshold (e.g., `1e-35`).
- This adaptive termination ensures efficient convergence without compromising accuracy.

---

## Output and Post-Processing

Each run automatically creates a results folder in the working directory, e.g.:

```
constB_outputs_demo/
dipoleB_outputs_paper/
```

These directories typically include:
- 3-D trajectory plots (`.png`, `.pdf`)
- Time-series diagnostics (kinetic energy, relative error, magnetic moment)
- Optional `.csv` or `.h5` data archives for post-processing
- Logs of PS order and truncation metrics (for paper mode)

Supplementary analysis and comparison figures are stored under **`misc_plots/`**, along with scripts that generate the publication figures for the dipole runs and substation analyses.

---

## Performance Notes

- **Numba acceleration:** All core PS recurrence loops and Runge–Kutta updates are JIT-compiled with `@njit` for near-C performance.  
- **Series order:** Typical PS orders range 10–40 depending on field geometry and tolerance.  
- **Step size:** Usually expressed in units of the gyroperiod; ensuring ~65–100 integration points per gyration yields accurate energy conservation.  
- **Conservation metrics:** Relative kinetic-energy errors reach \(10^{-11}\)–\(10^{-18}\) depending on precision.

---

## Extending the Framework

To modify or extend:
1. Add a new field configuration by creating a `functions_library_<field>.py` file and corresponding `<field>.py` driver.
2. Define recurrence relations for the PS method and include analytical or RK reference solvers for comparison.
3. Register any new helper in `functions_library_universal.py` for consistent plotting, timing, and error handling.

---

## Reproducibility and Data

- The repository is deterministic: given the same initial conditions, PS order, and tolerance, results should reproduce exactly.
- For large simulations, HDF5 caching can be enabled to store all particle time series efficiently.
- The `master_simulation_log.csv` records parameter sets for traceability.

---

## Citation
If you use this code or build upon it in your research, please cite:

> H. Jiles and R. Weigel, *“The Parker-Sochacki Method vs. Runge-Kutta Methods for Particle Motion
in Static Magnetic Fields,”* in preparation (2026).

---

## Contact
For questions, bug reports, or collaboration:
- **Author:** Heather Jiles  
- **Advisor:** Dr. Robert Weigel, George Mason University  
- **Email:** (add if desired)  
- Or open an issue on the repository.

---
