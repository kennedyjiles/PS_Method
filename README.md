# Parker–Sochacki Method: Charged-Particle Motion in Magnetic Fields

## Overview
This repository contains a suite of Python codes developed to compare the **Parker–Sochacki (PS) power-series integration method** against several **Runge–Kutta-based solvers** (fixed-step fourth order (RK4), adaptive Dormand-Prince (RK45), and the symplectic application of the Gauss-Lagrange Runge-Kutta (RKG)) for charged-particle motion in various magnetic-field configurations, demonstrating the PS method achieves superior energy conservation.

The project was developed as part of graduate research at the Physics and Astronomy Department at **George Mason University**. This repository accompanies the research paper *”High-Accuracy Numerical Solutions of Particle Motion in Static Magnetic Fields,”* 2026, by H. Jiles and R. Weigel ([arXiv:2604.20876](https://doi.org/10.48550/arXiv.2604.20876)), providing simulation codes and analysis scripts used in the study.

The repository also includes ongoing thesis work applying the PS method to the **Dragt–Störmer problem** - full Lorentz-orbit trapping of high-energy particles in a dipole magnetic field - extending the integrator with adaptive time stepping, Dragt (1965) diagnostics, Michel (1980) phase-space analysis, and AP-8-style flux mapping (see [Dragt–Störmer Analysis](#dragtstörmer-analysis-in-progress) below).

> **Note:** The Dragt–Störmer analysis tools (batch runner, flux map builder, Michel phase portraits, adaptive integrator, and associated utilities) are part of active thesis research and are under ongoing development. Interfaces, parameter defaults, and output formats may change as the work progresses.

Three benchmark problems are included:
- **`constB.py`** - Uniform magnetic field: 
```math
\mathbf{B}=B_0\mathbf{\hat{z}}
```
- **`hyperB.py`** - Hyperbolic tangent field (1-D current-sheet analog): 
```math
\mathbf{B}= B_0 \tanh(y/\gamma)\mathbf{\hat{z}}
```
- **`dipoleB.py`** - Dipole magnetic field (Earth's dipole analog): 
```math
\mathbf{B(r)}=\frac{\mu_0}{4\pi}\left[\frac{3\mathbf{r(m\cdot r)}}{r^5}-\frac{\mathbf{m}}{r^3}\right]
```

Each of these drivers can be run in **demo** or **paper** modes, depending on whether a fast diagnostic or full-scale reproduction of the paper results is desired.


---

## Repository Layout

```
.
├── constB.py                   # Main simulation driver (uniform field)
├── hyperB.py                   # Main simulation driver (hyperbolic/current-sheet field)
├── dipoleB.py                  # Main simulation driver (dipole field, fixed + adaptive step)
├── constants.py                # Shared physical constants (q_e, m_e, m_p, RE, B_0, etc.)
│
├── functions/
│   ├── functions_library_constB.py
│   ├── functions_library_hyper.py
│   ├── functions_library_dipole.py      # PS dipole integrator (fixed step, streaming)
│   ├── functions_library_dipole_adp.py  # PS dipole integrator (adaptive step, streaming)
│   ├── functions_library_dragt.py       # Dragt diagnostics (W₀², P_φ, adiabaticity, Poincaré)
│   ├── functions_library_universal.py   # Shared numerical + plotting utilities
│   └── functions_library_universal_chunk.py  # Chunked energy/mu computation for large runs
│
├── utility_scripts/
│   ├── batch_flux_runner.py    # Batch parameter sweep runner (energy, L, pitch angle phases)
│   ├── flux_map_builder.py     # AP-8-style meridian flux map from batch h5 trajectories
│   ├── michel_phase_portrait.py # Michel (1971/1980) phase portraits (α vs φ at equatorial crossings)
│   ├── si_to_dragt.py          # Convert SI initial conditions to Dragt dimensionless units
│   ├── dragt.py                # Standalone Dragt orbit integration
│   ├── inspect_hdf5.py         # Inspect h5 data files
│   ├── project_setup.py        # Shared imports, logger, constants
│   ├── logger_util.py          # Logging configuration
│   ├── run_at_night.py         # Batch runner for overnight parameter sweeps
│   ├── walt_diagnostics_v3.py  # Walt stability diagram: ε vs L with Dragt classification
│   ├── walt_diagnostics_v2.py  # Multi-panel Walt diagnostic plots
│   ├── test_dragt_roundtrip.py # SI ↔ Dragt unit conversion tests
│   └── test_dragt_thorough.py  # Extended Dragt parameter validation
│
├── test_particles/
│   ├── constB_testparticles.py
│   ├── hyperB_testparticles.py
│   └── dipoleB_testparticles.py
│
├── misc_plots/                 # Additional plotting and post-processing scripts
├── USER_MANUAL_batch_tools.txt # User manual for batch runner, flux map, and Michel scripts
├── ps_method.yml               # Conda environment specification
└── README.md
```

---

## Installation and Environment Setup

### Option 1 - Virtual environment with pip
```bash
git clone https://github.com/kennedyjiles/PS_Method.git
cd PS_Method

# Create and activate a local virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install exact versions used for the paper
pip install numpy==1.21.5 scipy==1.9.1 matplotlib==3.5.2 \
            pandas==1.4.4 h5py==3.7.0 numba==0.56.3
```
Alternatively, to use the latest available versions: 
```bash
pip install numpy scipy matplotlib pandas h5py numba
```
Results may vary slightly between versions and hardware, specifically for plots showing errors near machine precision and the use of `float128` or `longdouble`. 

### Option 2 - Conda
To exactly replicate the environment used for the paper:

```bash
git clone https://github.com/kennedyjiles/PS_Method.git
cd PS_Method
conda env create -f ps_method.yml
conda activate ps_method
```

### **Key dependencies**

| Package | Version used (paper) |
|----------|----------------------|
| Python   | 3.9.13 |
| NumPy    | 1.21.5 |
| SciPy    | 1.9.1 |
| Matplotlib | 3.5.2 |
| Pandas   | 1.4.4 |
| h5py     | 3.7.0 |
| Numba    | 0.56.3 |

> **Note:**  
> The versions listed above reproduce the exact results and figures from the paper.  
> Later versions of these libraries generally work but may yield small numerical differences in precision- or tolerance-limited plots.



---


## Running Simulations

Each of the three main simulation drivers (`constB.py`, `hyperB.py`, `dipoleB.py`) can be executed directly from the repository root:

```bash
python constB.py
python hyperB.py
python dipoleB.py
```

By default, **all scripts are run in `demo` mode**. The default settings produce short, lightweight simulations suitable for quick verification and visualization.

At the top of each main driver file, a small configuration block defines the **run mode**, with a note on available modes (multiple `paper` modes are available for hyperb.py and dipoleb.py i.e., `paper1`, `paper2`, etc.). To switch between modes, update this block or supply a command-line argument. The main driver file (e.g., `dipoleB.py`) determines which configuration to execute (`demo` or `paper`) and calls the corresponding setup and integration routines.

Each **test particle file** (e.g., `constB_testparticles.py`, `hyperB_testparticles.py`, and `dipoleB_testparticles.py`) defines the initial particle parameters (position, velocity, charge-to-mass ratio, kinetic energy, etc.) and integration settings (time step, number of steps, maximum PS order, tolerances, etc.). These parameters are passed to the solver functions in the corresponding field-specific library. 

The dipole analysis involved a significant number of simulations. For reproducibility, only the configurations corresponding to the representative simulations in the paper’s text are included. Appendix-level datasets can be generated by altering the parameters in the **test particle files** or by adding new run modes. However, the datapoints and scripts are provided in `misc_plots/` for reference and use.

### Run modes
| Mode | Purpose | Description |
|------|----------|-------------|
| `demo` | Quick verification | Runs a shorter simulation of the high-order PS expansion vs. RK methods to verify correct setup and visualize particle trajectories. Ideal for testing configurations and generating quick diagnostic plots (typically seconds).|
| `paper` | Full simulation | Runs full simulation using the same high-order PS expansion to evaluate energy conservation and numerical stability over longer time durations. This mode reproduces the long-timescale results presented in the paper and can take several minutes in `float64` precision.|

Example (inside a driver script):
```python
run = "demo"    # options: "demo", "paper1", "paper2", "paper3", or "paper4"
```

---

## Example Command-Line Usage

### 1. Constant Magnetic Field
Quick demo:
```bash
python constB.py
```
Simulating the trajectory of a 100 eV electron for about 10 gyroperiods in constant magnetic field. Expected outputs are saved under `outputs/outputs_demo/` with trajectory and error summaries.

### 2. Hyperbolic Magnetic Field
Reproduce full paper dataset:
```bash
python hyperB.py paper2
```
Full simulation an 10 keV electron near a 1-D current sheet with a half-thickness of 500 km. Expected outputs are saved under `outputs/outputs_paper/` with trajectory slice and error summaries.

### 3. Dipole Magnetic Field
Quick demo:
```bash
python dipoleB.py 
```
Short demo of 100 keV electron in Earth's dipole magnetic field located at a distance of 5 Earth Radii. Expected outputs are saved under `outputs/outputs_demo/` with trajectory and error summaries.

Reproduce full paper dataset:
```bash
python dipoleB.py paper2
```
Full simulation of a 100 keV electron in Earth's dipole magnetic field located at a distance of 5 Earth Radii, exhibiting characteristic bounce and drift motions. Expected outputs are saved under `outputs/outputs_paper/` with trajectory slice and error summaries.

---

## Precision and Truncation Control

### Floating-point precision
By default, all scripts are set to use `float64`. For the uniform field (`constB`) and hyperbolic field (`hyperB`) cases, precision can be changed globally using the `USE_FLOAT128` toggle inside the **test particle file**:
- `USE_FLOAT128=False`- defaults to `float64`, fast and sufficient for most simulations. Recommended for most uses.
- `USE_FLOAT128=True` - engages `float128` or `longdouble` extended precision for analyzing numerical gains if available, long run times. Not recommended for general use.

When using `float128`, make sure your platform supports it (Linux/macOS only; Windows typically maps it to `longdouble`).

> **Note:** The `float128` option is currently only functional for the `constB` and `hyperB` cases. The dipole integrator (`dipoleB`) uses `float64` exclusively. 

### Adaptive PS-order truncation
The Parker–Sochacki expansion is truncated dynamically based on term magnitude:
- The **PS order** increases until consecutive term contributions drop below a chosen tolerance.
- Tolerances can be set to machine epsilon (`np.finfo(npfloat).eps`) or a user-defined threshold (e.g., `1e-35`).
- This adaptive termination ensures efficient convergence without compromising accuracy.

---

## Output and Post-Processing

Upon first running, a simulation will create an `outputs/` directory in the working directory with subdirectories organized by run type:

```
outputs/
├── outputs_demo/              # Demo-mode runs
├── outputs_paper/             # Paper-mode runs
│   ├── ConstB/
│   ├── hyperB/
│   ├── dipole/
│   ├── protons/
│   ├── electrons/
│   └── master_simulation_log.csv
├── outputs_rawdata/           # Raw h5 trajectory data
├── outputs_extended_runs/     # Extended-duration simulations
├── outputs_giant_runs/        # Large-scale parameter sweeps
├── outputs_giant_RKG_PS/      # RKG vs PS comparison runs
├── dragt/                     # Dragt–Störmer analysis outputs
├── walt/                      # Walt stability diagram outputs
└── paper/                     # Additional paper figures
```

These directories typically include:
- 2-D and/or 3-D trajectory plots (`.png`), either full trajectories and slices of the final orbits for trajectory comparison
- Plots of relative kinetic error calculations (`.png`) 
- A summary of the simulation such as initial conditions, run time by method, time step size, etc. (`.txt`)
- For dipole simulations, a `master_simulation_log.csv` records parameter sets and Dragt diagnostics across runs
- Dipole runs create per-run subfolders (named by run hash) containing plots and the summary `.txt`; raw `.h5` data is stored separately in the run storage directory

Hyperbolic and dipole simulations have the ability to write the raw data to an Hierarchical Data Format 5 file, h5, this can be toggled on with `WRITE_DATA=True` in the **test particle scripts**. This option will automatically create a folder:
```
outputs/outputs_rawdata/
```
These files can then be accessed to re-create plots and perform additional analysis by the `READ_DATA=True` toggle in the test particles scripts. Both `WRITE_DATA` and `READ_DATA` are set to `False` by default. Caution should be exercised using these options as a single simulation can generate up to 2GB of data. The `h5` files can be inspected with the `inspect_hdf5.py` script:

```
python utility_scripts/inspect_hdf5.py outputs/outputs_rawdata/run_d7e387cd8e81f8a9.h5
```

Supplementary analysis and comparison figures are stored under **`misc_plots/`**, along with scripts that generate the additional publication figures for the dipole simulations and analyses.

### HDF5 Storage Format

The dipole integrators stream trajectory data to h5 files during integration. The stored dataset `ps/y` contains 9 rows per time step:

| Indices | Contents |
|---------|----------|
| 0–2 | Position (x, y, z) in R_E |
| 3–5 | Velocity (vx, vy, vz) in dimensionless units |
| 6–8 | Magnetic field (Bx, By, Bz) in dimensionless units |

The 8 internal PS auxiliary variables used during series computation are not saved to disk, as they are never needed by post-processing tools. The `save_rows` attribute on the `ps` group identifies the format. Orders used by the PS series at each step are stored in a separate `ps/orders` dataset. To inspect the contents of any h5 file:

```bash
python utility_scripts/inspect_hdf5.py outputs/outputs_rawdata/run_d7e387cd8e81f8a9.h5
```

---

## Reproducing Paper Figures

After setting up the environment as described in the **Installation and Environment Setup**, execute the following:
```python
python constB.py paper # wait several seconds
python hyperB.py paper1 # wait a few minutes
python hyperB.py paper2 # wait a few minutes
python hyperB.py paper3 # wait 5-10 minutes
python hyperB.py paper4 # wait a few minutes
python dipoleB.py paper1 # wait approximately 15-20 minutes
python dipoleB.py paper2 # wait approximately 15-20 minutes
python dipoleB.py paper3 # wait a few minutes
```
This produces the main `float64` figures from the report. See **Floating-point precision** section on how to turn on `float128` or `longdouble` in each of the **test particle** scripts and repeat the above commands. Note that it will take several hours to reproduce these figures and assumes your system is capable of executing.

As noted, the dipole analysis involved a significant number of simulations. For reproducibility, only the configurations corresponding to the representative simulations in the paper’s text are included. Datasets from Appendix can be generated by altering the parameters in the **test particle files** or by adding new run modes. However, the datapoints and scripts are provided in `misc_plots/` for reference and use.

---

## Running Custom Dipole Simulations

The dipole test particle file (`test_particles/dipoleB_testparticles.py`) serves as the central configuration for `dipoleB.py`. Fixed-step vs adaptive integration is controlled by the `USE_ADAPTIVE` flag in each run mode. To run a custom simulation, add a new run mode to the `load_params()` function with your desired parameters.

### Adding a New Run Mode

Inside `test_particles/dipoleB_testparticles.py`, add a new `elif` block to `load_params()`. The physics parameters control the simulation; the plotting parameters control visualization only and do not affect the trajectory data or h5 file creation.

```python
elif run == "my_run":
    output_folder = "outputs/my_analysis"
    os.makedirs(output_folder, exist_ok=True)

    READ_DATA  = True     # load from cache if a matching h5 exists
    WRITE_DATA = True     # save trajectory to h5

    # --- integrator selection ---
    USE_RK45 = False      # include RK45 comparison
    USE_RK4  = False      # include RK4 comparison
    USE_RKG  = False      # include RKG comparison (protons only)
    USE_PS   = True       # Parker-Sochacki integrator
    PS_decimate = 1       # save every Nth step (1 = all)
    PS_CHUNKING = True    # stream to disk in chunks

    # --- physics parameters ---
    pitch_deg   = npfloat(60.0)       # equatorial pitch angle (degrees)
    phi_deg     = npfloat(0.0)        # initial gyrophase (degrees)
    x_initial   = npfloat(3.0)       # launch L-shell (R_E)
    y_initial   = npfloat(0)
    z_initial   = npfloat(0)
    KE_particle = npfloat(10e6)       # kinetic energy (eV)
    mass_si     = m_p                 # m_p for protons, m_e for electrons

    T_gyro = 2.0 * np.pi * (x_initial**3)

    N_STEPS_PER_GYRO_rk4 = 65
    N_STEPS_PER_GYRO_ps  = 65
    N_STEPS_PER_GYRO_rkg = 65
    rk4_step = npfloat(round(T_gyro / N_STEPS_PER_GYRO_rk4, 1))
    ps_step  = npfloat(round(T_gyro / N_STEPS_PER_GYRO_ps, 1))
    rkg_step = npfloat(round(T_gyro / N_STEPS_PER_GYRO_rkg, 1))

    gyroperiods = 1e4
    norm_time   = npfloat(gyroperiods) * T_gyro

    # --- plotting parameters (do not affect h5 data) ---
    USE_PLOT_TITLES = True
    USE_FULL_PLOT   = True
    window_time  = npfloat(11.6)
    slice_mode   = "last"
    N_GYRO       = 50
    gyro_window  = "last"

    USE_EXTERNAL_H5_ps   = False
    USE_EXTERNAL_H5_rk4  = False
    USE_EXTERNAL_H5_rk45 = False
    USE_EXTERNAL_H5_rkg  = False
    external_h5_ps   = "outputs_rawdata/"
    external_h5_rk4  = "outputs_rawdata/"
    external_h5_rk45 = "outputs_rawdata/"
    external_h5_rkg  = "outputs_rawdata/"
```

Then execute:

```bash
python dipoleB.py my_run
```

To enable adaptive stepping (recommended for high energy or low pitch angle), set `USE_ADAPTIVE = True` in your run mode's parameter dictionary. The adaptive stepper will automatically adjust the time step based on the local magnetic field strength.

### Key Parameters

The h5 file identity is determined by the physics parameters (energy, position, pitch angle, mass, step size). Changing only plotting parameters will reuse the same cached h5 file. The main parameters to adjust are `KE_particle` (in eV), `x_initial` (the L-shell in R_E), `pitch_deg`, `mass_si` (use `m_p` for protons or `m_e` for electrons), and `gyroperiods` (duration of the simulation in units of the characteristic gyroperiod at the equator).

Setting `WRITE_DATA = True` saves the trajectory to `outputs/outputs_rawdata/`. Setting `READ_DATA = True` will load from a previously saved h5 if one exists with matching parameters, skipping re-integration. Set `USE_FULL_PLOT = True` to generate all diagnostic plots including trajectory slices, Poincaré sections, and Dragt invariant monitoring.

---

## Dragt–Störmer Analysis (In Progress)

This section describes ongoing thesis work extending the PS dipole integrator to study the **Dragt–Störmer problem**: the full Lorentz-orbit dynamics of high-energy trapped particles in an axisymmetric dipole magnetic field, where the guiding-center approximation (GCA) breaks down.

### Background

Dragt (1965) showed that charged-particle trapping in a dipole is governed by two exact invariants - total energy *E* and canonical angular momentum *P*_φ - independent of adiabatic invariance. The dimensionless energy *W*₀² and the trapping boundary *W*₀² < *P*_φ⁴/16 determine whether an orbit is topologically confined, even when the adiabaticity parameter ε = *r*_g · |∇*B*|/*B* exceeds unity and the first adiabatic invariant μ is no longer conserved.

Michel (1980) extended this analysis to show that the phase-space topology of trapped particles contains islands of permanent stability (KAM islands) around elliptic fixed points, explaining why particles with high pitch angles near 90° can remain trapped indefinitely even in the non-adiabatic regime.

### Adaptive PS Time Stepping

`dipoleB.py` supports both fixed-step and adaptive PS integration, selected by the `USE_ADAPTIVE` flag in each run mode. The adaptive integrator (`functions_library_dipole_adp.py`) uses a hybrid approach: a fast path with fixed time step for regions where the PS series converges easily, and a slow path that computes the local gyroperiod from |B| and subdivides accordingly (200 steps per local gyroperiod) when the field gradient is steep. The integrator automatically switches between paths based on PS series order convergence.

Key features include NaN/Inf guards (rollback to pre-chunk state if the auxiliary tether variables diverge), atmosphere impact detection (flags when the orbit radius drops below 1 R_E), and streaming-to-disk via HDF5. Both integrators save only position, velocity, and B-field components to h5 (9 channels), omitting the 8 internal PS auxiliary variables (r², a, b, c, d, e, f, g) that are only needed during integration. This reduces file sizes by ~47% compared to saving the full 17-element state vector.

### Dragt Diagnostics

`dipoleB.py` computes and logs the following Dragt diagnostics for each simulation run:

- *W*₀² (dimensionless energy) and *P*_φ (canonical angular momentum) from initial conditions
- Trapping boundary status: CLOSED (*W*₀² < *P*_φ⁴/16) or OPEN
- Orbit character: REGULAR (*W*₀² < 0.012μ²) or CHAOTIC, following Dragt (1965) eq. 6.1
- Adiabaticity parameter ε(t) - initial, mean, and maximum values
- Atmosphere impact flag (orbit radius < 1 R_E)
- Dragt Poincaré surface of section at z = 0 (ρ vs ρ̇ in Dragt dimensionless units)
- Meridian-plane trajectory (ρ vs z in Dragt units)
- Gyrophase vs magnetic moment scatter at equatorial crossings

These diagnostics are written to each run's summary `.txt` file and appended to `master_simulation_log.csv` for cross-run analysis.

The diagnostic functions live in `functions/functions_library_dragt.py`, which provides `compute_dragt_params`, `compute_dragt_boundary`, `compute_z_crossings`, `compute_gyrophase_mu`, `calculate_adiabaticity`, and the `DragtMonitor` class for per-step conservation monitoring during integration.

> **Note on units:** The canonical angular momentum conservation plots use the paper's dimensionless normalization (position in R_E, velocity normalized to initial speed), not Dragt's dimensionless unit system. The Poincaré surfaces of section and meridian-plane trajectories are in Dragt units.

### Walt Stability Diagrams

`walt_diagnostics_v3.py` generates ε vs L-shell stability diagrams from `master_simulation_log.csv`, with three horizontal zone bands (trapped 20+ years, trapped but hits atmosphere, untrapped) and Dragt REGULAR/CHAOTIC point classification. Energy series are connected by faint lines and labeled. Zone boundaries auto-adjust from the data.

```bash
python utility_scripts/walt_diagnostics_v3.py master_simulation_log.csv
```

### Unit Conversion

`utility_scripts/si_to_dragt.py` converts SI initial conditions (energy in keV, L-shell, pitch angle, gyrophase) to the Dragt dimensionless system and back, handling the degenerate φ = 0 case with a linear-equation branch. Roundtrip tests are in `utility_scripts/test_dragt_roundtrip.py` and `utility_scripts/test_dragt_thorough.py`.

### Prototype Scripts

The `utility_scripts/` directory also contains several prototype scripts for ongoing research that do not affect the core simulation code. These include a batch parameter sweep runner (`batch_flux_runner.py`), a flux map builder (`flux_map_builder.py`), and a Michel phase portrait generator (`michel_phase_portrait.py`). See `USER_MANUAL_batch_tools.txt` for usage details. These scripts are under active development and may change.

### References

- Dragt, A. J. (1965). Trapped orbits in a magnetic dipole field. *Reviews of Geophysics*, 3(2), 255–298.
- Dragt, A. J., & Finn, J. M. (1976). Insolubility of trapped particle motion in a magnetic dipole field. *Journal of Geophysical Research*, 81(13), 2327–2340.
- Michel, F. C. (1980). Permanent magnetic trapping. *Journal of Geophysical Research*, 85(A2), 557–562.

---

## Citation
If you use this code or build upon it in your research, please cite:

> H. Jiles and R. Weigel, *”High-Accuracy Numerical Solutions of Particle Motion in Static Magnetic Fields,”* 2026. [arXiv:2604.20876](https://doi.org/10.48550/arXiv.2604.20876)

