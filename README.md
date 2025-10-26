Parker-Sochacki Method — Particle Simulation Utilities
=====================================================

Summary
-------
This repository contains Python code and supporting utilities for simulating charged-particle motion using the Parker–Sochacki Method. It includes example simulation drivers for different magnetic field geometries, a set of reusable function libraries, test particle scripts and plotting helpers.

Repository layout
-----------------
- `constB.py`, `dipoleB.py`, `hyperB.py` — top-level simulation drivers for constant, dipole and hyperbolic magnetic-field configurations.
- `inspect_hdf5.py` — helper to inspect HDF5 output files (if used by simulations).
- `master_simulation_log.csv` — CSV log used to record simulation runs and parameters.
- `functions/` — reusable function libraries used across simulations:
  - `functions_library_constB.py`
  - `functions_library_dipole.py`
  - `functions_library_hyper.py`
  - `functions_library_universal.py`
- `test_particles/` — example/test particle scripts and small test drivers (includes archived example particle scripts).
- `misc_plots/` — scripts and generated plots (example outputs, CSVs and plotting helpers).

Requirements
------------
- Python 3.9+ (project contains .pyc files from CPython 3.9, but newer 3.x versions should work)
- Typical scientific Python stack (install with pip):
  - numpy
  - scipy
  - matplotlib
  - pandas
  - h5py (optional — only if using HDF5 output/inspection)

Install (recommended virtualenv)
-------------------------------
1. Create and activate a virtual environment:

   python3 -m venv .venv
   source .venv/bin/activate

2. Install dependencies (example):

   pip install numpy scipy matplotlib pandas h5py

Running simulations
-------------------
- To run one of the main simulation drivers from the repository root:

  python constB.py
  python dipoleB.py
  python hyperB.py

- See the top of each driver for configurable parameters (initial conditions, timesteps, output paths).

Testing and example particles
-----------------------------
- The `test_particles/` folder contains lightweight scripts for exercising the integrators and particle setup. Each script includes two modes:

  - Demo mode: quick, minimal examples that typically run in a couple of seconds and produce a small set of outputs useful for verifying the environment and visualizing a short trajectory.

  - Paper mode: configuration that reproduces the longer runs and parameter choices used in the paper; these runs can take substantially longer and generate the datasets and figures used in published results.

- How to choose a mode:
  - Many scripts accept a command-line argument (e.g. `demo` or `paper`) or expose a top-level `MODE` or configuration variable — check the header of the specific script for details. Example usages (if supported by the script):

    - Quick demo: `python test_particles/constB_testparticles.py demo`
    - Reproduce paper results: `python test_particles/constB_testparticles.py paper`

  - If the script does not accept arguments, open the script and set the small configuration block at the top to `demo` or `paper`, then run the script normally.

- Recommendation: use demo mode to confirm the environment and get fast visual feedback. Use paper mode when you need to reproduce published figures or generate the full datasets; allocate more time or run on a dedicated compute node for paper-mode runs.

- To change the physical parameters or variables for a simulation (such as initial positions, velocities, field strengths, or integration settings), edit the relevant values directly in the corresponding script within the `test_particles/` directory. Each test particle file contains a configuration section near the top where these parameters can be adjusted to explore different scenarios or reproduce specific results.

Plots and analysis
------------------
- The main simulation codes will automatically generate output folders containing plots and data relevant to each run. These folders are created in the working directory and include figures such as particle trajectories, kinetic energy error, and other diagnostics for the specific simulation parameters used.

- The `misc_plots/` directory contains additional plotting scripts and example outputs. These are especially useful for reproducing or complementing figures from the paper, with a particular focus on the dipole simulations and the collection of substation runs. Use these scripts to further analyze, compare, or visualize results beyond the standard outputs of the main codes.

- If you are interested in the substation runs or want to explore the broader set of results used in the paper, start by reviewing the scripts and CSVs in `misc_plots/`.

Project notes
-------------
- The core numerical helpers are in `functions/`. Split across files by field-geometry so the drivers remain small.
- This repository appears to have been developed for research/educational purposes. Back up any important data before running long simulations.

Contact
-------
For questions about usage or development, open an issue in the repository or contact the project owner.

(Generated README — edit to add project-specific instructions, parameter descriptions and usage examples.)
