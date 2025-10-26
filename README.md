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
- The `test_particles/` folder contains lightweight scripts for exercising the integrators and particle setup. Run them directly to follow simple example trajectories:

  python test_particles/constB_testparticles.py
  python test_particles/dipoleB_testparticles.py
  python test_particles/hyperB_testparticles.py

Plots and analysis
------------------
- Example plotting utilities and sample outputs are stored in `misc_plots/`. Use these scripts as a starting point to visualize energies, trajectories, and other diagnostics.

Project notes
-------------
- The core numerical helpers are in `functions/`. Split across files by field-geometry so the drivers remain small.
- This repository appears to have been developed for research/educational purposes. Back up any important data before running long simulations.

Contributing
------------
- Open an issue or submit a pull request with focused changes. Include a short description and, if appropriate, a small example demonstrating the change.

License
-------
No license file is included in the repository. If you plan to distribute or publish code derived from this project, add a LICENSE file or consult the original author for licensing terms.

Contact
-------
For questions about usage or development, open an issue in the repository or contact the project owner.

(Generated README — edit to add project-specific instructions, parameter descriptions and usage examples.)
