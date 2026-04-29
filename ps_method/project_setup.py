# === Standard Library ===
import os
import sys
import time
import json
import gc
import logging
import tracemalloc
from datetime import datetime

# === Environment Setup ===
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

# === Third-party Libraries ===
import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from matplotlib.ticker import (
    LogLocator, LogFormatterSciNotation, NullFormatter, FuncFormatter
)

# === Project-Specific ===
import builtins
from ps_method.constants import q_e, m_e, m_p, evtoj, spdlight, RE, B_0

# --- System-level settings (hardcoded, not per-run) ---
USE_FLOAT128    = False                                          # RKG will be disabled if True
npfloat         = np.float128 if USE_FLOAT128 else np.float64
builtins.npfloat = npfloat                                       # make available to functions library
tol             = 1.0 * np.finfo(npfloat).eps                    # machine epsilon, scaled by tau_0 later
MAX_PLOT_POINTS = 1_000_000                                      # cap points per graph

# --- Defaults for per-run settings (overridden by YAML config) ---
PS_order        = 40
PS_chunk_steps  = int(1e4)
rtol_rk45       = 1e-8
atol_rk45       = 1e-10
user_min_phase  = npfloat(0.1)

# --- System-level constants (unlikely to need per-run tuning) ---
CACHE_VELOCITY_RTOL = 0.005       # relative tolerance for cache velocity mismatch warning
PLOT_BOUNDARY_PAD   = 1.1         # padding multiplier for trajectory plot boundaries

# --- Matplotlib backend settings ---
plt.rcParams['agg.path.chunksize'] = 100000 if USE_FLOAT128 else 1000


# === Functions ===
from ps_method.universal import (
    rk4_fixed_step, plt_config, sparse_labels, data_to_fig
)

from ps_method.dipole_physics import (
    lorentz_force_dipole,
    compute_mu_ps,
    compute_mu_rk,
    vector_potential_dipole,
    rkgl4_hamiltonian,
    hamiltonian_rhs,
    slice_solution,
    compute_energy_ps_chunked,
    init_drift_stream_state,
    bounce_summary,
    run_ps_streaming_with_decimation,
    process_bounce_and_drift_chunk,
    init_bounce_stream_state,
    check_time_grids,
    finalize_drift_stream,
    compute_pphi_error_chunked,
    compute_mu_deviation_rk,
    compute_mu_deviation_ps,
)

from ps_method.writers import (
    _to_serializable,
    run_hash,
    h5_path_for,
    write_dict,
    summarize_error,
    summarize,
    get_run_params_dipole as get_run_params,
    build_run_stem,
    build_figure_filename,
    expand_h5_to_full,
    save_results_h5_dipole as save_results_h5,
    load_results_h5_dipole as load_results_h5,
    append_results_h5_dipole as append_results_h5,
    load_legacy_file,
)

from ps_method.dragt_physics import (
    calculate_adiabaticity,
    compute_dragt_params,
    compute_dragt_boundary,
    compute_z_crossings,
    compute_gyrophase_mu,
    dragt_analysis_chunked,
    DragtMonitor,
)

from ps_method.dipole_adaptive import run_ps_streaming_adaptive

from ps_method.dipole_plots import (
    plot_full_2d, plot_full_3d, plot_slice_2d, plot_slice_3d,
    plot_ke_error,
    plot_dragt_poincare, plot_gyrophase_mu, plot_polar_phase_space,
    plot_meridian_plane, plot_adiabaticity, plot_pphi_error,
    plot_mu_deviation,
)

from ps_method.writers import write_summary_txt, write_master_csv

from ps_method.universal import setup_logger

from configs.config_loader import load_config, compute_derived_dipole as compute_derived