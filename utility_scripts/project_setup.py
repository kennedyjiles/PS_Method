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
import test_particles.dipoleB_testparticles as tp
builtins.npfloat = np.float128 if tp.USE_FLOAT128 else np.float64
from test_particles.dipoleB_testparticles import *

# === Functions ===
from functions.functions_library_universal_chunk import (
    rk4_fixed_step, plt_config, sparse_labels, data_to_fig
)

from functions.functions_library_dipole import (
    lorentz_force_dipole,
    compute_mu_ps,
    compute_mu_rk,
    vector_potential_dipole,
    rkgl4_hamiltonian,
    hamiltonian_rhs,
    summarize,
    slice_solution,
    append_results_h5,
    compute_energy_ps_chunked,
    load_legacy_file,
    init_drift_stream_state,
    write_dict,
    bounce_summary,
    get_run_params,
    h5_path_for,
    save_results_h5,
    load_results_h5,
    summarize_error,
    run_ps_streaming_with_decimation,
    build_figure_filename,
    process_bounce_and_drift_chunk,
    init_bounce_stream_state,
    check_time_grids,
    finalize_drift_stream,
    expand_h5_to_full,
    compute_pphi_error_chunked,
)

from functions.functions_library_dragt import (
    calculate_adiabaticity,
    compute_dragt_params,
    compute_dragt_boundary,
    compute_z_crossings,
    compute_gyrophase_mu,
    dragt_analysis_chunked,
    DragtMonitor,
)

from functions.functions_library_dipole_adp import run_ps_streaming_adaptive

from functions.dipoleB_plots import (
    plot_full_2d, plot_full_3d, plot_slice_2d, plot_slice_3d,
    plot_ke_error,
    plot_dragt_poincare, plot_gyrophase_mu, plot_polar_phase_space,
    plot_meridian_plane, plot_adiabaticity, plot_pphi_error,
    plot_mu_deviation,
)

from functions.dipoleB_writers import write_summary_txt, write_master_csv

from .logger_util import setup_logger

from configs.config_loader import load_config