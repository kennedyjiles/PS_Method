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
    PS_dipoleB, 
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
    mirror_times_from_PS, 
    bounce_summary, 
    drift_period_from_PS,
    get_run_params, 
    h5_path_for, 
    save_results_h5, 
    load_results_h5,
    summarize_error, 
    run_ps_streaming_with_decimation, 
    build_figure_filename,
    process_bounce_and_drift_chunk, 
    init_bounce_stream_state, 
    finalize_drift_stream
)

from logger_util import setup_logger
