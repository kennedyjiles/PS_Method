"""
dipoleb.py — Main driver for charged particle trajectory simulation in a
             magnetic dipole field using the power series (PS) method with
             optional RK4, RK45, and RKG (symplectic) solvers for comparison.

Usage:
    python dipoleb.py                       # runs the default config (demo)
    python dipoleb.py demo                  # named config  → configs/dipoleb/demo.yml
    python dipoleb.py paper1                # named config  → configs/dipoleb/paper1.yml
    python dipoleb.py configs/dipoleb/my_run.yml   # direct path to a custom YAML config

Available named configs:
    demo, paper1, paper2, paper3, dragt, walt, monster_ps, manual, testrun

To create a custom run:
    1. Copy configs/dipoleb/base.yml to configs/dipoleb/my_run.yml
    2. Edit the parameters you want to change (energy, pitch, x_initial, etc.)
    3. Run:  python dipoleb.py my_run

Your config is automatically merged with base.yml — any parameter you don't
specify falls back to the default value. Do NOT edit base.yml directly; it
serves as the reference for all runs.
"""

import numpy as np
import builtins
import os
import sys
import time
import json
import logging
import tracemalloc
from types import SimpleNamespace

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

import h5py
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from ps_method.constants import q_e, m_e, m_p, evtoj, spdlight, RE, B_0
from configs.config_loader import load_config, compute_derived_dipoleb as compute_derived, copy_config_to_output


def main(cfg_path, replot=False):
    """Run a dipole-B simulation from a YAML config file path.

    Parameters
    ----------
    cfg_path : str – path to a YAML config file.
    replot   : bool – if True, force READ_DATA=True (skip solvers,
               regenerate plots from cached h5 data).
    """

    cfg        = load_config(cfg_path)

    # --- Resolve float type BEFORE importing physics modules so @maybe_njit
    #     sees the correct type (float128 skips njit, float64 compiles). ---
    USE_FLOAT128 = cfg.get("use_float128", False)
    npfloat = np.float128 if USE_FLOAT128 else np.float64
    builtins.npfloat = npfloat
    tol = 1.0 * np.finfo(npfloat).eps
    plt.rcParams['agg.path.chunksize'] = 100000 if USE_FLOAT128 else 1000

    # --- Import physics modules AFTER builtins.npfloat is set ---
    from ps_method import dipoleb_physics as dp
    from ps_method import dipoleb_moment_analysis as mp
    from ps_method import dipoleb_bouncedrift_analysis as bd
    from ps_method import dipoleb_dragt_analysis as df
    from ps_method import dipoleb_energy_analysis as ea
    from ps_method import dipoleb_debug as dbg
    from ps_method import dipoleb_plots as dplt
    from ps_method import writers as wr
    from ps_method import utils as ul
    from ps_method.dipoleb_adaptive import run_ps_streaming_adaptive

    # B_0 reassigned in cache-reload branches — provide unconditional initial
    # assignment so reads before the conditional branches don't hit UnboundLocalError.
    from ps_method.constants import B_0

    DEBUG = False # WARNING: Adds computation time. TURN OFF FOR LONG RUNS
    if DEBUG:
        logger = dbg.setup_logger("dipole_logger", "dipoleb.log", level=logging.DEBUG) #This logger will log to a file in the working directory, it will overwrite each run unless you change the filename
        tracemalloc.start()

    params     = compute_derived(cfg, npfloat=npfloat)

    # =========================================================
    # ============= Assign YML file parameters ================
    # =========================================================

    # --- Always needed (from _defaults + every run mode) ---
    READ_DATA       = params["READ_DATA"]
    if replot:
        READ_DATA = True
    USE_RK45        = params["USE_RK45"]
    USE_RK4         = params["USE_RK4"]
    USE_RKG         = params["USE_RKG"]
    USE_PS          = params["USE_PS"]
    USE_ADAPTIVE    = params["USE_ADAPTIVE"]
    PS_decimate     = params["PS_decimate"]
    y_initial       = params["y_initial"]
    z_initial       = params["z_initial"]
    USE_PLOT_TITLES = params["USE_PLOT_TITLES"]
    USE_FULL_PLOT   = params["USE_FULL_PLOT"]
    slice_mode      = params["slice_mode"]
    gyro_window     = params["gyro_window"]
    output_folder   = params["output_folder"]
    run_storage     = params["run_storage"]
    window_time     = params["window_time"]
    N_GYRO          = params["N_GYRO"]

    USE_EXTERNAL_H5_ps   = params["USE_EXTERNAL_H5_ps"]
    USE_EXTERNAL_H5_rk4  = params["USE_EXTERNAL_H5_rk4"]
    USE_EXTERNAL_H5_rk45 = params["USE_EXTERNAL_H5_rk45"]
    USE_EXTERNAL_H5_rkg  = params["USE_EXTERNAL_H5_rkg"]
    external_h5_ps       = params["external_h5_ps"]
    external_h5_rk4      = params["external_h5_rk4"]
    external_h5_rk45     = params["external_h5_rk45"]
    external_h5_rkg      = params["external_h5_rkg"]

    # --- Physics (set by all modes except manual, which load from h5) ---
    pitch_deg    = params.get("pitch_deg",    None)
    phi_deg      = params.get("phi_deg",      None)
    x_initial    = params.get("x_initial",    None)
    KE_particle  = params.get("KE_particle",  None)
    mass_si      = params.get("mass_si",      None)
    T_gyro       = params.get("T_gyro",       None)
    gyroperiods  = params.get("gyroperiods",  None)
    norm_time    = params.get("norm_time",    None)

    # --- Step sizes (set by all modes except legacy/manual) ---
    ps_step              = params.get("ps_step",  None)
    rk4_step             = params.get("rk4_step", None)
    rkg_step             = params.get("rkg_step", None)
    N_STEPS_PER_GYRO_ps  = params.get("N_STEPS_PER_GYRO_ps",  None)
    N_STEPS_PER_GYRO_rk4 = params.get("N_STEPS_PER_GYRO_rk4", None)
    N_STEPS_PER_GYRO_rkg = params.get("N_STEPS_PER_GYRO_rkg", None)

    # --- Optional overrides (only some modes set these) ---
    # compute_derived always populates these keys with its own defaults,
    # so no module-level fallback is needed.
    PS_order        = params["PS_order"]
    PS_chunk_steps  = params["PS_chunk_steps"]
    rtol_rk45       = params["rtol_rk45"]
    atol_rk45       = params["atol_rk45"]
    user_min_phase  = params["user_min_phase"]
    MAX_PLOT_POINTS_local = params.get("MAX_PLOT_POINTS", 1_000_000)
    CACHE_VELOCITY_RTOL   = params.get("CACHE_VELOCITY_RTOL", 0.005)
    PLOT_BOUNDARY_PAD     = params.get("PLOT_BOUNDARY_PAD", 1.1)
    manual_h5_path  = params.get("manual_h5_path", None)

    # --- Adaptive PS settings ---
    ps_adaptive = params.get("ps_adaptive", {})

    # --- Dragt monitor ---
    dragt_monitor_rtol = params.get("dragt_monitor_rtol", 1e-4)

    # --- Bounce/drift detection ---
    bounce_drift_cfg        = params.get("bounce_drift", {})
    velocity_epsilon_scale  = bounce_drift_cfg.get("velocity_epsilon_scale", 1e-14)
    min_gap_steps           = bounce_drift_cfg.get("min_gap_steps", 3)
    gap_gyro_fraction       = bounce_drift_cfg.get("gap_gyro_fraction", 0.5)

    # === Misc Odds and Ends ===
    PS_CHUNKING = True     # PS data always streamed to disk in chunks (no in-memory option)
    WRITE_DATA  = True     # always write h5 (required by chunked streaming)
    ul.plt_config(scale=1)                        # config file for setting plot sizes and fonts (from Dr. W)
    os.makedirs(run_storage, exist_ok=True)    # ensures file for the storagae for raw data exists
    os.makedirs(output_folder, exist_ok=True)  # ensures file for the storagae for images and text file exists
    plt.ioff()                                 # turn off interactive mode for plots
    if USE_FLOAT128: USE_RKG = False


    # --- Safety defaults for variables assigned only inside conditional
    #     branches (cache-reload, solver-execution).  Ensures no
    #     UnboundLocalError regardless of which path is taken. ---
    solution_ps   = None
    solution_rk4  = None
    solution_rk45 = None
    solution_rkg  = None
    orders_used   = None
    y_rk45_common = None
    summary       = {}
    timing        = {}
    stem          = ""
    max_ps_value  = None
    steps_rk4     = None
    steps_rkg     = None

    # ===============================================
    # ============= Manual File Load ================
    # ===============================================

    USE_MANUAL_FILE = manual_h5_path is not None and os.path.exists(manual_h5_path)
    if USE_MANUAL_FILE:
        cache_path = manual_h5_path
        print(f"You have manually selected a file: {cache_path}\n")
        if os.path.exists(cache_path):
            print(f"Found existing results: {os.path.basename(cache_path)} — loading.\n")
            with h5py.File(cache_path, "r") as cached:
                # creating summary dictionary, legacy files should create a similar dictionary now
                if "summary_json" not in cached.attrs:
                    raise RuntimeError(
                        "Cached file missing summary_json. "
                        "This file was written by an older version."
                    )

                summary = json.loads(cached.attrs["summary_json"])

                # ---- meta ----
                meta = summary["meta"]
                timing = meta["timing"]

                stem = meta["stem"]
                mass_si = summary["meta"]["mass_si"]
                particle_type = meta["particle"]
                KE_particle = meta["energy_eV"]
                pitch_deg = meta["pitch_deg"]
                phi_deg = meta["phi_deg"]
                x_initial = meta["x0"]
                y_initial = meta["y0"]
                z_initial = meta["z0"]
                B_0 = meta["B0_T"]
                gyroperiods = meta["gyroperiods"]
                norm_time = meta["norm_time"]
                npfloat = np.dtype(meta["dtype"]).type  # optional

                T_gyro = meta.get("T_gyro", 2.0 * np.pi * (x_initial**3))  # fallback for older h5 files

                # ---- PS config ----
                ps_cfg = summary["ps"]
                USE_PS = ps_cfg["enabled"]
                PS_CHUNKING = ps_cfg["streaming"]
                ps_step = ps_cfg["dt"]
                steps_ps = ps_cfg["steps"]
                PS_decimate = ps_cfg["decimate"]
                PS_chunk_steps = ps_cfg["chunksize"]
                N_STEPS_PER_GYRO_ps = ps_cfg["numberstepspergyro"]
                max_ps_value = ps_cfg["max_ps"]
                E0_ps = ps_cfg["E0"]
                mu0_ps = ps_cfg["mu0"]

                # ---- RK4 config ----
                rk4_cfg = summary["rk4"]
                USE_RK4 = rk4_cfg["enabled"]
                rk4_step = rk4_cfg["dt"]
                steps_rk4 = rk4_cfg["steps"]
                N_STEPS_PER_GYRO_rk4 = rk4_cfg["numberstepspergyro"]


                # ---- RK45 config ----
                rk45_cfg = summary["rk45"]
                USE_RK45 = rk45_cfg["enabled"]
                rtol_rk45 = rk45_cfg["rtol"]
                atol_rk45 = rk45_cfg["atol"]

                # ---- RKG config ----
                rkg_cfg = summary["rkg"]
                USE_RKG = rkg_cfg["enabled"]
                rkg_step = rkg_cfg["dt"]
                steps_rkg = rkg_cfg["steps"]
                N_STEPS_PER_GYRO_rkg = rkg_cfg["numberstepspergyro"]


                # ---- Load solver data ------
                """
                Earlier editions of the code loaded everything into memory, for extended runs this has become untenable.
                Chunking allows files to be written and read in chunks which takes far less memory. Note, right now ONLY PS method
                does the chunking method. I have not tried to apply it to RK method until we find specific needs beccause it was a lot of work.
                """

                # === PS (chunked — data stays on disk, read in slices later) ===
                if USE_PS and "ps" in cached:
                    solution_ps = None
                    orders_used = None
                    # Trimmed files have fewer columns than the original run.
                    # Override steps_ps and norm_time so downstream windowing
                    # uses actual data size.
                    n_store_actual = cached["ps"]["y"].shape[1]
                    ps_store_stride = PS_decimate if PS_decimate > 1 else 1
                    steps_ps_actual = n_store_actual * ps_store_stride
                    if steps_ps_actual < steps_ps:
                        print(f"  Trimmed file detected: {n_store_actual:,} stored columns "
                              f"(original {steps_ps:,} steps → effective {steps_ps_actual:,})")
                        steps_ps = steps_ps_actual
                        norm_time = steps_ps * ps_step
                        gyroperiods = norm_time / (2.0 * np.pi)

                # === RK4 ===
                if USE_RK4 and "rk4" in cached:
                    solution_rk4 = cached["rk4"]["y"][()]

                # === RK45 ===
                if USE_RK45 and "rk45" in cached:
                    solution_rk45 = SimpleNamespace(t=cached["rk45"]["t"][()], y=cached["rk45"]["y"][()])

                # === RKG ===
                if USE_RKG and "rkg" in cached:
                    solution_rkg = cached["rkg"]["y"][()]

    # for file/plot naming
    if mass_si == m_e: particle_type = "Electron"
    elif mass_si == m_p: particle_type = "Proton"
    else: particle_type = "Particle"

    qoverm      = npfloat(-1) if mass_si == m_e else npfloat(1)
    charge_sign = float(qoverm)   # +1 proton, -1 electron (used in Dragt canonical momentum)

    # === Misc Conversions  ===
    KE_joules = KE_particle * evtoj                     # converting KE from eV to Joules
    gamma = 1.0 + KE_joules / (mass_si * spdlight**2)   # Lorentz factor
    mass = gamma * mass_si                              # Relativistic mass used for magnetic moment calculations

    v_si = spdlight * np.sqrt(1.0 - 1.0 / gamma**2)     # m/s
    tau_time = gamma * mass_si / (abs(q_e) * abs(B_0))  # this is tau0 from paper
    v_tau = v_si * tau_time / RE                        # dimensionless velocity

    physical_time = norm_time * abs(tau_time)           # actual physical time, t; normalized time =t/tau_time
    window_duration = window_time/tau_time              # converting window_time to dimensionless time
    tol_local = npfloat(tol) * tau_time                 # Scale tolerance by tau_0

    # === Velocity Config based on INput Angles ===
    pitch_rad = npfloat(np.radians(pitch_deg))              # degrees to radians, pitch
    phi_rad = npfloat(np.radians(phi_deg))                  # degrees to radians, phi
    v_par = npfloat(v_tau) * npfloat(np.cos(pitch_rad))     # parallel velocity component
    v_perp = npfloat(v_tau) * npfloat(np.sin(pitch_rad))    # perpendicular velocity component

    vx_initial = npfloat(v_perp * np.cos(phi_rad))
    vy_initial = npfloat(v_perp * np.sin(phi_rad))
    vz_initial = npfloat(v_par)

    # ===  cleaning small trig values to zero ===
    if abs(vx_initial) < (1.0 * np.finfo(npfloat).eps): vx_initial = npfloat(0.0)
    if abs(vy_initial) < (1.0 * np.finfo(npfloat).eps): vy_initial = npfloat(0.0)
    if abs(vz_initial) < (1.0 * np.finfo(npfloat).eps): vz_initial = npfloat(0.0)

    gyro_radius_si = (gamma * mass_si * v_si * np.sin(pitch_rad) / (np.abs(q_e) * (B_0 / x_initial**3)))
    gyro_radius_RE=float(gyro_radius_si/RE)
    initial_pos_vel = np.array([x_initial, y_initial, z_initial, vx_initial, vy_initial, vz_initial], dtype=npfloat)

    if DEBUG:
        logger.info("Starting chunked dipole run.")
        logger.debug(f"Initial velocity: {vx_initial}, {vy_initial}, {vz_initial}")
        logger.debug(f"Initial position: {x_initial}, {y_initial}, {z_initial}")
        logger.debug(f"Initial gyroradius: {gyro_radius_RE}")

    # --- Initial invariants for E0 and mu0 for h5 file ---
    """
    To streamline memory for large files, we are slicing out what we need
    directly from the h5 file, this just establishes the E0 and mu0 values for those calculations
    """
    vx0, vy0, vz0 = initial_pos_vel[3:6]
    E0_ps = npfloat(0.5) * (vx0*vx0 + vy0*vy0 + vz0*vz0)
    y0_ps = np.zeros((17, 1), dtype=npfloat)
    y0_ps[0:6, 0] = initial_pos_vel
    x0, y0, z0 = initial_pos_vel[0:3]
    r2 = x0*x0 + y0*y0 + z0*z0
    r5inv = r2**(-2.5)
    y0_ps[14, 0] = -3 * x0 * z0 * r5inv
    y0_ps[15, 0] = -3 * y0 * z0 * r5inv
    y0_ps[16, 0] = -(3*z0*z0 - r2) * r5inv
    mu0_ps = mp.compute_mu_ps(y0_ps, mass)[0]


    # === Build parameter tracer & check cache ===
    """
    This first part is scanning the files already stored in 'run_storage' based on input parameters (not specifically
    lodaded legacy files) in the yml to see if we already have the data. If it finds the data, it will
    load relevant parameters. If it does not find a file, it will start running the solvers to get the needed data.
    Beware that these files can be GB size for dipole.
    """
    if not USE_MANUAL_FILE:
        run_params = wr.get_run_params_dipoleb(USE_RK45, USE_RK4, USE_RKG, USE_PS, PS_decimate, PS_CHUNKING,   # parameters it is scanning
                        mass_si, q_e, B_0, gamma, user_min_phase,
                        x_initial, y_initial, z_initial,
                        pitch_deg, phi_deg,
                        norm_time, ps_step, rk4_step, rkg_step,
                        PS_order, tol_local, qoverm, rtol_rk45, atol_rk45)
        cache_path = wr.h5_path_for(run_params, run_storage)
        if os.path.exists(cache_path) and READ_DATA:
            print(f"Found existing results: {os.path.basename(cache_path)} — loading.\n")

            with h5py.File(cache_path, "r") as cached:

                # creating summary dictionary, legacy files should create a similar dictionary now
                if "summary_json" not in cached.attrs:
                    raise RuntimeError(
                        "Cached file missing summary_json. "
                        "This file was written by an older version."
                    )

                summary = json.loads(cached.attrs["summary_json"])

                # ---- meta ----
                meta = summary["meta"]
                timing = meta["timing"]

                stem = meta["stem"]
                particle_type = meta["particle"]
                KE_particle = meta["energy_eV"]
                pitch_deg = meta["pitch_deg"]
                phi_deg = meta["phi_deg"]
                x_initial = meta["x0"]
                y_initial = meta["y0"]
                z_initial = meta["z0"]
                B_0 = meta["B0_T"]
                gyroperiods = meta["gyroperiods"]
                norm_time = meta["norm_time"]
                npfloat = np.dtype(meta["dtype"]).type  # optional

                # ---- PS config ----
                ps_cfg = summary["ps"]
                USE_PS = ps_cfg["enabled"]
                PS_CHUNKING = ps_cfg["streaming"]
                ps_step = ps_cfg["dt"]
                steps_ps = ps_cfg["steps"]
                PS_decimate = ps_cfg["decimate"]
                PS_chunk_steps = ps_cfg["chunksize"]
                N_STEPS_PER_GYRO_ps = ps_cfg["numberstepspergyro"]
                max_ps_value = ps_cfg["max_ps"]
                E0_ps = ps_cfg["E0"]
                mu0_ps = ps_cfg["mu0"]

                # ---- RK4 config ----
                rk4_cfg = summary["rk4"]
                USE_RK4 = rk4_cfg["enabled"]
                rk4_step = rk4_cfg["dt"]
                steps_rk4 = rk4_cfg["steps"]
                N_STEPS_PER_GYRO_rk4 = rk4_cfg["numberstepspergyro"]


                # ---- RK45 config ----
                rk45_cfg = summary["rk45"]
                USE_RK45 = rk45_cfg["enabled"]
                rtol_rk45 = rk45_cfg["rtol"]
                atol_rk45 = rk45_cfg["atol"]

                # ---- RKG config ----
                rkg_cfg = summary["rkg"]
                USE_RKG = rkg_cfg["enabled"]
                rkg_step = rkg_cfg["dt"]
                steps_rkg = rkg_cfg["steps"]
                N_STEPS_PER_GYRO_rkg = rkg_cfg["numberstepspergyro"]


                # ---- Load solver data ------
                # === PS ===
                if USE_PS and "ps" in cached:
                    solution_ps = None
                    orders_used = None
                    # Guard against trimmed files with fewer columns than metadata claims
                    n_store_actual = cached["ps"]["y"].shape[1]
                    ps_store_stride = PS_decimate if PS_decimate > 1 else 1
                    steps_ps_actual = n_store_actual * ps_store_stride
                    if steps_ps_actual < steps_ps:
                        print(f"  Trimmed file detected: {n_store_actual:,} stored columns "
                              f"(original {steps_ps:,} steps → effective {steps_ps_actual:,})")
                        steps_ps = steps_ps_actual
                        norm_time = steps_ps * ps_step
                        gyroperiods = norm_time / (2.0 * np.pi)

                # === RK4 ===
                if USE_RK4 and "rk4" in cached:
                    solution_rk4 = cached["rk4"]["y"][()]

                # === RK45 ===
                if USE_RK45 and "rk45" in cached:
                    solution_rk45 = SimpleNamespace(t=cached["rk45"]["t"][()], y=cached["rk45"]["y"][()])

                # === RKG ===
                if USE_RKG and "rkg" in cached:
                    solution_rkg = cached["rkg"]["y"][()]
        else:
            print("No matching file or 'Read Data' skipped. Running solvers...\n")

            # Common grid size (used by RK45, PS needs to be enabled)
            steps_ps = int(norm_time / ps_step)
            # ====== Run PS ======
            max_ps = None
            if USE_PS:
                start_time_ps = time.time()

                # --- Dragt conservation monitor ---
                # Compute L-shell from initial canonical momentum (same logic as post-run)
                _rho_i = np.sqrt(x_initial**2 + y_initial**2)
                _vphi_i = (x_initial * vy_initial - y_initial * vx_initial) / _rho_i
                _Pphi_i = _rho_i * _vphi_i - charge_sign / _rho_i
                if charge_sign * _Pphi_i < 0:
                    _L_mon = float(-charge_sign / _Pphi_i)
                else:
                    _r_i = np.sqrt(x_initial**2 + y_initial**2 + z_initial**2)
                    _L_mon = float(_r_i**3 / _rho_i**2)
                dragt_mon = df.conservation_monitor(_L_mon, charge_sign,
                                         check_every=1, rtol=dragt_monitor_rtol)
                # ----------------------------------
                _stream_args = dict(
                    initial_pos_vel_ps=initial_pos_vel,
                    steps_ps=steps_ps,
                    ps_step=ps_step,
                    PS_order=PS_order,
                    tol=tol_local,
                    qoverm=qoverm,
                    E0_ps=E0_ps,
                    mu0_ps=mu0_ps,
                    cache_path=cache_path,
                    write_data=True,
                    chunk_steps=PS_chunk_steps,
                    decimate=PS_decimate,
                    N_STEPS_PER_GYRO_ps=N_STEPS_PER_GYRO_ps,
                    user_min_phase=user_min_phase,
                    dragt_monitor=dragt_mon,
                )
                if USE_ADAPTIVE:
                    _stream_args.update(
                        order_low=ps_adaptive.get("order_low", 50),
                        order_high=ps_adaptive.get("order_high", 300),
                        grow_factor=ps_adaptive.get("grow_factor", 1.5),
                        shrink_factor=ps_adaptive.get("shrink_factor", 0.5),
                        steps_per_local_gyro=ps_adaptive.get("steps_per_local_gyro", 200),
                        min_fast_path_N=ps_adaptive.get("min_fast_path_N", 100),
                    )
                    max_ps, elapsed_ps = run_ps_streaming_adaptive(**_stream_args)
                else:
                    max_ps, elapsed_ps = dp.run_ps_streaming_with_decimation(**_stream_args)
                dragt_mon.summary()
                solution_ps = None
                orders_used = None
                end_time_ps = time.time()

            # ====== Run RK45 ======
            if USE_RK45:
                start_time_rk45 = time.time()
                t_common = ps_step * np.arange(steps_ps + 1, dtype=npfloat)
                solution_rk45 = solve_ivp(
                    dp.lorentz_force,
                    (0.0, norm_time),
                    initial_pos_vel,
                    method="RK45",
                    args=(qoverm,),
                    t_eval=t_common,
                    rtol=rtol_rk45,
                    atol=atol_rk45,)
                end_time_rk45 = time.time()

            # ====== Run RK4 ======
            if USE_RK4:
                steps_rk4 = int(norm_time / rk4_step)
                start_time_rk4 = time.time()
                solution_rk4 = ul.rk4_fixed_step(
                    dp.lorentz_force,
                    initial_pos_vel,
                    rk4_step,
                    steps_rk4,
                    args=(qoverm,),)
                end_time_rk4 = time.time()

            # ====== Run RKG ======
            if USE_RKG:
                # === Symplectic Implementations =====
                r0 = np.array([x_initial, y_initial, z_initial], dtype=npfloat)   # already normalized RE units
                v_tau_vec = np.array([vx_initial, vy_initial, vz_initial], dtype=npfloat)

                A0 = dp.vector_potential(r0)
                p0 = v_tau_vec + A0
                y0 = np.concatenate((r0, p0))   # for Hamiltonian in RKG
                # y0 = np.concatenate((r0, v_tau_vec))  # for Lorentz force in RKG, used as a sanity check

                steps_rkg = int(norm_time / rkg_step)
                steps_rkg = max(1, steps_rkg)

                start_time_rkg = time.time()
                solution_rkg = dp.rkgl4_hamiltonian(
                    dp.hamiltonian_rhs,
                    y0,
                    rkg_step,
                    steps_rkg,
                    args=(qoverm,),
                )
                end_time_rkg = time.time()

            results = {
                "ps": None,
                "rk4": None,
                "rk45": None,
                "rkg": None,
                "meta": {
                    "timing": {},
                    "physical_time": float(physical_time),
                    "norm_time": float(norm_time),
                    "percent_c": float(v_si/spdlight),
                    "particle": particle_type,
                    "mass_si": mass_si,
                    "q_e": q_e,
                    "energy_eV": npfloat(KE_particle),
                    "pitch_deg": npfloat(pitch_deg),
                    "phi_deg": npfloat(phi_deg),
                    "x0": npfloat(x_initial),
                    "y0": npfloat(y_initial),
                    "z0": npfloat(z_initial),
                    "B0_T": npfloat(B_0),
                    "gyroperiods": npfloat(gyroperiods),
                    "tau0": npfloat(tau_time),
                    "dtype": npfloat.__name__,
                }
            }

            if USE_PS:
                max_ps_value = int(max_ps) if max_ps is not None else None
            else:
                max_ps_value = None

            results["ps"] = { "enabled": bool(USE_PS),}
            if USE_PS:
                results["ps"].update({
                    "y": None,
                    "orders": None,
                    "ordercap": PS_order,
                    "max_ps": max_ps_value,
                    "numberstepspergyro": N_STEPS_PER_GYRO_ps,
                    "dt": ps_step,
                    "steps": steps_ps,
                    "streaming": True,
                    "chunksize": PS_chunk_steps,
                    "decimate": PS_decimate,
                    "tol": tol_local,
                    "minphase" : user_min_phase,
                    "E0": float(E0_ps),
                    "mu0": float(mu0_ps),
                    "t0": 0.0,
                })
                results["meta"]["timing"]["ps"] = end_time_ps - start_time_ps

            results["rk4"] = { "enabled": bool(USE_RK4),}
            if USE_RK4:
                results["rk4"].update({
                    "y": solution_rk4,
                    "numberstepspergyro": N_STEPS_PER_GYRO_rk4,
                    "dt": npfloat(rk4_step),
                    "steps": int(steps_rk4),
                    "t0": 0.0,
                })
                results["meta"]["timing"]["rk4"] = end_time_rk4 - start_time_rk4

            results["rk45"] = { "enabled": bool(USE_RK45),}
            if USE_RK45:
                results["rk45"].update({
                    "y": solution_rk45.y,
                    "t": solution_rk45.t,
                    "rtol": rtol_rk45,
                    "atol": atol_rk45,
                })
                results["meta"]["timing"]["rk45"] = end_time_rk45 - start_time_rk45

            results["rkg"] = { "enabled": bool(USE_RKG),}
            if USE_RKG:
                results["rkg"].update({
                    "y": solution_rkg,
                    "numberstepspergyro": N_STEPS_PER_GYRO_rkg,
                    "dt": npfloat(rkg_step),
                    "steps": int(steps_rkg),
                    "t0": 0.0
                })
                results["meta"]["timing"]["rkg"] = end_time_rkg - start_time_rkg

            # =========================
            # ====== Save Results =====
            # =========================
            stem = os.path.splitext(os.path.basename(cache_path))[0]
            timing = results["meta"]["timing"]
            results["meta"]["stem"]=stem
            if WRITE_DATA:
                summary = {
                    "meta": {
                        "stem": stem,
                        "particle": particle_type,
                        "mass_si": mass_si,
                        "q_e": q_e,
                        "energy_eV": float(KE_particle),
                        "pitch_deg": float(pitch_deg),
                        "phi_deg": float(phi_deg),
                        "x0": float(x_initial),
                        "y0": float(y_initial),
                        "z0": float(z_initial),
                        "B0_T": float(B_0),
                        "gyroperiods": float(gyroperiods),
                        "norm_time": float(norm_time),
                        "physical_time": float(physical_time),
                        "percent_c": float(v_si/spdlight),
                        "qoverm": float(qoverm),
                        "dtype": npfloat.__name__,
                        "tau0": tau_time,
                        "T_gyro": float(T_gyro),
                        "timing": results["meta"]["timing"],
                    },
                    "ps": {
                        "enabled": USE_PS,
                        "dt": ps_step if USE_PS else None,
                        "steps": steps_ps if USE_PS else None,
                        "streaming": True if USE_PS else None,
                        "ordercap": PS_order if USE_PS else None,
                        "max_ps": max_ps_value,
                        "chunksize": PS_chunk_steps if USE_PS else None,
                        "decimate": PS_decimate if USE_PS else None,
                        "numberstepspergyro": N_STEPS_PER_GYRO_ps if USE_PS else None,
                        "E0": float(E0_ps) if USE_PS else None,
                        "mu0": float(mu0_ps) if USE_PS else None,
                        "minphase": user_min_phase if USE_PS else None,
                        "tol": float(tol_local)
                    },
                    "rk4": {
                        "enabled": USE_RK4,
                        "dt": float(rk4_step) if USE_RK4 else None,
                        "steps": int(steps_rk4) if USE_RK4 else None,
                        "numberstepspergyro": N_STEPS_PER_GYRO_rk4 if USE_RK4 else None,
                    },
                    "rk45": {
                        "enabled": USE_RK45,
                        "rtol": rtol_rk45 if USE_RK45 else None,
                        "atol": atol_rk45 if USE_RK45 else None,
                    },
                    "rkg": {
                        "enabled": USE_RKG,
                        "dt": float(rkg_step) if USE_RKG else None,
                        "steps": int(steps_rkg) if USE_RKG else None,
                        "numberstepspergyro": N_STEPS_PER_GYRO_rkg if USE_RKG else None,
                    },
                }

                # ====== h5 file creation =============
                if USE_PS:
                    wr.append_results_h5_dipoleb(cache_path, results, summary)
                    print(f"Updated streamed file → {os.path.basename(cache_path)}")
                else:
                    wr.save_results_h5_dipoleb(cache_path, results, summary)
                    print(f"Saved results → {os.path.basename(cache_path)}")

    if DEBUG:
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        logger.info(f"Peak memory usage for load/write h5: {peak / 1024**2:.2f} MB\n")
        logger.debug(dbg.check_time_grids(
        norm_time=norm_time,
        ps_step=ps_step if USE_PS else None,
        steps_ps=steps_ps if USE_PS else None,
        rk4_step=rk4_step if USE_RK4 else None,
        steps_rk4=steps_rk4 if USE_RK4 else None,
        rkg_step=rkg_step if USE_RKG else None,
        steps_rkg=steps_rkg if USE_RKG else None,
        rk45_t=solution_rk45.t if USE_RK45 else None,
    ))


    # ==================================
    # ==== Dictionary of run params ====
    # ==================================
    """
    These are the plotting parameters, these can be varied without impacting the h5 file or scanned parameters
    and are saved to the summary text file. They are not appended to the h5 file though and should not be as the
    raw date remains unchanged
    """


    summary["plot"] = {
        "trajwindow_s": window_time,
        "slicemode": slice_mode,
        "NGYRO" : N_GYRO,
        "gyroslice": gyro_window,
        "maxplotpoints": MAX_PLOT_POINTS_local,
        "externalps": external_h5_ps if USE_EXTERNAL_H5_ps else None,
        "externalrk4": external_h5_rk4 if USE_EXTERNAL_H5_rk4 else None,
        "externalrk45": external_h5_rk45 if USE_EXTERNAL_H5_rk45 else None,
        "externalrkg": external_h5_rkg if USE_EXTERNAL_H5_rkg else None,
    }

    # ===============================
    # Build RK45 solution on PS grid
    # ===============================
    """
    this is building RK45 time base for points we want on PS grid. Not meant for long runs
    as this can be a memory hog but rk45 is not great on long runs anyways
    """
    if USE_RK45 and not USE_PS:
        raise RuntimeError(
            "RK45 requires USE_PS=True in this workflow; it builds a grid to match PS."
        )

    if USE_RK45:
        y_rk45_common = solution_rk45.y

    # =====================================================
    # ============= Data Set Access for Stream ============
    # =====================================================
    tracemalloc.start()

    ps_order_label = None # for plotting later

    if USE_PS:
        with h5py.File(cache_path, "r") as ps_h5:
            ps_grp = ps_h5["ps"]
            n_ps = steps_ps
            stride = max(1, n_ps // MAX_PLOT_POINTS_local)
            ps_order_label = int(ps_grp.attrs["max_ps"])

            if USE_FULL_PLOT:
                ps_y_h5 = ps_grp["y"]
                x_ps_plot = ps_y_h5[0, ::stride]
                y_ps_plot = ps_y_h5[1, ::stride]
                z_ps_plot = ps_y_h5[2, ::stride]

    if DEBUG:
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        logger.debug(f"Data access for plottings: {peak / 1024**2:.2f} MB\n")


    print(f"\n{'='*60}")
    print(f"  Run Statistics")
    print(f"{'='*60}")
    # === Timing Summary ===
    print(f"Particle        : {KE_particle:.1e} eV {particle_type}")
    if USE_RK45 and "rk45" in timing:
        print(f"Run Time RK45   : {timing['rk45']:.2f} s")
    if USE_RK4 and "rk4" in timing:
        print(f"Run Time RK4    : {timing['rk4']:.2f} s")
    if USE_RKG and "rkg" in timing:
        print(f"Run Time RKG    : {timing['rkg']:.2f} s")
    if USE_PS and "ps" in timing:
        print(f"Run Time PS     : {timing['ps']:.2f} s")

    print(f"Norm Time       : {norm_time:.2e} ")
    print(f"Physical Time   : {physical_time:.2e} s")
    print(f"PS Orders       : max={ps_order_label}")
    print(f"% of c          : {100*v_si/spdlight:.8f}")

    if DEBUG:
        logger.debug(f"Norm Time: {norm_time:.2e} MB")
        logger.debug(f"Physical Time   : {physical_time:.2e} s")
        logger.debug(f"ps_step: {ps_step}, norm_time: {norm_time}, steps_ps: {steps_ps}")
        # logger.debug(f"t_common[0]: {t_common[0]}, t_common[-1]: {t_common[-1]}")
    print(f"{'='*60}")


    # === Create run-specific output subfolders ===
    # data/<config>/<run-hash>/figures/   ← plots
    # data/<config>/<run-hash>/           ← summary, config copy, log
    # data/<config>/_rawdata/             ← h5 trajectory files
    run_folder = os.path.join(output_folder, stem)
    fig_folder = os.path.join(run_folder, "figures")
    os.makedirs(fig_folder, exist_ok=True)

    # --- Redirect debug log to run folder ---
    if DEBUG:
        _log_path = os.path.join(run_folder, f"{stem}.log")
        dbg.redirect_logger(logger, _log_path)
        print(f"Debug log redirected to {_log_path}\n")

    # --- Copy config YAML to run folder (with git hash) ---
    copy_config_to_output(cfg_path, run_folder)

    # =====================================================
    # ============== Full Trajectory Plots ================
    # =====================================================
    plotbounds = x_initial + PLOT_BOUNDARY_PAD

    if USE_FULL_PLOT:
        _traj_common = dict(
            summary=summary, run_folder=fig_folder, stem=stem,
            particle_type=particle_type, plotbounds=plotbounds,
            ps_order_label=ps_order_label, USE_PLOT_TITLES=USE_PLOT_TITLES,
            USE_RK45=USE_RK45, USE_RK4=USE_RK4, USE_RKG=USE_RKG, USE_PS=USE_PS,
            solution_rk45=solution_rk45 if USE_RK45 else None,
            solution_rk4=solution_rk4 if USE_RK4 else None,
            solution_rkg=solution_rkg if USE_RKG else None,
            x_ps_plot=x_ps_plot if USE_PS else None,
            y_ps_plot=y_ps_plot if USE_PS else None,
        )
        dplt.full_2d(**_traj_common)
        dplt.full_3d(**_traj_common, z_ps_plot=z_ps_plot if USE_PS else None)

    # ========================================================================
    # ================ Creating Plot Window (slice of time) ==================
    # ========================================================================
    if DEBUG: tracemalloc.start()

    _sw = dplt.prepare_slice_window(
        slice_mode, window_duration, norm_time,
        USE_PS=USE_PS, cache_path=cache_path, ps_step=ps_step,
        steps_ps=steps_ps, PS_decimate=PS_decimate,
        MAX_PLOT_POINTS=MAX_PLOT_POINTS_local,
        USE_RK4=USE_RK4, solution_rk4=solution_rk4, rk4_step=rk4_step,
        USE_RKG=USE_RKG, solution_rkg=solution_rkg, rkg_step=rkg_step,
        USE_RK45=USE_RK45, y_rk45_common=y_rk45_common,
    )
    ps_x_slice   = _sw["ps_x_slice"]
    ps_y_slice   = _sw["ps_y_slice"]
    ps_z_slice   = _sw["ps_z_slice"]
    rk4_x_slice  = _sw["rk4_x_slice"]
    rk4_y_slice  = _sw["rk4_y_slice"]
    rk4_z_slice  = _sw["rk4_z_slice"]
    rkg_x_slice  = _sw["rkg_x_slice"]
    rkg_y_slice  = _sw["rkg_y_slice"]
    rkg_z_slice  = _sw["rkg_z_slice"]
    rk45_x_slice = _sw["rk45_x_slice"]
    rk45_y_slice = _sw["rk45_y_slice"]
    rk45_z_slice = _sw["rk45_z_slice"]
    if _sw["ps_order_label"] is not None:
        ps_order_label = _sw["ps_order_label"]

    if DEBUG:
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        logger.info(f"Peak memory usage for slice analysis: {peak / 1024**2:.2f} MB")



    # =====================================================
    # ================ Trajectory Slice Plots =============
    # =====================================================
    _slice_common = dict(
        summary=summary, run_folder=fig_folder, stem=stem,
        particle_type=particle_type, ps_order_label=ps_order_label,
        USE_PLOT_TITLES=USE_PLOT_TITLES,
        USE_RK45=USE_RK45, USE_RK4=USE_RK4, USE_RKG=USE_RKG, USE_PS=USE_PS,
        rk45_x_slice=rk45_x_slice if USE_RK45 else None,
        rk45_y_slice=rk45_y_slice if USE_RK45 else None,
        rk4_x_slice=rk4_x_slice if USE_RK4 else None,
        rk4_y_slice=rk4_y_slice if USE_RK4 else None,
        rkg_x_slice=rkg_x_slice if USE_RKG else None,
        rkg_y_slice=rkg_y_slice if USE_RKG else None,
        ps_x_slice=ps_x_slice if USE_PS else None,
        ps_y_slice=ps_y_slice if USE_PS else None,
    )

    if USE_FULL_PLOT:
        dplt.slice_2d(**_slice_common)

    dplt.slice_3d(
        **_slice_common, plotbounds=plotbounds,
        rk45_z_slice=rk45_z_slice if USE_RK45 else None,
        rk4_z_slice=rk4_z_slice if USE_RK4 else None,
        rkg_z_slice=rkg_z_slice if USE_RKG else None,
        ps_z_slice=ps_z_slice if USE_PS else None,
    )


    # =====================================================
    # ============== KE Relative Error Plot ===============
    # =====================================================
    if DEBUG: tracemalloc.start()

    _ke = ea.compute_ke_errors(
        T_gyro, n_ps=n_ps, MAX_PLOT_POINTS=MAX_PLOT_POINTS_local,
        USE_PS=USE_PS, cache_path=cache_path, ps_step=ps_step,
        PS_decimate=PS_decimate, E0_ps=E0_ps,
        USE_RK4=USE_RK4, solution_rk4=solution_rk4, rk4_step=rk4_step,
        USE_RKG=USE_RKG, solution_rkg=solution_rkg, rkg_step=rkg_step,
        USE_RK45=USE_RK45, y_rk45_common=y_rk45_common,
        USE_EXTERNAL_H5_ps=USE_EXTERNAL_H5_ps,   external_h5_ps=external_h5_ps,
        USE_EXTERNAL_H5_rk4=USE_EXTERNAL_H5_rk4, external_h5_rk4=external_h5_rk4,
        USE_EXTERNAL_H5_rk45=USE_EXTERNAL_H5_rk45, external_h5_rk45=external_h5_rk45,
        USE_EXTERNAL_H5_rkg=USE_EXTERNAL_H5_rkg,   external_h5_rkg=external_h5_rkg,
        vector_potential_func=dp.vector_potential,
        load_results_h5_func=wr.load_results_h5_dipoleb,
    )

    time_factor    = _ke["time_factor"]
    energy_stride  = _ke["energy_stride"]
    rel_drift_ps   = _ke["rel_drift_ps"]
    rel_drift_rk4  = _ke["rel_drift_rk4"]
    rel_drift_rk45 = _ke["rel_drift_rk45"]
    rel_drift_rkg  = _ke["rel_drift_rkg"]

    dplt.ke_error(
        summary=summary, run_folder=fig_folder, stem=stem,
        particle_type=particle_type, ps_order_label=ps_order_label,
        USE_PLOT_TITLES=USE_PLOT_TITLES, time_factor=time_factor, norm_time=norm_time,
        ps_data=_ke["ke_ps"], rk4_data=_ke["ke_rk4"],
        rk45_data=_ke["ke_rk45"], rkg_data=_ke["ke_rkg"],
        ext_ps_data=_ke["ke_ext_ps"], ext_rk4_data=_ke["ke_ext_rk4"],
        ext_rk45_data=_ke["ke_ext_rk45"], ext_rkg_data=_ke["ke_ext_rkg"],
    )

    if DEBUG:
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        logger.info(f"Peak memory usage for KE analysis: {peak / 1024**2:.2f} MB")
        if USE_PS and rel_drift_ps is not None:
            midpoint_ps = int(round(len(rel_drift_ps) / 2))
            logger.info(f"energy stride: {energy_stride}")
            logger.debug(f"[PS] E rel drift initial ={rel_drift_ps[0]:.2e}, E rel drift mid ={rel_drift_ps[midpoint_ps]:.2e}, E rel drift final ={rel_drift_ps[-1]:.2e}")
        for _lbl, _rd, _flag in [("RK4", rel_drift_rk4, USE_RK4),
                                  ("RKG", rel_drift_rkg, USE_RKG),
                                  ("RK45", rel_drift_rk45, USE_RK45)]:
            if _flag and _rd is not None:
                _mid = int(round(len(_rd) / 2))
                logger.debug(f"[{_lbl}] E rel drift initial ={_rd[0]:.2e}, E rel drift mid ={_rd[_mid]:.2e}, E rel drift final ={_rd[-1]:.2e}")

    # ==================================================
    # ======== Dragt Analysis + Poincaré Plots =========
    # ==================================================
    dragt_log, L_shell_dragt = df.run_section(
        x_initial, y_initial, z_initial,
        vx_initial, vy_initial, v_tau,
        charge_sign, gamma,
        USE_PS=USE_PS, cache_path=cache_path,
        ps_step=ps_step, time_factor=time_factor,
        CACHE_VELOCITY_RTOL=CACHE_VELOCITY_RTOL,
        fig_folder=fig_folder, stem=stem,
        poincare_func=dplt.poincare,
        gyrophase_mu_func=dplt.gyrophase_mu,
        polar_phase_space_func=dplt.polar_phase_space,
        meridian_plane_func=dplt.meridian_plane,
        adiabaticity_func=dplt.adiabaticity,
    )

    # =========================================================
    # PLOT RELATIVE ERROR OF CANONICAL ANGULAR MOMENTUM
    # =========================================================

    pphi = ea.compute_pphi_error_chunked(cache_path, initial_pos_vel, charge_sign, ps_step, time_factor)
    dplt.pphi_error(fig_folder, pphi["t_gyro"], pphi["rel_error_log"],
                    pphi["P_phi_initial"], pphi["max_err"], pphi["ylabel"], stem=stem)


    # ============================================================
    # ================ Magnetic Moment Deviations ================
    # ============================================================
    if DEBUG: tracemalloc.start()

    mu_rk4_result = mu_rkg_result = mu_rk45_result = mu_ps_result = None

    if USE_RK4:
        mu_rk4_result = mp.compute_mu_deviation_rk(
            solution_rk4, steps_rk4, rk4_step,
            N_GYRO, N_STEPS_PER_GYRO_rk4, mass, gyro_window, time_factor,
            solver_type="rk4")

    if USE_RKG:
        mu_rkg_result = mp.compute_mu_deviation_rk(
            solution_rkg, steps_rkg, rkg_step,
            N_GYRO, N_STEPS_PER_GYRO_rkg, mass, gyro_window, time_factor,
            solver_type="rkg")

    if USE_RK45:
        mu_rk45_result = mp.compute_mu_deviation_rk(
            y_rk45_common, steps_ps, ps_step,
            N_GYRO, N_STEPS_PER_GYRO_ps, mass, gyro_window, time_factor,
            solver_type="rk45")

    if USE_PS:
        mu_ps_result = mp.compute_mu_deviation_ps(
            cache_path, steps_ps, ps_step, PS_decimate,
            N_GYRO, N_STEPS_PER_GYRO_ps, mass, mu0_ps,
            gyro_window, time_factor, max_plot_points=MAX_PLOT_POINTS_local)
        ps_order_label = mu_ps_result["ps_order_label"]

    # --- Unpack mu0 values needed by the summary writer ---
    mu0_rk4  = mu_rk4_result["mu0"]  if mu_rk4_result  else None
    mu0_rkg  = mu_rkg_result["mu0"]  if mu_rkg_result  else None
    mu0_rk45 = mu_rk45_result["mu0"] if mu_rk45_result else None

    dplt.mu_deviation(
        summary, fig_folder, stem, particle_type, ps_order_label,
        USE_PLOT_TITLES,
        ps_data=(mu_ps_result["t"], mu_ps_result["mudrift_plot"]) if mu_ps_result else None,
        rk4_data=(mu_rk4_result["t"], mu_rk4_result["mudrift"]) if mu_rk4_result else None,
        rk45_data=(mu_rk45_result["t"], mu_rk45_result["mudrift"]) if mu_rk45_result else None,
        rkg_data=(mu_rkg_result["t"], mu_rkg_result["mudrift"]) if mu_rkg_result else None,
    )


    if DEBUG:
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        logger.info(f"Peak memory usage for moment analysis: {peak / 1024**2:.2f} MB")
        for _lbl, _res in [("PS", mu_ps_result), ("RK4", mu_rk4_result),
                            ("RKG", mu_rkg_result), ("RK45", mu_rk45_result)]:
            if _res is not None:
                _md = _res.get("mudrift", _res.get("mudrift_plot"))
                _mid = int(round(len(_md) / 2))
                logger.debug(f"[{_lbl}] mu rel drift initial={_md[0]:.2e}, mid={_md[_mid]:.2e}, final={_md[-1]:.2e}")


    # ===================================================
    # ================ Mirror and Drift  ================
    # ===================================================
    # Bounce and drift are only calculated for PS method, using chunked h5 streaming.
    bounce_results = None
    drift_results  = None

    if DEBUG: tracemalloc.start()

    print(f"\n{'='*60}")
    print(f"  Bounce/Drift Statistics")
    print(f"{'='*60}")

    if USE_PS:
        v_eps = npfloat(velocity_epsilon_scale) * v_tau
        user_min_gap = max(min_gap_steps, int(gap_gyro_fraction * T_gyro / ps_step))

        bounce_state = bd.init_bounce_stream_state()
        drift_state  = bd.init_drift_stream_state()

        ps_store_stride = PS_decimate if PS_decimate > 1 else 1
        dt_store = ps_step * ps_store_stride

        with h5py.File(cache_path, "r") as ps_h5:
            ps_y = ps_h5["ps"]["y"]
            n_store = ps_y.shape[1]

            for j0_chunk in range(0, n_store, PS_chunk_steps):
                j1 = min(j0_chunk + PS_chunk_steps, n_store)

                y_chunk = wr.expand_h5_to_full(ps_y[:, j0_chunk:j1])
                t_chunk = dt_store * np.arange(j0_chunk, j1, dtype=npfloat)

                bd.process_bounce_and_drift_chunk(
                    y_chunk=y_chunk,
                    t_chunk=t_chunk,
                    bounce_state=bounce_state,
                    drift_state=drift_state,
                    min_gap_tau=user_min_gap * ps_step,
                    s_eps=v_eps,
                )

        # --- Bounce ---
        bounce_stats = bd.bounce_summary(
            bounce_state["crossing_times"],
            time_scale_sec=tau_time
        )

        if bounce_stats["full_mean_s"] is not None:
            print("Mirror crossings:", bounce_stats["n_crossings"])
            print(f"Full bounce period (mean): {bounce_stats['full_mean_s']:.6g} s")
            print("Bounce frequency [Hz]:", bounce_stats["bounce_frequency_hz"])
        else:
            print("No mirror motion detected (no full-bounce interval).")

        print(f"Initial gyroradius: {gyro_radius_si:.4e} m  ({gyro_radius_RE:.4f} R_E)")
        gyro_freq_hz = 1.0 / (T_gyro * abs(tau_time))
        print(f"Gyrofrequency: {gyro_freq_hz:.4f} Hz  (period: {T_gyro * abs(tau_time):.4e} s)")

        bounce_results = {
            "n_crossings": bounce_stats["n_crossings"],
            "full_mean_tau": bounce_stats["full_mean_tau"],
            "full_mean_s": bounce_stats["full_mean_s"],
            "frequency_hz": bounce_stats["bounce_frequency_hz"],
        }

        # --- Drift ---
        drift_stats = bd.finalize_drift_stream(
            drift_state,
            time_scale_sec=tau_time,
            min_phase_rad=user_min_phase,
        )

        T_drift_s   = drift_stats["period_s_fit"]
        T_drift_tau = drift_stats.get("period_tau_fit", None)
        direction   = drift_stats["direction"]

        if T_drift_s is None:
            print("Drift period: not enough azimuthal motion to estimate (yet).")
        else:
            print(
                f"Drift period ≈ {T_drift_s:.6g} s "
                f"(direction {'east' if direction > 0 else 'west'})"
            )

        drift_results = {
            "period_s": T_drift_s,
            "period_tau": T_drift_tau,
            "direction": direction,
        }

    if DEBUG:
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        logger.info(f"Peak memory usage for bounce/drift analysis: {peak / 1024**2:.2f} MB")

    # ===========================================================
    # ========= Write Summary Output to File & CVS Log ==========
    # ===========================================================

    if DEBUG: tracemalloc.start()

    wr.summary_txt_dipoleb(
        summary=summary, run_folder=run_folder, stem=stem,
        dragt_log=dragt_log, bounce_results=bounce_results, drift_results=drift_results,
        gyroperiods=gyroperiods, norm_time=norm_time, mass=mass, cache_path=cache_path,
        USE_PS=USE_PS, USE_RK4=USE_RK4, USE_RK45=USE_RK45, USE_RKG=USE_RKG,
        PS_decimate=PS_decimate,
        ps_step=ps_step,
        rk4_step=rk4_step if USE_RK4 else None,
        rkg_step=rkg_step if USE_RKG else None,
        rel_drift_ps=rel_drift_ps if USE_PS else None,
        rel_drift_rk4=rel_drift_rk4 if USE_RK4 else None,
        rel_drift_rk45=rel_drift_rk45 if USE_RK45 else None,
        rel_drift_rkg=rel_drift_rkg if USE_RKG else None,
        mu0_ps=mu0_ps if USE_PS else None,
        mu0_rk4=mu0_rk4 if USE_RK4 else None,
        mu0_rk45=mu0_rk45 if USE_RK45 else None,
        mu0_rkg=mu0_rkg if USE_RKG else None,
        solution_rk4=solution_rk4 if USE_RK4 else None,
        solution_rkg=solution_rkg if USE_RKG else None,
        y_rk45_common=y_rk45_common if USE_RK45 else None,
        ps_store_stride=ps_store_stride if USE_PS else 1,
        npfloat=npfloat,
        compute_mu_ps=mp.compute_mu_ps,
        compute_mu_rk=mp.compute_mu_rk,
        vector_potential=dp.vector_potential,
    )

    if DEBUG:
        if USE_RK4:logger.debug(f"  rk4 step size = {rk4_step}")
        if USE_RKG: logger.debug(f"  rkg step size = {rkg_step}")
        if USE_RK4: logger.debug(f"  rk4 steps     = {steps_rk4}")
        if USE_RKG: logger.debug(f"  rkg steps     = {steps_rkg}")
        if USE_PS: logger.debug(f"  ps steps      = {steps_ps}")

    # === Write to master simulation log CSV ===
    _method_records = []
    if USE_RK4:  _method_records.append(("RK4",  steps_rk4, rk4_step, rel_drift_rk4,  mu_rk4_result["mudrift"]))
    if USE_RK45: _method_records.append(("RK45", steps_ps,  ps_step,  rel_drift_rk45, mu_rk45_result["mudrift"]))
    if USE_RKG:  _method_records.append(("RKG",  steps_rkg, rkg_step, rel_drift_rkg,  mu_rkg_result["mudrift"]))
    if USE_PS:   _method_records.append(("PS",   steps_ps,  ps_step,  rel_drift_ps,   mu_ps_result["mudrift"]))

    wr.master_csv(
        output_folder=output_folder, stem=stem, particle_type=particle_type,
        KE_particle=KE_particle, x_initial=x_initial, y_initial=y_initial,
        z_initial=z_initial, pitch_deg=pitch_deg, phi_deg=phi_deg,
        dragt_log=dragt_log,
        method_records=_method_records,
    )

    print(f"\nRun Complete → {run_folder}")
    print(f"  Figures → {fig_folder}")


    if DEBUG:
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        logger.info(f"Peak memory usage for summary write up: {peak / 1024**2:.2f} MB")


if __name__ == "__main__":
    run = "demo"
    if len(sys.argv) > 1:
        run = sys.argv[1]
        print(f"Run mode set from command line: {run}\n")
    else:
        print(f"Using default run mode: {run}\n")

    _configs_dir = os.path.join(os.path.dirname(__file__), "configs", "dipoleb")

    if run.endswith((".yml", ".yaml")) and os.path.isfile(run):
        _yaml_path = run
    elif os.path.isfile(os.path.join(_configs_dir, f"{run}.yml")):
        _yaml_path = os.path.join(_configs_dir, f"{run}.yml")
    else:
        raise FileNotFoundError(
            f"No YAML config found for '{run}'. "
            f"Expected configs/dipoleb/{run}.yml or a direct path to a .yml file.\n"
            f"Available configs: {[f.replace('.yml','') for f in os.listdir(_configs_dir) if f.endswith('.yml') and f != 'base.yml']}"
        )

    print(f"Loading YAML config: {_yaml_path}\n")
    main(_yaml_path)
