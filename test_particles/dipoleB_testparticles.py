import numpy as np
import os
import matplotlib.pyplot as mpl
import json as _json
from ps_method.constants import q_e, m_e, m_p, evtoj, spdlight, RE, B_0


USE_FLOAT128 = False  # RKG Will be diabled if this is True

if USE_FLOAT128: npfloat = np.float128
else: npfloat = np.float64


# ===== Tolerances/Truncations =====
PS_order = 40                       # Max Power Series Order, system will truncate
tol = 1.0 * np.finfo(npfloat).eps   # setting tolerance to machine epsilon to drop terms later, will be multiplied by tau_0
rtol_rk45 = 1e-8                    # RK45 relative tolerance
atol_rk45 = 1e-10                   # RK45 adapative tolerance
user_min_phase = npfloat(.1)        # Minimum radians to extrapolate drift from,   .0000001 needed for L=1, 10 eV electron
mpl.rcParams['agg.path.chunksize'] = 100   # may have to adjust if matplotlib barfs on large datasets

if USE_FLOAT128: mpl.rcParams['agg.path.chunksize'] = 100000
else: mpl.rcParams['agg.path.chunksize'] = 1000


MAX_PLOT_POINTS= 1000000 # cap number of points on each graph to speed things up
PS_chunk_steps = int(1e4)

run_storage = "outputs/outputs_rawdata"       # where raw trajectory files go when USE_WRITE_DATA = True
# run_storage = "/Volumes/Extreme SSD/Thesis/PS_method/outputs_rawdata"       # where raw trajectory files go when USE_WRITE_DATA = True, set to external drive for large runs


# ===================================================================
# ==============Toggle Parameters for Dipole Script ================
# ===================================================================
"""
Toggles:
    USE_RK45        -- Set to True to include RK45 (Dormand-Prince) analysis
    USE_RK4         -- Set to True to include RK4 analysis
    USE_RKG         -- Set to True to include RKG analysis (protons only)
    USE_PS          -- Set to True to include PS analysis
    PS_decimate     -- Save every Nth PS step to reduce file size (1 = save all)
    READ_DATA       -- Set to True to scan for saved runs and load from cache
    USE_PLOT_TITLES -- Set to True to include plot titles
    USE_FULL_PLOT   -- Set to False for paper plots only, True for all diagnostic plots

    USE_EXTERNAL_H5_* -- Set to True to load an external h5 file for energy comparison,
                         useful when comparing runs with different parameter sets or timescales
    external_h5_*     -- Path to external h5 file; for PS also set PS_order_ext to match

Physics parameters (changing these creates a new h5 file):
    pitch_deg   -- equatorial pitch angle (degrees)
    phi_deg     -- initial gyrophase (degrees)
    x_initial   -- launch L-shell (Earth radii, R_E)
    y_initial   -- initial y position (R_E), typically 0
    z_initial   -- initial z position (R_E), typically 0
    KE_particle -- kinetic energy (eV)
    mass_si     -- particle mass: m_p for protons, m_e for electrons, or manual (kg)
    T_gyro      -- characteristic gyroperiod at equator (normalized time): 2*pi*L^3

Integration parameters:
    N_STEPS_PER_GYRO_* -- number of steps per characteristic gyroperiod for each integrator
    rk4_step    -- RK4 time step: T_gyro / N_STEPS_PER_GYRO_rk4
    ps_step     -- PS time step: set equal to rk4_step for direct comparison
    rkg_step    -- RKG time step: set equal to rk4_step for direct comparison
    gyroperiods -- number of characteristic gyroperiods to simulate
    norm_time   -- total integration time in normalized units: gyroperiods * T_gyro

Plotting parameters (do not affect h5 data):
    window_time  -- time window in physical seconds (SI) for trajectory slice inspection,
                    typically one drift period; converted to normalized time internally via tau_0
    slice_mode   -- "last" or "first", slices window from end or beginning of simulation
    N_GYRO       -- number of gyroperiods to display in slice
    gyro_window  -- "first", "last", or "all", which gyroperiods to display in the window

Run modes:
    demo         -- quick verification run (seconds)
    paper1-3     -- reproduce paper figures (minutes each)
    dragt        -- Dragt monitoring test (PS only, relativistic proton)
    walt         -- high pitch angle test (PS only, relativistic proton)
    batch:path   -- batch mode, reads parameters from JSON config (used by batch_flux_runner.py)
    legacy       -- load old-format h5 files
    manual       -- load specific h5 file
    monster_ps   -- extended PS-only run

This file is shared by dipoleB.py, which handles both fixed-step and adaptive modes via USE_ADAPTIVE.
"""

# ===================================================================
def _defaults():
    """Baseline parameters shared by most run modes. Each mode overrides what differs."""
    return dict(
        READ_DATA  = True,

        USE_RK45     = False,
        USE_RK4      = False,
        USE_RKG      = False,
        USE_PS       = True,
        USE_ADAPTIVE = False,   # True = adaptive PS stepping, False = fixed-step
        PS_decimate  = 1,

        y_initial = npfloat(0),
        z_initial = npfloat(0),

        USE_PLOT_TITLES = False,
        USE_FULL_PLOT   = True,
        slice_mode  = "last",
        gyro_window = "last",

        USE_EXTERNAL_H5_ps   = False,
        USE_EXTERNAL_H5_rk4  = False,
        USE_EXTERNAL_H5_rk45 = False,
        USE_EXTERNAL_H5_rkg  = False,
        external_h5_ps   = "outputs/outputs_rawdata/",
        external_h5_rk4  = "outputs/outputs_rawdata/",
        external_h5_rk45 = "outputs/outputs_rawdata/",
        external_h5_rkg  = "outputs/outputs_rawdata/",
    )


def _compute_steps(T_gyro, N_ps=65, N_rk4=65, N_rkg=65, rounding=True):
    """Compute integrator time steps from T_gyro and steps-per-gyroperiod.
    Returns (ps_step, rk4_step, rkg_step, N_ps, N_rk4, N_rkg)."""
    if rounding:
        ps_step  = npfloat(round(T_gyro / N_ps,  1))
        rk4_step = npfloat(round(T_gyro / N_rk4, 1))
        rkg_step = npfloat(round(T_gyro / N_rkg, 1))
    else:
        ps_step  = npfloat(T_gyro / N_ps)
        rk4_step = npfloat(T_gyro / N_rk4)
        rkg_step = npfloat(T_gyro / N_rkg)
    return ps_step, rk4_step, rkg_step, N_ps, N_rk4, N_rkg


def _compute_relativistic_L_eff(KE_particle, mass_si, pitch_deg, phi_deg, x_initial):
    """Relativistic gyro-physics: compute effective L-shell, gamma, and physics-based T_gyro.
    Used by dragt, walt, and batch modes for high-energy particles."""
    E_kinetic = KE_particle * abs(q_e)
    E_rest    = mass_si * (spdlight**2)
    gamma     = 1.0 + (E_kinetic / E_rest)
    v_total   = spdlight * np.sqrt(1.0 - (1.0 / gamma**2))
    alpha_rad = np.radians(pitch_deg)
    v_perp    = v_total * np.sin(alpha_rad)

    B_at_launch = B_0 / (x_initial**3)
    omega_init  = (abs(q_e) * B_at_launch) / (gamma * mass_si)
    r_g_RE      = (v_perp / omega_init) / RE

    phi_rad = np.radians(phi_deg)
    L_eff   = x_initial + (r_g_RE * np.sin(phi_rad))

    T_gyro_physics = 2.0 * np.pi * (L_eff**3)
    return L_eff, gamma, T_gyro_physics


# ===================================================================
# ======================== Run Modes ================================
# ===================================================================

def load_params(run):

    if run == "demo": # paper2 physics at reduced norm_time for quick testing
        if USE_FLOAT128: print("Running DEMO simulation in float128...this may take a few minutes\n")
        else: print("Running DEMO simulation...this takes just few seconds\n")

        p = _defaults()
        p["output_folder"] = "outputs/demo"
        os.makedirs(p["output_folder"], exist_ok=True)

        # Solvers
        p["USE_RK45"] = True
        p["USE_RK4"]  = True

        # Physics
        p["pitch_deg"]   = npfloat(60.0)
        p["phi_deg"]     = npfloat(90.0)
        p["x_initial"]   = npfloat(5)
        p["KE_particle"] = npfloat(100e6)
        p["mass_si"]     = m_e
        T_gyro = 2.0 * np.pi * (p["x_initial"]**3)
        p["T_gyro"] = T_gyro

        # Steps
        ps_step, rk4_step, rkg_step, N_ps, N_rk4, N_rkg = _compute_steps(T_gyro)
        p.update(ps_step=ps_step, rk4_step=rk4_step, rkg_step=rkg_step,
                 N_STEPS_PER_GYRO_ps=N_ps, N_STEPS_PER_GYRO_rk4=N_rk4, N_STEPS_PER_GYRO_rkg=N_rkg)
        p["gyroperiods"] = 4.5e2
        p["norm_time"]   = npfloat(p["gyroperiods"]) * T_gyro

        # Plotting
        p["USE_PLOT_TITLES"] = True
        p["window_time"]  = npfloat(11.6)
        p["slice_mode"]   = "first"
        p["N_GYRO"]       = 75

        return p

    elif run == "paper1": #100 keV proton, 30deg pitch, 5 RE
        if USE_FLOAT128: print("Running PAPER simulation in float128...this may take a >30 minutes\n")
        else: print("Running full PAPER simulation...this can take a few minutes\n")

        p = _defaults()
        p["output_folder"] = "outputs/paper"
        os.makedirs(p["output_folder"], exist_ok=True)

        # Solvers
        p["USE_RK45"] = True
        p["USE_RK4"]  = True
        p["USE_RKG"]  = True

        # Physics
        p["pitch_deg"]   = npfloat(30.0)
        p["phi_deg"]     = npfloat(90.0)
        p["x_initial"]   = npfloat(5)
        p["KE_particle"] = npfloat(100e3)
        p["mass_si"]     = m_p
        T_gyro = 2.0 * np.pi * (p["x_initial"]**3)
        p["T_gyro"] = T_gyro

        # Steps (paper mode: defined via total integration steps)
        ps_step, rk4_step, rkg_step, N_ps, N_rk4, N_rkg = _compute_steps(T_gyro)
        p.update(ps_step=ps_step, rk4_step=rk4_step, rkg_step=rkg_step,
                 N_STEPS_PER_GYRO_ps=N_ps, N_STEPS_PER_GYRO_rk4=N_rk4, N_STEPS_PER_GYRO_rkg=N_rkg)
        totatl_integration_steps = 1e7
        p["norm_time"]   = npfloat(totatl_integration_steps) * ps_step
        p["gyroperiods"] = npfloat(totatl_integration_steps) * ps_step / T_gyro

        # Plotting
        p["window_time"] = npfloat(6209.0)
        p["N_GYRO"]      = 150

        # External h5
        p["external_h5_ps"]  = "/Volumes/Extreme SSD/Thesis/PS_method/outputs_rawdata/run_1f911496e90d7c4b.h5"
        p["external_h5_rkg"] = "/Volumes/Extreme SSD/Thesis/PS_method/outputs_rawdata/run_ae2d63d764f68d00.h5"

        return p

    elif run == "paper2":  # 100 MeV electron, 60deg pitch, 5 RE
        if USE_FLOAT128: print("Running PAPER simulation in float128...this may take a >30 minutes\n")
        else: print("Running full PAPER simulation...this can take a few minutes\n")

        p = _defaults()
        p["output_folder"] = "outputs/paper"
        os.makedirs(p["output_folder"], exist_ok=True)

        # Solvers
        p["USE_RK45"] = True
        p["USE_RK4"]  = True
        p["USE_RKG"]  = True

        # Physics
        p["pitch_deg"]   = npfloat(60.0)
        p["phi_deg"]     = npfloat(90.0)
        p["x_initial"]   = npfloat(5)
        p["KE_particle"] = npfloat(100e6)
        p["mass_si"]     = m_e
        T_gyro = 2.0 * np.pi * (p["x_initial"]**3)
        p["T_gyro"] = T_gyro

        # Steps (paper mode: defined via total integration steps)
        ps_step, rk4_step, rkg_step, N_ps, N_rk4, N_rkg = _compute_steps(T_gyro)
        p.update(ps_step=ps_step, rk4_step=rk4_step, rkg_step=rkg_step,
                 N_STEPS_PER_GYRO_ps=N_ps, N_STEPS_PER_GYRO_rk4=N_rk4, N_STEPS_PER_GYRO_rkg=N_rkg)
        totatl_integration_steps = 1e7
        p["norm_time"]   = npfloat(totatl_integration_steps) * ps_step
        p["gyroperiods"] = npfloat(totatl_integration_steps) * ps_step / T_gyro

        # Plotting
        p["USE_PLOT_TITLES"] = True
        p["window_time"] = npfloat(11.6)
        p["N_GYRO"]      = 75

        # External h5
        p["external_h5_ps"] = "outputs/outputs_rawdata/run_5f2698f4194712e0.h5"

        return p

    elif run == "paper3": # paper1 physics at larger ps_step and smaller rk4_step
        if USE_FLOAT128: print("Running PAPER simulation in float128...this may take a >30 minutes\n")
        else: print("Running full PAPER simulation...this can take a few minutes\n")

        p = _defaults()
        p["output_folder"] = "outputs/paper"
        os.makedirs(p["output_folder"], exist_ok=True)

        # Solvers
        p["USE_RK4"] = True

        # Physics
        p["pitch_deg"]   = npfloat(30.0)
        p["phi_deg"]     = npfloat(90.0)
        p["x_initial"]   = npfloat(5)
        p["KE_particle"] = npfloat(100e3)
        p["mass_si"]     = m_p
        T_gyro = 2.0 * np.pi * (p["x_initial"]**3)
        p["T_gyro"] = T_gyro
        p["PS_order"] = 16

        # Steps (extreme mismatch: PS takes huge steps, RK4 takes tiny steps)
        _, _, rkg_step, _, _, N_rkg = _compute_steps(T_gyro)
        p["N_STEPS_PER_GYRO_rk4"] = 1570.8
        p["N_STEPS_PER_GYRO_ps"]  = 14.5
        p["N_STEPS_PER_GYRO_rkg"] = N_rkg
        p["rk4_step"] = npfloat(0.5)
        p["ps_step"]  = npfloat(54.0)
        p["rkg_step"] = rkg_step
        p["norm_time"]   = npfloat(6206.0 * 4 / .00033464094804535314)
        p["gyroperiods"] = npfloat(p["norm_time"]) / T_gyro

        # Plotting
        p["USE_FULL_PLOT"] = True
        p["window_time"]   = npfloat(6209.0)
        p["N_GYRO"]        = 150

        # External h5
        p["external_h5_ps"]  = "outputs/outputs_rawdata/run_36f4c2c523cdbb17.h5"
        p["external_h5_rk4"] = "outputs/outputs_rawdata/run_7b2f03f027541ae1.h5"

        return p

    elif run == "dragt":
        if USE_FLOAT128: print("Running PAPER simulation in float128...this may take >30 minutes\n")
        else: print("Running full PAPER simulation...this can take a few minutes\n")

        p = _defaults()
        p["output_folder"] = "outputs/dragt"
        os.makedirs(p["output_folder"], exist_ok=True)
        p["USE_ADAPTIVE"] = True

        # Physics
        p["pitch_deg"]   = npfloat(67.61877044327187)
        p["phi_deg"]     = npfloat(-90.0)
        p["x_initial"]   = npfloat(3.34483896)
        p["KE_particle"] = npfloat(210.5306105318196e6)
        p["mass_si"]     = m_p

        # Relativistic L_eff
        L_eff, gamma, T_gyro = _compute_relativistic_L_eff(
            p["KE_particle"], p["mass_si"], p["pitch_deg"], p["phi_deg"], p["x_initial"])
        p["T_gyro"] = T_gyro

        # Steps
        ps_step, rk4_step, rkg_step, N_ps, N_rk4, N_rkg = _compute_steps(T_gyro, rounding=False)
        p.update(ps_step=ps_step, rk4_step=rk4_step, rkg_step=rkg_step,
                 N_STEPS_PER_GYRO_ps=N_ps, N_STEPS_PER_GYRO_rk4=N_rk4, N_STEPS_PER_GYRO_rkg=N_rkg)
        p["gyroperiods"] = 1e5
        p["norm_time"]   = npfloat(p["gyroperiods"]) * T_gyro
        p["PS_order"]    = 1000

        # Plotting
        p["USE_PLOT_TITLES"] = True
        p["USE_FULL_PLOT"]   = True
        p["window_time"]  = npfloat(5)
        p["slice_mode"]   = "first"
        p["N_GYRO"]       = 175
        p["gyro_window"]  = "first"

        # External h5
        p["external_h5_ps"]   = "outputs/outputs_rawdata/run_304603211ada647c.h5"
        p["external_h5_rk4"]  = "outputs/outputs_rawdata/run_60a0558ebec9f956.h5"
        p["external_h5_rk45"] = "outputs/outputs_rawdata/run_60a0558ebec9f956.h5"
        p["external_h5_rkg"]  = "outputs/outputs_rawdata/run_60a0558ebec9f956.h5"

        return p

    elif run == "walt":
        if USE_FLOAT128: print("Running PAPER simulation in float128...this may take >30 minutes\n")
        else: print("Running full PAPER simulation...this can take a few minutes\n")

        p = _defaults()
        p["output_folder"] = "outputs/walt"
        os.makedirs(p["output_folder"], exist_ok=True)
        p["READ_DATA"] = False
        p["USE_ADAPTIVE"] = True

        # Physics
        p["pitch_deg"]   = npfloat(89.0)
        p["phi_deg"]     = npfloat(0.0)
        p["x_initial"]   = npfloat(5.7045989)
        p["KE_particle"] = npfloat(1e8)
        p["mass_si"]     = m_p

        # Relativistic L_eff
        L_eff, gamma, T_gyro = _compute_relativistic_L_eff(
            p["KE_particle"], p["mass_si"], p["pitch_deg"], p["phi_deg"], p["x_initial"])
        p["T_gyro"] = T_gyro

        # Steps (PS uses 20 steps/gyro instead of 65)
        ps_step, rk4_step, rkg_step, N_ps, N_rk4, N_rkg = _compute_steps(T_gyro, N_ps=20, rounding=False)
        p.update(ps_step=ps_step, rk4_step=rk4_step, rkg_step=rkg_step,
                 N_STEPS_PER_GYRO_ps=N_ps, N_STEPS_PER_GYRO_rk4=N_rk4, N_STEPS_PER_GYRO_rkg=N_rkg)
        p["gyroperiods"] = 1e4
        p["norm_time"]   = npfloat(p["gyroperiods"]) * T_gyro
        p["PS_order"]    = 1000

        # Plotting
        p["USE_PLOT_TITLES"] = True
        p["USE_FULL_PLOT"]   = True
        p["window_time"]  = npfloat(5)
        p["slice_mode"]   = "first"
        p["N_GYRO"]       = 175

        # External h5
        p["external_h5_ps"]   = "outputs/outputs_rawdata/run_304603211ada647c.h5"
        p["external_h5_rk4"]  = "outputs/outputs_rawdata/run_60a0558ebec9f956.h5"
        p["external_h5_rk45"] = "outputs/outputs_rawdata/run_60a0558ebec9f956.h5"
        p["external_h5_rkg"]  = "outputs/outputs_rawdata/run_60a0558ebec9f956.h5"

        return p

    elif run.startswith("batch"):
        """
        Batch mode: reads parameters from a JSON config file.
        Used by scripts/batch_flux_runner.py for parameter sweeps.
        Does not modify any hardcoded parameters in this file.

        Accepts: "batch"  (reads default scripts/batch_config.json)
                 "batch:/path/to/config.json"  (reads specified config file)
        The colon-path form allows parallel workers to each use their own
        config file without race conditions.
        """
        if ":" in run:
            _config_path = run.split(":", 1)[1]
        else:
            _config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                         "scripts", "batch_config.json")
        with open(_config_path, "r") as _f:
            _cfg = _json.load(_f)

        p = _defaults()
        p["output_folder"] = _cfg.get("output_folder", "outputs/flux_map")
        os.makedirs(p["output_folder"], exist_ok=True)
        p["USE_ADAPTIVE"] = _cfg.get("use_adaptive", True)

        # Physics
        p["pitch_deg"]   = npfloat(_cfg["pitch_deg"])
        p["phi_deg"]     = npfloat(_cfg.get("phi_deg", 0.0))
        p["x_initial"]   = npfloat(_cfg["L_shell"])
        p["KE_particle"] = npfloat(_cfg["energy_eV"])
        p["mass_si"]     = m_p

        # Relativistic L_eff
        L_eff, gamma, T_gyro = _compute_relativistic_L_eff(
            p["KE_particle"], p["mass_si"], p["pitch_deg"], p["phi_deg"], p["x_initial"])
        p["T_gyro"] = T_gyro

        # Steps
        ps_step, rk4_step, rkg_step, N_ps, N_rk4, N_rkg = _compute_steps(
            T_gyro, N_ps=65, N_rk4=20, rounding=False)
        p.update(ps_step=ps_step, rk4_step=rk4_step, rkg_step=rkg_step,
                 N_STEPS_PER_GYRO_ps=N_ps, N_STEPS_PER_GYRO_rk4=N_rk4, N_STEPS_PER_GYRO_rkg=N_rkg)
        p["gyroperiods"] = _cfg.get("gyroperiods", 5e4)
        p["norm_time"]   = npfloat(p["gyroperiods"]) * T_gyro
        p["PS_order"]    = 1000

        # Plotting (minimal for batch)
        p["window_time"]  = npfloat(5)
        p["slice_mode"]   = "first"
        p["N_GYRO"]       = 50

        print(f"BATCH: E={p['KE_particle']/1e6:.0f} MeV  L={p['x_initial']:.2f}  "
              f"pitch={p['pitch_deg']:.1f}°  gyroperiods={p['gyroperiods']:.0e}")

        return p

    elif run == "legacy": # this may be obsolete now, I am not sure how many legacy files are left.
        if USE_FLOAT128: print("Running PAPER simulation in float128...this may take a >30 minutes\n")
        else: print("Running full PAPER simulation...this can take a few minutes\n")
        """
        This allows legacy files to be loaded directly through the 'legacy' run in the test particle function.
        Early runs didn't have all the parameters we are now tracking, so the scanning doesn't work properly.
        The functions take the old h5 files we did have and reconstruct a dictionary in the format we are using now.
        """

        p = _defaults()
        p["output_folder"] = "outputs/legacy"
        os.makedirs(p["output_folder"], exist_ok=True)

        p["legacy_h5_path"] = "outputs_rawdata/run_98a2efbd7550732a-1.h5"

        # Plotting
        p["window_time"] = npfloat(6209.0)
        p["N_GYRO"]      = 150

        return p

    elif run == "manual": # load specific h5 file
        if USE_FLOAT128: print("Running PAPER simulation in float128...this may take a >30 minutes\n")
        else: print("Running full PAPER simulation...this can take a few minutes\n")

        p = _defaults()
        p["output_folder"] = "outputs/manual"
        os.makedirs(p["output_folder"], exist_ok=True)

        p["manual_h5_path"] = "outputs/outputs_rawdata/run_60f693535958025f.h5"

        # Plotting
        p["window_time"] = npfloat(5.0)
        p["N_GYRO"]      = 150

        return p

    elif run == "monster_ps":  # giant ps runs
        if USE_FLOAT128: print("Running PAPER simulation in float128...this may take a >30 minutes\n")
        else: print("Running full PAPER simulation...this can take a few minutes\n")

        p = _defaults()
        p["output_folder"] = "outputs/extended_runs"
        os.makedirs(p["output_folder"], exist_ok=True)
        p["READ_DATA"] = False

        # Physics
        p["pitch_deg"]   = npfloat(85.0)
        p["phi_deg"]     = npfloat(90.0)
        p["x_initial"]   = npfloat(8)
        p["KE_particle"] = npfloat(1e7)
        p["mass_si"]     = m_p
        T_gyro = 2.0 * np.pi * (p["x_initial"]**3)
        p["T_gyro"] = T_gyro

        # Steps
        ps_step, rk4_step, rkg_step, N_ps, N_rk4, N_rkg = _compute_steps(T_gyro)
        p.update(ps_step=ps_step, rk4_step=rk4_step, rkg_step=rkg_step,
                 N_STEPS_PER_GYRO_ps=N_ps, N_STEPS_PER_GYRO_rk4=N_rk4, N_STEPS_PER_GYRO_rkg=N_rkg)
        p["gyroperiods"] = 1e4
        p["norm_time"]   = npfloat(p["gyroperiods"]) * T_gyro

        # Plotting
        p["window_time"]   = npfloat(6209.0)
        p["N_GYRO"]        = 150
        p["gyro_window"]   = "first"

        return p

    else:
        raise ValueError("run must be 'demo', 'paper1', 'paper2', 'paper3', 'dragt', 'walt', "
                         "'batch', 'legacy', 'manual', or 'monster_ps'")
