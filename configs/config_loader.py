"""
YAML config loader for dipoleB simulations.

Reads a run config YAML, merges with defaults.yml, and computes all
derived quantities (T_gyro, step sizes, norm_time, etc.).

Returns the same params dict that load_params() currently returns,
so dipoleB.py doesn't need to change its unpacking logic.
"""

import os
import yaml
import numpy as np
from constants import q_e, m_e, m_p, spdlight, RE, B_0

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))


def _deep_merge(base, override):
    """merge override into base. override wins on conflicts."""
    merged = base.copy()
    for key, val in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(val, dict):
            merged[key] = _deep_merge(merged[key], val)
        else:
            merged[key] = val
    return merged


def _resolve_mass(particle_name):
    """Map particle name to SI mass."""
    name = particle_name.strip().lower()
    if name in ("electron", "e"):
        return m_e
    elif name in ("proton", "p"):
        return m_p
    else:
        raise ValueError(f"Unknown particle type: {particle_name!r}. Use 'proton' or 'electron'.")


def _compute_steps(T_gyro, N_ps=65, N_rk4=65, N_rkg=65, rounding=True):
    """Compute integrator step sizes from T_gyro and steps-per-gyroperiod."""
    if rounding:
        ps_step  = round(T_gyro / N_ps,  1)
        rk4_step = round(T_gyro / N_rk4, 1)
        rkg_step = round(T_gyro / N_rkg, 1)
    else:
        ps_step  = T_gyro / N_ps
        rk4_step = T_gyro / N_rk4
        rkg_step = T_gyro / N_rkg
    return ps_step, rk4_step, rkg_step, N_ps, N_rk4, N_rkg


def _compute_relativistic_L_eff(KE_eV, mass_si, pitch_deg, phi_deg, x_initial): #namely for dragt work or where gyroradius is large
    """Relativistic gyro-physics: effective L-shell, gamma, physics-based T_gyro."""
    E_kinetic = KE_eV * abs(q_e)
    E_rest    = mass_si * (spdlight ** 2)
    gamma     = 1.0 + (E_kinetic / E_rest)
    v_total   = spdlight * np.sqrt(1.0 - (1.0 / gamma ** 2))
    alpha_rad = np.radians(pitch_deg)
    v_perp    = v_total * np.sin(alpha_rad)

    B_at_launch = B_0 / (x_initial ** 3)
    omega_init  = (abs(q_e) * B_at_launch) / (gamma * mass_si)
    r_g_RE      = (v_perp / omega_init) / RE

    phi_rad = np.radians(phi_deg)
    L_eff   = x_initial + (r_g_RE * np.sin(phi_rad))

    T_gyro_physics = 2.0 * np.pi * (L_eff ** 3)
    return L_eff, gamma, T_gyro_physics


# ---------------------------------------------------------------------------
# Main loader- this essentialy is doing what the old dipoleb_testparticles was doing.
# ---------------------------------------------------------------------------

def load_config(run_config_path, npfloat=np.float64):
    """
    Load a run YAML, merge with defaults, compute derived quantities.

    Parameters
    ----------
    run_config_path : str
        Path to the run config YAML (e.g. "configs/demo.yml").
    npfloat : dtype
        Floating-point type to use (np.float64 or np.float128).

    Returns
    -------
    dict
        The same keys that load_params() currently returns, ready for
        explicit unpacking in dipoleB.py.
    """

    # --- Load defaults and run config ---
    defaults_path = os.path.join(_THIS_DIR, "defaults.yml")
    with open(defaults_path, "r") as f:
        defaults = yaml.safe_load(f)

    with open(run_config_path, "r") as f:
        run_cfg = yaml.safe_load(f)

    cfg = _deep_merge(defaults, run_cfg)

    # --- Resolve particle mass ---
    mass_si = _resolve_mass(cfg["particle"])

    # --- Output folder ---
    output_folder = cfg["output_folder"]
    os.makedirs(output_folder, exist_ok=True)

    # --- Physics seeds ---
    pitch_deg   = npfloat(cfg["pitch_deg"])
    phi_deg     = npfloat(cfg["phi_deg"])
    x_initial   = npfloat(cfg["L_shell"])
    KE_particle = npfloat(cfg["energy_eV"])
    y_initial   = npfloat(cfg.get("y_initial", 0.0))
    z_initial   = npfloat(cfg.get("z_initial", 0.0))

    # --- T_gyro (relativistic or simple) ---
    if cfg.get("use_gyroradius_L_correction", False):
        L_eff, gamma, T_gyro = _compute_relativistic_L_eff(
            KE_particle, mass_si, pitch_deg, phi_deg, x_initial)
    else:
        T_gyro = 2.0 * np.pi * (x_initial ** 3)

    # --- Step sizes ---
    spg = cfg.get("steps_per_gyro", {})
    N_ps  = spg.get("ps",  65)
    N_rk4 = spg.get("rk4", 65)
    N_rkg = spg.get("rkg", 65)
    rounding = cfg.get("round_steps", True)

    ps_step, rk4_step, rkg_step, N_ps, N_rk4, N_rkg = _compute_steps(
        T_gyro, N_ps=N_ps, N_rk4=N_rk4, N_rkg=N_rkg, rounding=rounding)

    # --- Step overrides (optional — bypass computed values) ---
    overrides = cfg.get("step_overrides", {})
    if overrides.get("ps") is not None:
        ps_step = npfloat(overrides["ps"])
    if overrides.get("rk4") is not None:
        rk4_step = npfloat(overrides["rk4"])
    if overrides.get("rkg") is not None:
        rkg_step = npfloat(overrides["rkg"])

    # --- Integration time ---
    if cfg.get("norm_time_override") is not None:
        norm_time   = npfloat(cfg["norm_time_override"])
        gyroperiods = npfloat(norm_time / T_gyro)
    elif cfg.get("total_steps") is not None:
        total_steps = cfg["total_steps"]
        norm_time   = npfloat(total_steps) * ps_step
        gyroperiods = npfloat(norm_time / T_gyro)
    else:
        gyroperiods = npfloat(cfg["gyroperiods"])
        norm_time   = npfloat(gyroperiods) * T_gyro

    # --- Plotting ---
    plot_cfg = cfg.get("plotting", {})

    # --- External h5 ---
    use_ext = cfg.get("use_external_h5", {})
    ext     = cfg.get("external_h5", {})
    _default_ext = "outputs/outputs_rawdata/"
    ext_ps   = ext.get("ps")   or _default_ext
    ext_rk4  = ext.get("rk4")  or _default_ext
    ext_rk45 = ext.get("rk45") or _default_ext
    ext_rkg  = ext.get("rkg")  or _default_ext

    # --- Solvers ---
    solvers = cfg.get("solvers", {})

    # --- Build the params dict (same keys as load_params) ---
    params = {
        # Toggles
        "READ_DATA":       cfg.get("read_data", True),
        "USE_RK45":        solvers.get("rk45", False),
        "USE_RK4":         solvers.get("rk4", False),
        "USE_RKG":         solvers.get("rkg", False),
        "USE_PS":          solvers.get("ps", True),
        "USE_ADAPTIVE":    solvers.get("adaptive", False),
        "PS_decimate":     cfg.get("ps_decimate", 1),

        # Initial position
        "y_initial": y_initial,
        "z_initial": z_initial,

        # Plotting
        "USE_PLOT_TITLES": plot_cfg.get("titles", False),
        "USE_FULL_PLOT":   plot_cfg.get("full_plot", True),
        "slice_mode":      plot_cfg.get("slice_mode", "last"),
        "gyro_window":     plot_cfg.get("gyro_window", "last"),

        # External h5
        "USE_EXTERNAL_H5_ps":   use_ext.get("ps", False),
        "USE_EXTERNAL_H5_rk4":  use_ext.get("rk4", False),
        "USE_EXTERNAL_H5_rk45": use_ext.get("rk45", False),
        "USE_EXTERNAL_H5_rkg":  use_ext.get("rkg", False),
        "external_h5_ps":   ext_ps,
        "external_h5_rk4":  ext_rk4,
        "external_h5_rk45": ext_rk45,
        "external_h5_rkg":  ext_rkg,

        # Output
        "output_folder": output_folder,
        "run_storage":   cfg.get("run_storage", "outputs/outputs_rawdata"),

        # Physics
        "pitch_deg":   pitch_deg,
        "phi_deg":     phi_deg,
        "x_initial":   x_initial,
        "KE_particle": KE_particle,
        "mass_si":     mass_si,
        "T_gyro":      T_gyro,
        "gyroperiods": gyroperiods,
        "norm_time":   norm_time,

        # Steps
        "ps_step":              npfloat(ps_step),
        "rk4_step":             npfloat(rk4_step),
        "rkg_step":             npfloat(rkg_step),
        "N_STEPS_PER_GYRO_ps":  N_ps,
        "N_STEPS_PER_GYRO_rk4": N_rk4,
        "N_STEPS_PER_GYRO_rkg": N_rkg,

        # Plotting windows
        "window_time": npfloat(cfg.get("window_time", 11.6)),
        "N_GYRO":      plot_cfg.get("n_gyro", 75),

        # Optional overrides
        "PS_order":        cfg.get("ps_order", 40),
        "PS_chunk_steps":  int(cfg.get("ps_chunk_steps", 10000)),
        "rtol_rk45":       cfg.get("rtol_rk45", 1e-8),
        "atol_rk45":       cfg.get("atol_rk45", 1e-10),
        "user_min_phase":  cfg.get("user_min_phase", 0.1),
        "MAX_PLOT_POINTS": cfg.get("max_plot_points", 1_000_000),

        # Special modes
        "legacy_h5_path": cfg.get("legacy_h5_path"),
        "manual_h5_path": cfg.get("manual_h5_path"),
    }

    return params
