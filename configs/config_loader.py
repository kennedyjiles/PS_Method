"""
YAML config loader for dipoleB simulations.

Two-stage design (following advisor's config pattern):
  1. load_config()      — loads YAML, merges with base, validates, prints.
                           Returns the raw merged dict (no physics computation).
  2. compute_derived()  — takes the raw config dict, computes all derived
                           quantities (T_gyro, step sizes, norm_time, mass, …).
                           Returns a flat params dict ready for dipoleB.py.

Usage in dipoleB.py:
    cfg    = load_config("configs/demo.yml")
    params = compute_derived(cfg, npfloat=npfloat)
"""

import os
import copy
import yaml
import numpy as np
from ps_method.constants import q_e, m_e, m_p, spdlight, RE, B_0

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))


def _deep_merge(base, override):
    """In-place deep merge: override wins on conflicts (matches advisor pattern)."""
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v


def _collect_keys(d, prefix=""):
    """Recursively collect all keys from a nested dict as dot-paths."""
    keys = set()
    for k, v in d.items():
        full = f"{prefix}.{k}" if prefix else k
        keys.add(full)
        if isinstance(v, dict):
            keys |= _collect_keys(v, full)
    return keys


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


def _compute_relativistic_L_eff(KE_eV, mass_si, pitch_deg, phi_deg, x_initial):
    """Relativistic gyro-physics: effective L-shell, gamma, physics-based T_gyro.
    Used for dragt work or where gyroradius is a significant fraction of x_initial."""
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
# Stage 1: Load, merge, validate, print
# ---------------------------------------------------------------------------

def load_config(conf_file):
    """
    Load a run YAML, merge with base config, validate keys, and print.

    If the run config contains a 'base_config' key, that file is loaded as the
    base.  Otherwise, base.yml (in this directory) is used.

    Parameters
    ----------
    conf_file : str
        Path to the run config YAML.

    Returns
    -------
    dict
        The raw merged config (no derived quantities).
    """

    # --- Config log (written to file later, not printed) ---
    log = []

    # --- Load run config ---
    with open(conf_file, "r") as f:
        run_cfg = yaml.safe_load(f) or {}

    log.append("=" * 60)
    log.append(f"Run config: {conf_file}")
    log.append("=" * 60)
    log.append(yaml.dump(run_cfg, default_flow_style=False, sort_keys=False).rstrip())
    log.append("")

    # --- Load base config ---
    if "base_config" in run_cfg:
        base_path = run_cfg.pop("base_config")  # consumed, not passed through
        # resolve relative paths against the run config's directory
        if not os.path.isabs(base_path):
            base_path = os.path.join(os.path.dirname(os.path.abspath(conf_file)), base_path)
    else:
        base_path = os.path.join(_THIS_DIR, "base.yml")

    with open(base_path, "r") as f:
        base_cfg = yaml.safe_load(f)

    log.append("=" * 60)
    log.append(f"Base config: {base_path}")
    log.append("=" * 60)
    log.append(yaml.dump(base_cfg, default_flow_style=False, sort_keys=False).rstrip())
    log.append("")

    # --- Validate: warn about unknown keys ---
    base_keys = _collect_keys(base_cfg)
    run_keys  = _collect_keys(run_cfg)
    unknown   = run_keys - base_keys
    # Filter out keys that are valid overrides (e.g. total_steps, norm_time_override)
    # These are intentionally absent from defaults because they represent alternate modes
    _known_extras = {
        "total_steps", "norm_time_override", "base_config",
        "output_folder", "run_storage",  # auto-derived but allowed as overrides
    }
    real_unknown = {k for k in unknown if k.split(".")[0] not in _known_extras}
    if real_unknown:
        # Warnings still print to screen — you want to see these immediately
        print(f"\n  WARNING: Unknown config keys (possible typos): {sorted(real_unknown)}")
        print(f"  Valid top-level keys: {sorted(k for k in base_keys if '.' not in k)}\n")
        log.append(f"WARNING: Unknown config keys: {sorted(real_unknown)}")
        log.append("")

    # --- Deep merge: run config overrides base ---
    cfg = copy.deepcopy(base_cfg)
    _deep_merge(cfg, run_cfg)

    log.append("=" * 60)
    log.append("Merged configuration")
    log.append("=" * 60)
    log.append(yaml.dump(cfg, default_flow_style=False, sort_keys=False).rstrip())
    log.append("")

    # --- Validate critical values ---
    particle = cfg.get("particle", "").strip().lower()
    if particle not in ("proton", "p", "electron", "e"):
        raise ValueError(f"Unknown particle type: {cfg.get('particle')!r}. Use 'proton' or 'electron'.")

    energy = cfg.get("energy_eV")
    if energy is not None and float(energy) <= 0:
        raise ValueError(f"energy_eV must be positive, got {energy}")

    pitch = cfg.get("pitch_deg")
    if pitch is not None and not (0 < float(pitch) < 180):
        raise ValueError(f"pitch_deg must be between 0 and 180 (exclusive), got {pitch}")

    # --- Inject metadata for downstream use ---
    cfg["_config_name"] = os.path.splitext(os.path.basename(conf_file))[0]
    cfg["_config_log"]  = log   # written to file by dipoleB.py once output dirs exist

    return cfg


# ---------------------------------------------------------------------------
# Stage 2: Compute derived quantities
# ---------------------------------------------------------------------------

def compute_derived(cfg, npfloat=np.float64):
    """
    Take a raw merged config dict and compute all derived quantities.

    Parameters
    ----------
    cfg : dict
        Raw config from load_config().
    npfloat : dtype
        Floating-point type (np.float64 or np.float128).

    Returns
    -------
    dict
        Flat params dict with both raw config values and derived physics
        quantities, ready for explicit unpacking in dipoleB.py.
    """

    # --- Resolve particle mass ---
    mass_si = _resolve_mass(cfg["particle"])

    # --- Output paths (auto-derived from config name) ---
    config_name = cfg.get("_config_name", "default")
    output_folder = os.path.join("data", config_name)
    run_storage   = os.path.join(output_folder, "_rawdata")
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(run_storage, exist_ok=True)

    # --- Physics seeds ---
    pitch_deg   = npfloat(cfg["pitch_deg"])
    phi_deg     = npfloat(cfg["phi_deg"])
    x_initial   = npfloat(cfg["x_initial"])
    KE_particle = npfloat(cfg["energy_eV"])
    y_initial   = npfloat(cfg.get("y_initial", 0.0))
    z_initial   = npfloat(cfg.get("z_initial", 0.0))

    # --- T_gyro (relativistic or simple) ---
    if cfg.get("use_gyroradius_L_correction", False):
        L_eff, gamma, T_gyro = _compute_relativistic_L_eff(
            KE_particle, mass_si, pitch_deg, phi_deg, x_initial)
        cfg.setdefault("_config_log", []).append(
            f"  gyroradius L correction: L_eff {L_eff:.6f} R_E used for T_gyro (launch unchanged at {float(x_initial):.6f})")
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

    # --- Build the params dict ---
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
        "run_storage":   run_storage,

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

        # Adaptive PS settings
        "ps_adaptive": cfg.get("ps_adaptive", {}),

        # Dragt monitor
        "dragt_monitor_rtol": cfg.get("dragt_monitor_rtol", 1e-4),

        # Bounce/drift detection
        "bounce_drift": cfg.get("bounce_drift", {}),
    }

    return params
