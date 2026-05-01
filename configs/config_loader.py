"""
YAML config loader for all field types (dipole, constB, hyperB).

Two-stage design (following advisor's config pattern):
  1. load_config()               — loads YAML, merges with base, validates, prints.
                                   Returns the raw merged dict (no physics computation).
                                   Base config is auto-discovered from the run YAML's directory.
  2. compute_derived_<field>()   — field-specific derived quantities.
                                   Returns a flat params dict ready for the driver script.

Usage:
    cfg    = load_config("configs/dipole/demo.yml")
    params = compute_derived_dipole(cfg, npfloat=npfloat)

    cfg    = load_config("configs/constB/demo.yml")
    params = compute_derived_constB(cfg, npfloat=npfloat)

    cfg    = load_config("configs/hyper/demo.yml")
    params = compute_derived_hyper(cfg, npfloat=npfloat)
"""

import os
import copy
import yaml
import numpy as np
from ps_method.constants import q_e, m_e, m_p, spdlight, RE, B_0 as B_0_dipole

# ---------------------------------------------------------------------------
# Shared helpers
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


def _compute_step_sizes(T_gyro, steps_per_gyro, round_decimals=1):
    """Compute step sizes from T_gyro and a {solver: N} dict.

    Returns a dict of {solver: step_size}.
    """
    steps = {}
    for solver, N in steps_per_gyro.items():
        raw = T_gyro / N
        steps[solver] = round(raw, round_decimals)
    return steps


def _apply_step_overrides(steps, overrides, npfloat=np.float64):
    """Apply explicit step_overrides on top of computed steps."""
    if overrides:
        for solver, val in overrides.items():
            if val is not None:
                steps[solver] = npfloat(val)
    return steps


def _resolve_output_paths(config_name, field_prefix=""):
    """Auto-derive output_folder and run_storage from config name."""
    if field_prefix:
        output_folder = os.path.join("data", field_prefix, config_name)
    else:
        output_folder = os.path.join("data", config_name)
    run_storage = os.path.join(output_folder, "_rawdata")
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(run_storage, exist_ok=True)
    return output_folder, run_storage


def copy_config_to_output(cfg_path, output_folder):
    """Write the fully-merged config into the output folder with the git hash.

    The saved file is self-contained (includes all base defaults) so it can
    be loaded directly from data/ without needing base.yml nearby.  A
    ``base_config: none`` key tells load_config to skip the base merge.

    Parameters
    ----------
    cfg_path      : str – path to the original YAML config.
    output_folder : str – the run's output directory (e.g. data/dipoleB/demo/).
    """
    import subprocess

    # --- Load and merge (same logic as load_config, but we keep the result) ---
    with open(cfg_path, "r") as f:
        run_cfg = yaml.safe_load(f) or {}

    # Find the base config
    if "base_config" in run_cfg:
        base_path = run_cfg.pop("base_config")
        if base_path and base_path.lower() != "none" and not os.path.isabs(base_path):
            base_path = os.path.join(os.path.dirname(os.path.abspath(cfg_path)), base_path)
    else:
        base_path = os.path.join(os.path.dirname(os.path.abspath(cfg_path)), "base.yml")

    if base_path and str(base_path).lower() != "none" and os.path.isfile(base_path):
        with open(base_path, "r") as f:
            merged = yaml.safe_load(f) or {}
        _deep_merge(merged, run_cfg)
    else:
        merged = run_cfg

    # --- Add git commit hash ---
    try:
        _hash = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=os.path.dirname(os.path.abspath(cfg_path)),
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        _dirty = subprocess.call(
            ["git", "diff", "--quiet"],
            cwd=os.path.dirname(os.path.abspath(cfg_path)),
            stderr=subprocess.DEVNULL,
        )
        if _dirty:
            _hash += "-dirty"
    except (subprocess.CalledProcessError, FileNotFoundError):
        _hash = "unavailable"

    merged["git_commit"] = _hash

    # Mark as self-contained so load_config skips the base merge
    merged["base_config"] = "none"

    # --- Write ---
    basename = os.path.basename(cfg_path)
    dest = os.path.join(output_folder, basename)
    with open(dest, "w") as f:
        f.write(f"# Fully merged config — generated from {os.path.basename(cfg_path)}\n")
        f.write(f"# Re-run with: python run.py {dest}\n\n")
        yaml.dump(merged, f, default_flow_style=False, sort_keys=False)

    print(f"Config copied → {dest}")
    return dest


# ---------------------------------------------------------------------------
# Dipole-specific helper
# ---------------------------------------------------------------------------

def _compute_relativistic_L_eff(KE_eV, mass_si, pitch_deg, phi_deg, x_initial):
    """Relativistic gyro-physics: effective L-shell, gamma, physics-based T_gyro.
    Used for dragt work or where gyroradius is a significant fraction of x_initial."""
    E_kinetic = KE_eV * abs(q_e)
    E_rest    = mass_si * (spdlight ** 2)
    gamma     = 1.0 + (E_kinetic / E_rest)
    v_total   = spdlight * np.sqrt(1.0 - (1.0 / gamma ** 2))
    alpha_rad = np.radians(pitch_deg)
    v_perp    = v_total * np.sin(alpha_rad)

    B_at_launch = B_0_dipole / (x_initial ** 3)
    omega_init  = (abs(q_e) * B_at_launch) / (gamma * mass_si)
    r_g_RE      = (v_perp / omega_init) / RE

    phi_rad = np.radians(phi_deg)
    L_eff   = x_initial + (r_g_RE * np.sin(phi_rad))

    T_gyro_physics = 2.0 * np.pi * (L_eff ** 3)
    return L_eff, gamma, T_gyro_physics


# ---------------------------------------------------------------------------
# Stage 1: Load, merge, validate, print  (shared across all field types)
# ---------------------------------------------------------------------------

def load_config(conf_file):
    """
    Load a run YAML, merge with base config, validate keys, and print.

    Base config discovery:
      1. If the run YAML contains a 'base_config' key, that path is used.
      2. Otherwise, base.yml in the SAME directory as the run YAML is used.
         This makes subdirectory layout (configs/dipole/, configs/constB/, …)
         work automatically.

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
        if base_path and str(base_path).lower() != "none":
            if not os.path.isabs(base_path):
                base_path = os.path.join(os.path.dirname(os.path.abspath(conf_file)), base_path)
        else:
            base_path = None  # self-contained config (e.g. copied to data/)
    else:
        # Look for base.yml in the same directory as the run config
        run_dir = os.path.dirname(os.path.abspath(conf_file))
        base_path = os.path.join(run_dir, "base.yml")

    if base_path is not None:
        with open(base_path, "r") as f:
            base_cfg = yaml.safe_load(f)

        log.append("=" * 60)
        log.append(f"Base config: {base_path}")
        log.append("=" * 60)
        log.append(yaml.dump(base_cfg, default_flow_style=False, sort_keys=False).rstrip())
        log.append("")
    else:
        base_cfg = {}
        log.append("Base config: none (self-contained)")
        log.append("")

    # --- Validate: warn about unknown keys ---
    base_keys = _collect_keys(base_cfg) if base_cfg else _collect_keys(run_cfg)
    run_keys  = _collect_keys(run_cfg)
    unknown   = run_keys - base_keys
    _known_extras = {
        "total_steps", "norm_time_override", "base_config",
        "output_folder", "run_storage",
    }
    real_unknown = {k for k in unknown if k.split(".")[0] not in _known_extras}
    if real_unknown:
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

    # --- Validate critical values (shared across field types) ---
    particle = cfg.get("particle", "").strip().lower()
    if particle not in ("proton", "p", "electron", "e"):
        raise ValueError(f"Unknown particle type: {cfg.get('particle')!r}. Use 'proton' or 'electron'.")

    energy = cfg.get("energy_eV")
    if energy is not None and float(energy) <= 0:
        raise ValueError(f"energy_eV must be positive, got {energy}")

    # --- Append git commit hash for reproducibility ---
    import subprocess
    try:
        _hash = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=os.path.dirname(os.path.abspath(conf_file)),
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        # Check for uncommitted changes
        _dirty = subprocess.call(
            ["git", "diff", "--quiet"],
            cwd=os.path.dirname(os.path.abspath(conf_file)),
            stderr=subprocess.DEVNULL,
        )
        if _dirty:
            _hash += " (dirty — uncommitted changes)"
        log.append(f"git commit: {_hash}")
    except (subprocess.CalledProcessError, FileNotFoundError):
        log.append("git commit: unavailable")
    log.append("")

    # --- Inject metadata for downstream use ---
    cfg["_config_name"] = os.path.splitext(os.path.basename(conf_file))[0]
    cfg["_config_log"]  = log

    return cfg


# ---------------------------------------------------------------------------
# Stage 2a: Compute derived — dipole
# ---------------------------------------------------------------------------

def compute_derived_dipole(cfg, npfloat=np.float64):
    """
    Dipole-specific derived quantities.

    Parameters
    ----------
    cfg : dict
        Raw config from load_config().
    npfloat : dtype
        Floating-point type (np.float64 or np.float128).

    Returns
    -------
    dict
        Flat params dict ready for dipoleB.py.
    """

    mass_si = _resolve_mass(cfg["particle"])
    config_name = cfg.get("_config_name", "default")
    output_folder, run_storage = _resolve_output_paths(config_name, field_prefix="dipoleB")

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
    round_dec = cfg.get("round_decimals", 1)
    raw_steps = _compute_step_sizes(T_gyro, {
        "ps":  spg.get("ps", 65),
        "rk4": spg.get("rk4", 65),
        "rkg": spg.get("rkg", 65),
    }, round_decimals=round_dec)
    raw_steps = _apply_step_overrides(raw_steps, cfg.get("step_overrides", {}), npfloat)

    ps_step  = npfloat(raw_steps["ps"])
    rk4_step = npfloat(raw_steps["rk4"])
    rkg_step = npfloat(raw_steps["rkg"])

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
        "ps_step":              ps_step,
        "rk4_step":             rk4_step,
        "rkg_step":             rkg_step,
        "N_STEPS_PER_GYRO_ps":  spg.get("ps", 65),
        "N_STEPS_PER_GYRO_rk4": spg.get("rk4", 65),
        "N_STEPS_PER_GYRO_rkg": spg.get("rkg", 65),

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
        "USE_FLOAT128":    cfg.get("use_float128", False),
        "CACHE_VELOCITY_RTOL": cfg.get("cache_velocity_rtol", 0.005),
        "PLOT_BOUNDARY_PAD":   cfg.get("plot_boundary_pad", 1.1),

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


# Backward compatibility alias
compute_derived = compute_derived_dipole


# ---------------------------------------------------------------------------
# Stage 2b: Compute derived — constant B
# ---------------------------------------------------------------------------

def compute_derived_constB(cfg, npfloat=np.float64):
    """
    Constant-B specific derived quantities.

    T_gyro = 2π in normalized units (time is normalized by τ = m/qB).

    Parameters
    ----------
    cfg : dict
        Raw config from load_config().
    npfloat : dtype
        Floating-point type (np.float64 or np.float128).

    Returns
    -------
    dict
        Flat params dict ready for constB.py.
    """

    mass = _resolve_mass(cfg["particle"])
    config_name = cfg.get("_config_name", "default")
    output_folder, run_storage = _resolve_output_paths(config_name, field_prefix="constB")

    # --- Physics seeds ---
    pitch_deg   = npfloat(cfg["pitch_deg"])
    phi_deg     = npfloat(cfg["phi_deg"])
    KE_particle = npfloat(cfg["energy_eV"])
    x_initial   = npfloat(cfg.get("x_initial", 0.0))
    y_initial   = npfloat(cfg.get("y_initial", 0.0))
    z_initial   = npfloat(cfg.get("z_initial", 0.0))
    Bfield_si   = np.array(cfg["Bfield_si"], dtype=npfloat)

    # --- T_gyro = 2π in normalized time ---
    T_gyro = 2.0 * np.pi

    # --- Step sizes ---
    spg = cfg.get("steps_per_gyro", {})
    round_dec = cfg.get("round_decimals", 3)
    raw_steps = _compute_step_sizes(T_gyro, {
        "ps":  spg.get("ps", 100),
        "rk4": spg.get("rk4", 100),
    }, round_decimals=round_dec)
    raw_steps = _apply_step_overrides(raw_steps, cfg.get("step_overrides", {}), npfloat)

    ps_step  = npfloat(raw_steps["ps"])
    rk4_step = npfloat(raw_steps["rk4"])

    # --- Integration time ---
    gyroperiods = npfloat(cfg["gyroperiods"])
    norm_time   = npfloat(gyroperiods) * T_gyro

    # --- Tolerance ---
    tol = 1.0 * np.finfo(npfloat).eps

    # --- Plotting ---
    plot_cfg = cfg.get("plotting", {})

    # --- Solvers ---
    solvers = cfg.get("solvers", {})

    params = {
        # Toggles
        "READ_DATA":       cfg.get("read_data", False),
        "WRITE_DATA":      cfg.get("write_data", True),
        "USE_RK45":        solvers.get("rk45", True),
        "USE_RK4":         solvers.get("rk4", True),
        "USE_ANALYTICAL":  solvers.get("analytical", True),
        "USE_FLOAT128":    cfg.get("use_float128", False),

        # Plotting
        "USE_PLOT_TITLES": plot_cfg.get("titles", True),
        "USE_FULL_PLOT":   plot_cfg.get("full_plot", True),
        "gyro_plot_slice": plot_cfg.get("gyro_plot_slice", 1.5),

        # External h5
        "USE_EXTERNAL_H5":  cfg.get("use_external_h5", False),
        "USE_EXTERNAL_H5b": cfg.get("use_external_h5b", False),
        "external_h5":      cfg.get("external_h5"),
        "external_h5b":     cfg.get("external_h5b"),
        "PS_order_ext":     cfg.get("external_h5_ps_order"),
        "PS_order_extb":    cfg.get("external_h5b_ps_order"),

        # Output
        "output_folder": output_folder,
        "run_storage":   run_storage,

        # Physics
        "particle":    cfg["particle"],
        "pitch_deg":   pitch_deg,
        "phi_deg":     phi_deg,
        "KE_particle": KE_particle,
        "mass":        mass,
        "Bfield_si":   Bfield_si,
        "x_initial":   x_initial,
        "y_initial":   y_initial,
        "z_initial":   z_initial,

        # Timing
        "T_gyro":      T_gyro,
        "gyroperiods": gyroperiods,
        "norm_time":   norm_time,

        # Steps
        "ps_step":  ps_step,
        "rk4_step": rk4_step,

        # Settings
        "PS_order":  cfg.get("ps_order", 40),
        "tol":       tol,
        "rtol_rk45": cfg.get("rtol_rk45", 1e-8),
        "atol_rk45": cfg.get("atol_rk45", 1e-10),
    }

    return params


# ---------------------------------------------------------------------------
# Stage 2c: Compute derived — hyperbolic B
# ---------------------------------------------------------------------------

def compute_derived_hyper(cfg, npfloat=np.float64):
    """
    Hyperbolic-B specific derived quantities.

    T_gyro = 2π in normalized units (time is normalized by τ = m/qB₀).

    Parameters
    ----------
    cfg : dict
        Raw config from load_config().
    npfloat : dtype
        Floating-point type (np.float64 or np.float128).

    Returns
    -------
    dict
        Flat params dict ready for hyperB.py.
    """

    mass_si = _resolve_mass(cfg["particle"])
    config_name = cfg.get("_config_name", "default")
    output_folder, run_storage = _resolve_output_paths(config_name, field_prefix="hyperB")

    # --- Physics seeds ---
    pitch_deg    = npfloat(cfg["pitch_deg"])
    phi_deg      = npfloat(cfg["phi_deg"])
    KE_particle  = npfloat(cfg["energy_eV"])
    delta        = cfg["delta"]
    B_0          = npfloat(cfg["B_0"])
    x_initial_si = npfloat(cfg.get("x_initial_si", 0.0))
    y_initial_si = npfloat(cfg.get("y_initial_si", 0.0))
    z_initial_si = npfloat(cfg.get("z_initial_si", 0.0))

    # --- T_gyro = 2π in normalized time ---
    T_gyro = 2.0 * np.pi

    # --- Step sizes ---
    spg = cfg.get("steps_per_gyro", {})
    round_dec = cfg.get("round_decimals", 3)
    raw_steps = _compute_step_sizes(T_gyro, {
        "ps":  spg.get("ps", 100),
        "rk4": spg.get("rk4", 100),
    }, round_decimals=round_dec)
    raw_steps = _apply_step_overrides(raw_steps, cfg.get("step_overrides", {}), npfloat)

    ps_step  = npfloat(raw_steps["ps"])
    rk4_step = npfloat(raw_steps["rk4"])

    # --- Integration time ---
    gyroperiods = npfloat(cfg["gyroperiods"])
    norm_time   = npfloat(gyroperiods) * T_gyro

    # --- Tolerance ---
    tol = 1.0 * np.finfo(npfloat).eps

    # --- Plotting ---
    plot_cfg = cfg.get("plotting", {})
    window_gyro = plot_cfg.get("window_gyroperiods", 8)
    window_duration = npfloat(window_gyro * 2 * np.pi)

    # --- Solvers ---
    solvers = cfg.get("solvers", {})

    params = {
        # Toggles
        "READ_DATA":       cfg.get("read_data", True),
        "WRITE_DATA":      cfg.get("write_data", True),
        "USE_RK45":        solvers.get("rk45", True),
        "USE_RK4":         solvers.get("rk4", True),
        "USE_FLOAT128":    cfg.get("use_float128", False),

        # Plotting
        "USE_PLOT_TITLES":    plot_cfg.get("titles", True),
        "USE_FULL_PLOT":      plot_cfg.get("full_plot", False),
        "window_duration":    window_duration,
        "slice_mode":         plot_cfg.get("slice_mode", "last"),
        "skip_rk4_slice":     plot_cfg.get("skip_rk4_slice", False),
        "slice_ylim":         plot_cfg.get("slice_ylim"),
        "slice_ylim_top":     plot_cfg.get("slice_ylim_top"),
        "slice_equal_aspect": plot_cfg.get("slice_equal_aspect", False),
        "energy_xlim_left":   plot_cfg.get("energy_xlim_left"),

        # External h5
        "USE_EXTERNAL_H5":  cfg.get("use_external_h5", False),
        "USE_EXTERNAL_H5b": cfg.get("use_external_h5b", False),
        "external_h5":      cfg.get("external_h5"),
        "external_h5b":     cfg.get("external_h5b"),
        "PS_order_ext":     cfg.get("external_h5_ps_order"),
        "PS_order_extb":    cfg.get("external_h5b_ps_order"),

        # Output
        "output_folder": output_folder,
        "run_storage":   run_storage,

        # Physics
        "particle":     cfg["particle"],
        "pitch_deg":    pitch_deg,
        "phi_deg":      phi_deg,
        "KE_particle":  KE_particle,
        "mass_si":      mass_si,
        "delta":        delta,
        "B_0":          B_0,
        "x_initial_si": x_initial_si,
        "y_initial_si": y_initial_si,
        "z_initial_si": z_initial_si,

        # Timing
        "T_gyro":      T_gyro,
        "gyroperiods": gyroperiods,
        "norm_time":   norm_time,

        # Steps
        "ps_step":  ps_step,
        "rk4_step": rk4_step,

        # Settings
        "PS_order":  cfg.get("ps_order", 40),
        "tol":       tol,
        "rtol_rk45": cfg.get("rtol_rk45", 1e-12),
        "atol_rk45": cfg.get("atol_rk45", 1e-14),
    }

    return params
