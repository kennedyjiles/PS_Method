"""
YAML config loader for all field types (dipoleb, constb, hyperb).

Two-stage design:
  1. load_config()               — loads YAML, merges with base, validates, prints.
                                   Returns the raw merged dict (no physics computation).
                                   Base config is auto-discovered from the run YAML's directory.
  2. compute_derived_<field>()   — field-specific derived quantities.
                                   Returns a flat params dict ready for the driver script.

Usage:
    cfg    = load_config("configs/dipoleb/demo.yml")
    params = compute_derived_dipoleb(cfg, npfloat=npfloat)

    cfg    = load_config("configs/constb/demo.yml")
    params = compute_derived_constb(cfg, npfloat=npfloat)

    cfg    = load_config("configs/hyperb/demo.yml")
    params = compute_derived_hyperb(cfg, npfloat=npfloat)
"""

import os
import copy
import json
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


def _all_base_keys():
    """Union of keys from every field-type base.yml.

    Used as the validation schema for self-contained (saved) configs that
    have no base merge — without this, typos in those configs go undetected
    because the run config's own keys would be treated as the schema.
    """
    keys = set()
    for field in ("dipoleb", "constb", "hyperb"):
        base_path = os.path.join(_THIS_DIR, field, "base.yml")
        if os.path.isfile(base_path):
            with open(base_path) as f:
                base = yaml.safe_load(f) or {}
            keys |= _collect_keys(base)
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


def _resolve_ext_path(root, path):
    """Resolve an external h5 file path, joining with root if relative.

    If *path* is None/empty, returns None.
    If *path* is already absolute, returns it unchanged.
    Otherwise, joins *root* (which may be None/empty) with *path*.
    Warns when *root* is empty and *path* is relative — the path will be
    resolved against cwd at run time, which is rarely what users want.
    """
    if not path:
        return None
    if os.path.isabs(path):
        return path
    if root:
        return os.path.join(root, path)
    print(f"\n  WARNING: external_h5 path '{path}' is relative with no root; "
          f"will be interpreted from cwd at run time\n")
    return path


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


def _resolve_output_paths(config_name, field_prefix="", output_root=None,
                          batch_group=None):
    """Auto-derive output_folder and run_storage from config name.

    Parameters
    ----------
    config_name  : str   — run config name (e.g. "demo")
    field_prefix : str   — field type (e.g. "dipoleb", "constb", "hyperb")
    output_root  : str or None — optional root path prepended to the
                   default data/<field_prefix>/<config_name> layout.
                   Examples:
                       None              → data/dipoleb/demo/
                       "thesis/chapter3" → thesis/chapter3/data/dipoleb/demo/
    batch_group  : str or None — optional subdirectory between field_prefix
                   and config_name, used by batch_runner.py to keep sweep
                   results separate from single-run configs.
                   Examples:
                       "flux_map" → data/dipoleb/flux_map/E1e+07_L2.00_P45.0/
                       "michel"   → data/dipoleb/michel/E1e+07_L1.50_P85.0_phi30.0/
    """
    parts = ["data"]
    if field_prefix:
        parts.append(field_prefix)
    if batch_group:
        parts.append(batch_group)
    parts.append(config_name)
    data_path = os.path.join(*parts)

    if output_root:
        output_folder = os.path.join(output_root, data_path)
    else:
        output_folder = data_path
    run_storage = os.path.join(output_folder, "_rawdata")
    # Note: directory creation is the caller's (driver's) responsibility — we
    # don't makedirs here so that loading or inspecting a config doesn't
    # silently create empty folders before the run actually executes.
    return output_folder, run_storage


def apply_manual_h5_overrides(cfg, manual_h5_path, field):
    """Override identity fields in cfg from a cached h5's params_json.

    Used by constb / hyperb manual-mode loading. The h5 is authoritative for
    identity (energy, pitch, position, B field, particle, step sizes); the
    yml is for plotting / output knobs only. Mutates ``cfg`` in place and
    forces ``read_data = True`` so the driver takes the cached-load path.

    Parameters
    ----------
    cfg : dict
        Parsed yml config (post base merge). Identity keys are overwritten.
    manual_h5_path : str
        Path to the h5 file. Must contain ``params_json`` at root attrs.
    field : {"constb", "hyperb"}
        Which field's identity schema to use. constb stores ``Bfield_si`` as
        a 3D vector and uses normalized positions (``x_initial``); hyperb
        stores ``B_0`` scalar plus ``delta`` and uses SI-unit positions
        (``x_initial_si``).
    """
    import h5py

    with h5py.File(manual_h5_path, "r") as f:
        if "params_json" not in f.attrs:
            raise RuntimeError(
                f"Manual h5 at {manual_h5_path} has no params_json — "
                "this file was written by an older version."
            )
        p = json.loads(f.attrs["params_json"])

    cfg["energy_eV"] = float(p["KE_particle"])
    cfg["pitch_deg"] = float(p["pitch_deg"])
    cfg["phi_deg"]   = float(p["phi_deg"])
    cfg["particle"]  = "electron" if float(p["qoverm"]) < 0 else "proton"

    if field == "constb":
        cfg["x_initial"] = float(p["x_initial"])
        cfg["y_initial"] = float(p["y_initial"])
        cfg["z_initial"] = float(p["z_initial"])
        # Direction is lost in cache (only B_0 magnitude is stored). Manual
        # mode doesn't run the integrator, so synthesizing [0, 0, B_0] is
        # safe — the only consumer is np.linalg.norm for tau_time/B_0.
        cfg["Bfield_si"] = [0.0, 0.0, float(p["B_0"])]
    elif field == "hyperb":
        cfg["x_initial_si"] = float(p["x_initial"])
        cfg["y_initial_si"] = float(p["y_initial"])
        cfg["z_initial_si"] = float(p["z_initial"])
        cfg["B_0"]          = float(p["B_0"])
        cfg["delta"]        = float(p["delta"])
    else:
        raise ValueError(f"field must be 'constb' or 'hyperb', got {field!r}")

    # Translate norm_time + step sizes back to gyroperiods / step_overrides.
    # T_gyro = 2π in normalized units for both constb and hyperb.
    norm_time = float(p["norm_time"])
    cfg["gyroperiods"] = norm_time / (2.0 * np.pi)

    cfg.setdefault("step_overrides", {})
    cfg["step_overrides"]["ps"]  = float(p["ps_step"])
    cfg["step_overrides"]["rk4"] = float(p["rk4_step"])

    cfg["ps_order"]  = int(p["PS_order"])
    cfg["rtol_rk45"] = float(p["rtol_rk45"])
    cfg["atol_rk45"] = float(p["atol_rk45"])

    cfg["read_data"] = True


def copy_config_to_output(cfg_path, output_folder, cfg=None):
    """Write the fully-merged config into the output folder with the git hash.

    The saved file is self-contained (includes all base defaults) so it can
    be loaded directly from data/ without needing base.yml nearby.  A
    ``base_config: none`` key tells load_config to skip the base merge.

    Parameters
    ----------
    cfg_path      : str – path to the original YAML config (used for the
                    saved-file header and basename).
    output_folder : str – the run's output directory (e.g. data/dipoleb/demo/).
    cfg           : dict, optional – pre-loaded cfg dict reflecting the actual
                    run state (e.g. with manual-mode h5 overrides already
                    applied). When provided, it's used verbatim instead of
                    re-loading from ``cfg_path``. This is what makes the
                    persisted yml truly match what the run executed — without
                    it, the persisted file would still carry the input yml's
                    placeholder identity values rather than the h5's actual
                    ones. Caller is responsible for the contents being
                    accurate.
    """
    import subprocess

    if cfg is None:
        # Re-load via the shared loader so we get merge + validation
        merged = load_config(cfg_path)
    else:
        # Caller passes an already-mutated cfg; deep-copy so we don't mutate
        # their dict when stripping runtime metadata below.
        merged = copy.deepcopy(cfg)

    # Strip runtime metadata that shouldn't be persisted to disk
    merged.pop("_config_name", None)
    merged.pop("_config_log", None)

    # Append git commit hash for reproducibility
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

    # Mark as self-contained so load_config skips the base merge on re-load
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
         This makes subdirectory layout (configs/dipoleb/, configs/constb/, …)
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
    base_keys = _collect_keys(base_cfg) if base_cfg else _all_base_keys()
    run_keys  = _collect_keys(run_cfg)
    unknown   = run_keys - base_keys
    _known_extras = {
        "total_steps", "norm_time_override", "base_config",
        "output_folder", "run_storage", "output_root", "batch_group",
        "git_commit",
    }
    real_unknown = {k for k in unknown if k.split(".")[0] not in _known_extras}
    if real_unknown:
        print(f"\n  WARNING: Unknown config keys (possible typos): {sorted(real_unknown)}")
        print(f"  Valid top-level keys: {sorted(k for k in base_keys if '.' not in k)}\n")
        log.append(f"WARNING: Unknown config keys: {sorted(real_unknown)}")
        log.append("")

    # --- Warn if multiple integration-time specs are set in the run config ---
    # Paper configs that intentionally override (e.g. gyroperiods: null + total_steps: N)
    # do NOT trigger this; only multiple non-null values in the same run YAML do.
    _time_priority = ["norm_time_override", "total_steps", "gyroperiods"]
    _run_set = [k for k in _time_priority if run_cfg.get(k) is not None]
    if len(_run_set) > 1:
        msg = (f"multiple integration-time specs in run config: {_run_set}; "
               f"using {_run_set[0]} (precedence: {' > '.join(_time_priority)})")
        print(f"\n  WARNING: {msg}\n")
        log.append(f"WARNING: {msg}")
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

def compute_derived_dipoleb(cfg, npfloat=None):
    """
    Dipole-specific derived quantities.

    Parameters
    ----------
    cfg : dict
        Raw config from load_config(). Mutated in place: when
        use_gyroradius_L_correction is true, a note is appended to
        cfg["_config_log"] recording the L_eff value used.
    npfloat : dtype, optional
        Floating-point type. If None (default), derived from
        cfg["use_float128"] — np.float128 when true, else np.float64.

    Returns
    -------
    dict
        Flat params dict ready for dipoleb.py.
    """
    if npfloat is None:
        npfloat = np.float128 if cfg["use_float128"] else np.float64

    mass_si = _resolve_mass(cfg["particle"])
    config_name = cfg.get("_config_name", "default")
    output_folder, run_storage = _resolve_output_paths(
        config_name, field_prefix="dipoleb",
        output_root=cfg.get("output_root"),
        batch_group=cfg.get("batch_group"),
    )

    # --- Physics seeds ---
    pitch_deg   = npfloat(cfg["pitch_deg"])
    phi_deg     = npfloat(cfg["phi_deg"])
    x_initial   = npfloat(cfg["x_initial"])
    KE_particle = npfloat(cfg["energy_eV"])
    y_initial   = npfloat(cfg["y_initial"])
    z_initial   = npfloat(cfg["z_initial"])

    # --- T_gyro (relativistic or simple) ---
    if cfg["use_gyroradius_L_correction"]:
        L_eff, gamma, T_gyro = _compute_relativistic_L_eff(
            KE_particle, mass_si, pitch_deg, phi_deg, x_initial)
        cfg.setdefault("_config_log", []).append(
            f"  gyroradius L correction: L_eff {L_eff:.6f} R_E used for T_gyro (launch unchanged at {float(x_initial):.6f})")
    else:
        T_gyro = 2.0 * np.pi * (x_initial ** 3)

    # --- Step sizes ---
    spg = cfg["steps_per_gyro"]
    round_dec = cfg["round_decimals"]
    raw_steps = _compute_step_sizes(T_gyro, {
        "ps":  spg["ps"],
        "rk4": spg["rk4"],
        "rkg": spg["rkg"],
    }, round_decimals=round_dec)
    raw_steps = _apply_step_overrides(raw_steps, cfg["step_overrides"], npfloat)

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
    elif cfg.get("gyroperiods") is not None:
        gyroperiods = npfloat(cfg["gyroperiods"])
        norm_time   = npfloat(gyroperiods) * T_gyro
    else:
        raise ValueError(
            "Specify one of: gyroperiods, total_steps, or norm_time_override."
        )

    # --- Plotting ---
    plot_cfg = cfg["plotting"]

    # --- External h5 ---
    use_ext = cfg["use_external_h5"]
    ext     = cfg["external_h5"]
    ext_root = ext.get("root") or ""
    ext_ps   = _resolve_ext_path(ext_root, ext.get("ps"))
    ext_rk4  = _resolve_ext_path(ext_root, ext.get("rk4"))
    ext_rk45 = _resolve_ext_path(ext_root, ext.get("rk45"))
    ext_rkg  = _resolve_ext_path(ext_root, ext.get("rkg"))

    # --- Solvers ---
    solvers = cfg["solvers"]

    params = {
        # Toggles
        "READ_DATA":       cfg["read_data"],
        "USE_RK45":        solvers["rk45"],
        "USE_RK4":         solvers["rk4"],
        "USE_RKG":         solvers["rkg"],
        "USE_PS":          solvers["ps"],
        "USE_ADAPTIVE":    solvers["adaptive"],
        "PS_decimate":     cfg["ps_decimate"],

        # Initial position
        "y_initial": y_initial,
        "z_initial": z_initial,

        # Plotting
        "USE_PLOT_TITLES": plot_cfg["titles"],
        "USE_FULL_PLOT":   plot_cfg["full_plot"],
        "slice_mode":      plot_cfg["slice_mode"],
        "gyro_window":     plot_cfg["gyro_window"],

        # External h5
        "USE_EXTERNAL_H5_ps":   use_ext["ps"],
        "USE_EXTERNAL_H5_rk4":  use_ext["rk4"],
        "USE_EXTERNAL_H5_rk45": use_ext["rk45"],
        "USE_EXTERNAL_H5_rkg":  use_ext["rkg"],
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
        "N_STEPS_PER_GYRO_ps":  spg["ps"],
        "N_STEPS_PER_GYRO_rk4": spg["rk4"],
        "N_STEPS_PER_GYRO_rkg": spg["rkg"],

        # Plotting windows
        "window_time": npfloat(cfg["window_time"]),
        "N_GYRO":      plot_cfg["n_gyro"],

        # Optional overrides
        "PS_order":        cfg["ps_order"],
        "PS_chunk_steps":  int(cfg["ps_chunk_steps"]),
        "rtol_rk45":       cfg["rtol_rk45"],
        "atol_rk45":       cfg["atol_rk45"],
        "user_min_phase":  cfg["user_min_phase"],
        "MAX_PLOT_POINTS": cfg.get("max_plot_points", 1_000_000),  # not in base.yml
        "USE_FLOAT128":    cfg["use_float128"],
        "CACHE_VELOCITY_RTOL": cfg["cache_velocity_rtol"],
        "PLOT_BOUNDARY_PAD":   cfg["plot_boundary_pad"],

        # Special modes
        "legacy_h5_path": cfg.get("legacy_h5_path"),  # not in base.yml
        "manual_h5_path": cfg["manual_h5_path"],

        # Adaptive PS settings
        "ps_adaptive": cfg["ps_adaptive"],

        # Dragt monitor
        "dragt_monitor_rtol": cfg["dragt_monitor_rtol"],

        # Bounce/drift detection
        "bounce_drift": cfg["bounce_drift"],

        # Atmospheric impact threshold (R_E)
        "r_atmosphere": cfg["r_atmosphere"],
    }

    return params


# Backward compatibility alias
compute_derived = compute_derived_dipoleb


# ---------------------------------------------------------------------------
# Stage 2b: Compute derived — constant B
# ---------------------------------------------------------------------------

def compute_derived_constb(cfg, npfloat=None):
    """
    Constant-B specific derived quantities.

    T_gyro = 2π in normalized units (time is normalized by τ = m/qB).

    Parameters
    ----------
    cfg : dict
        Raw config from load_config().
    npfloat : dtype, optional
        Floating-point type. If None (default), derived from
        cfg["use_float128"] — np.float128 when true, else np.float64.

    Returns
    -------
    dict
        Flat params dict ready for constb.py.
    """
    if npfloat is None:
        npfloat = np.float128 if cfg["use_float128"] else np.float64

    mass = _resolve_mass(cfg["particle"])
    config_name = cfg.get("_config_name", "default")
    output_folder, run_storage = _resolve_output_paths(
        config_name, field_prefix="constb",
        output_root=cfg.get("output_root"),
    )

    # --- Physics seeds ---
    pitch_deg   = npfloat(cfg["pitch_deg"])
    phi_deg     = npfloat(cfg["phi_deg"])
    KE_particle = npfloat(cfg["energy_eV"])
    x_initial   = npfloat(cfg["x_initial"])
    y_initial   = npfloat(cfg["y_initial"])
    z_initial   = npfloat(cfg["z_initial"])
    Bfield_si   = np.array(cfg["Bfield_si"], dtype=npfloat)

    # --- T_gyro = 2π in normalized time ---
    T_gyro = 2.0 * np.pi

    # --- Step sizes ---
    spg = cfg["steps_per_gyro"]
    round_dec = cfg["round_decimals"]
    raw_steps = _compute_step_sizes(T_gyro, {
        "ps":  spg["ps"],
        "rk4": spg["rk4"],
    }, round_decimals=round_dec)
    raw_steps = _apply_step_overrides(raw_steps, cfg["step_overrides"], npfloat)

    ps_step  = npfloat(raw_steps["ps"])
    rk4_step = npfloat(raw_steps["rk4"])

    # --- Integration time ---
    if cfg.get("gyroperiods") is None:
        raise ValueError("gyroperiods must be specified.")
    gyroperiods = npfloat(cfg["gyroperiods"])
    norm_time   = npfloat(gyroperiods) * T_gyro

    # --- Tolerance ---
    tol = 1.0 * np.finfo(npfloat).eps

    # --- Plotting ---
    plot_cfg = cfg["plotting"]

    # --- Solvers ---
    solvers = cfg["solvers"]

    # --- External h5 ---
    ext = cfg["external_h5"]
    ext_root = ext.get("root") or ""
    ext_a = ext.get("a") or {}
    ext_b = ext.get("b") or {}

    params = {
        # Toggles
        "READ_DATA":       cfg["read_data"],
        "WRITE_DATA":      cfg["write_data"],
        "USE_RK45":        solvers["rk45"],
        "USE_RK4":         solvers["rk4"],
        "USE_ANALYTICAL":  solvers["analytical"],
        "USE_FLOAT128":    cfg["use_float128"],

        # Plotting
        "USE_PLOT_TITLES": plot_cfg["titles"],
        "USE_FULL_PLOT":   plot_cfg["full_plot"],
        "gyro_plot_slice": plot_cfg["gyro_plot_slice"],

        # External h5
        "USE_EXTERNAL_H5":  ext_a.get("enabled", False),
        "USE_EXTERNAL_H5b": ext_b.get("enabled", False),
        "external_h5":      _resolve_ext_path(ext_root, ext_a.get("file")),
        "external_h5b":     _resolve_ext_path(ext_root, ext_b.get("file")),
        "PS_order_ext":     ext_a.get("ps_order"),
        "PS_order_extb":    ext_b.get("ps_order"),

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
        "PS_order":  cfg["ps_order"],
        "tol":       tol,
        "rtol_rk45": cfg["rtol_rk45"],
        "atol_rk45": cfg["atol_rk45"],
    }

    return params


# ---------------------------------------------------------------------------
# Stage 2c: Compute derived — hyperbolic B
# ---------------------------------------------------------------------------

def compute_derived_hyperb(cfg, npfloat=None):
    """
    Hyperbolic-B specific derived quantities.

    T_gyro = 2π in normalized units (time is normalized by τ = m/qB₀).

    Parameters
    ----------
    cfg : dict
        Raw config from load_config().
    npfloat : dtype, optional
        Floating-point type. If None (default), derived from
        cfg["use_float128"] — np.float128 when true, else np.float64.

    Returns
    -------
    dict
        Flat params dict ready for hyperb.py.
    """
    if npfloat is None:
        npfloat = np.float128 if cfg["use_float128"] else np.float64

    mass_si = _resolve_mass(cfg["particle"])
    config_name = cfg.get("_config_name", "default")
    output_folder, run_storage = _resolve_output_paths(
        config_name, field_prefix="hyperb",
        output_root=cfg.get("output_root"),
    )

    # --- Physics seeds ---
    pitch_deg    = npfloat(cfg["pitch_deg"])
    phi_deg      = npfloat(cfg["phi_deg"])
    KE_particle  = npfloat(cfg["energy_eV"])
    delta        = cfg["delta"]
    B_0          = npfloat(cfg["B_0"])
    x_initial_si = npfloat(cfg["x_initial_si"])
    y_initial_si = npfloat(cfg["y_initial_si"])
    z_initial_si = npfloat(cfg["z_initial_si"])

    # --- T_gyro = 2π in normalized time ---
    T_gyro = 2.0 * np.pi

    # --- Step sizes ---
    spg = cfg["steps_per_gyro"]
    round_dec = cfg["round_decimals"]
    raw_steps = _compute_step_sizes(T_gyro, {
        "ps":  spg["ps"],
        "rk4": spg["rk4"],
    }, round_decimals=round_dec)
    raw_steps = _apply_step_overrides(raw_steps, cfg["step_overrides"], npfloat)

    ps_step  = npfloat(raw_steps["ps"])
    rk4_step = npfloat(raw_steps["rk4"])

    # --- Integration time ---
    if cfg.get("gyroperiods") is None:
        raise ValueError("gyroperiods must be specified.")
    gyroperiods = npfloat(cfg["gyroperiods"])
    norm_time   = npfloat(gyroperiods) * T_gyro

    # --- Tolerance ---
    tol = 1.0 * np.finfo(npfloat).eps

    # --- Plotting ---
    plot_cfg = cfg["plotting"]
    window_gyro = plot_cfg["window_gyroperiods"]
    window_duration = npfloat(window_gyro * 2 * np.pi)

    # --- Solvers ---
    solvers = cfg["solvers"]

    # --- External h5 ---
    ext = cfg["external_h5"]
    ext_root = ext.get("root") or ""
    ext_a = ext.get("a") or {}
    ext_b = ext.get("b") or {}

    params = {
        # Toggles
        "READ_DATA":       cfg["read_data"],
        "WRITE_DATA":      cfg["write_data"],
        "USE_RK45":        solvers["rk45"],
        "USE_RK4":         solvers["rk4"],
        "USE_FLOAT128":    cfg["use_float128"],

        # Plotting
        "USE_PLOT_TITLES":    plot_cfg["titles"],
        "USE_FULL_PLOT":      plot_cfg["full_plot"],
        "window_duration":    window_duration,
        "slice_mode":         plot_cfg["slice_mode"],
        "skip_rk4_slice":     plot_cfg["skip_rk4_slice"],
        "slice_ylim":         plot_cfg["slice_ylim"],
        "slice_ylim_top":     plot_cfg["slice_ylim_top"],
        "slice_equal_aspect": plot_cfg["slice_equal_aspect"],
        "energy_xlim_left":   plot_cfg["energy_xlim_left"],

        # External h5
        "USE_EXTERNAL_H5":  ext_a.get("enabled", False),
        "USE_EXTERNAL_H5b": ext_b.get("enabled", False),
        "external_h5":      _resolve_ext_path(ext_root, ext_a.get("file")),
        "external_h5b":     _resolve_ext_path(ext_root, ext_b.get("file")),
        "PS_order_ext":     ext_a.get("ps_order"),
        "PS_order_extb":    ext_b.get("ps_order"),

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
        "PS_order":  cfg["ps_order"],
        "tol":       tol,
        "rtol_rk45": cfg["rtol_rk45"],
        "atol_rk45": cfg["atol_rk45"],
    }

    return params
