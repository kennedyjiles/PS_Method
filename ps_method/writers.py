"""
writers.py — Consolidated I/O for all field types (dipoleb, constb, hyperb).

Shared utilities:
    _to_serializable   — JSON encoder for numpy scalars (including float128)
    run_hash           — deterministic SHA-256 hash of a run-parameter dict
    h5_path_for        — build cache file path from params + output folder
    build_filename     — assemble a figure/output path from stem + tag
    summarize          — min/max/mean/final statistics for an error array
    summarize_to_file  — write summarize() output to an open file handle
    write_dict         — pretty-print a nested dict to a file handle

Field-specific run-param builders:
    get_run_params_dipoleb — parameter signature dict for dipole runs
    get_run_params_constb  — parameter signature dict for constant-B runs
    get_run_params_hyperb  — parameter signature dict for hyperbolic-B runs

Field-specific save/load:
    save_results_h5_dipoleb   — write dipole results to h5
    load_results_h5_dipoleb   — read dipole results from h5
    append_results_h5_dipoleb — append solver group to existing dipole h5
    save_results_h5_constb    — write constant-B results to h5
    load_results_h5_constb    — read constant-B results from h5
    save_results_h5_hyperb    — write hyperbolic-B results to h5
    load_results_h5_hyperb    — read hyperbolic-B results from h5

Field-specific summaries:
    summary_txt_dipoleb — human-readable run summary for dipole
    summary_txt_constb  — human-readable run summary for constant-B
    summary_txt_hyperb  — human-readable run summary for hyperbolic-B

Dipoleb-only extras:
    expand_h5_to_full — expand compact 9-row h5 array to full 17-row state
    _make_tail_mask   — boolean mask for tail-end sampling of a time series
    master_csv        — aggregate multi-run results into a CSV table
"""

import os
import gc
import json
import hashlib

import numpy as np
import pandas as pd
import h5py


# =====================================================================
# =====================  Shared Utilities  ============================
# =====================================================================

def _to_serializable(x):
    """Coerce numpy scalars and arrays to native Python types so json.dumps
    doesn't choke on them."""
    if isinstance(x, (np.floating, np.float32, np.float64)):
        return float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.ndarray,)):
        return x.tolist()
    return x


def run_hash(params: dict) -> str:
    """Produce a short unique hash from a run-parameter dict.

    Used as a cache key so that identical configs map to the same h5 file
    and re-running a simulation skips the solver if the cache already exists.
    """
    j = json.dumps(params, sort_keys=True, default=_to_serializable, separators=(",", ":"))
    return hashlib.sha1(j.encode("utf-8")).hexdigest()[:6]


def h5_path_for(params, output_folder):
    """Return the HDF5 cache path for a given run-parameter dict."""
    return os.path.join(output_folder, f"{run_hash(params)}.h5")


def build_filename(summary, output_folder, stem, figure_tag, ext="png"):
    """Build the full path for a figure or output file."""
    return os.path.join(output_folder, f"{stem}_{figure_tag}.{ext}")


def write_dict(f, d, indent=0):
    """Recursively pretty-print a nested dict to a file handle."""
    pad = " " * indent
    for k, v in d.items():
        if isinstance(v, dict):
            f.write(f"{pad}{k}:\n")
            write_dict(f, v, indent + 2)
        else:
            f.write(f"{pad}{k} = {v}\n")


def summarize(err):
    """Return mean / max / rms of |err| as a dict."""
    ae = np.abs(err)
    return {
        "mean": np.mean(ae),
        "max":  np.max(ae),
        "rms":  np.sqrt(np.mean(ae**2)),
    }


def summarize_to_file(label, err, f):
    """Compute summarize(err) and write a formatted line to file handle *f*."""
    s = summarize(err)
    f.write(
        f"  {label:<8}: "
        f"mean = {s['mean']:.2e}, "
        f"max = {s['max']:.2e}, "
        f"rms = {s['rms']:.2e}\n"
    )


# =====================================================================
# =================  Run-param builders  ==============================
# =====================================================================

def get_run_params_dipoleb(USE_RK45, USE_RK4, USE_RKG, USE_PS, decimate, PS_CHUNKING,
                          mass_si, q_e, B_0, gamma, user_min_phase,
                          x_initial, y_initial, z_initial,
                          pitch_deg, phi_deg,
                          norm_time, ps_step, rk4_step, rkg_step,
                          PS_order, tol, qoverm, rtol_rk45, atol_rk45):
    """Collect all knobs that define a unique dipoleb run."""
    return {
        # toggles
        "USE_RK45": bool(USE_RK45),
        "USE_RK4":  bool(USE_RK4),
        "USE_RKG":  bool(USE_RKG),
        "USE_PS":   bool(USE_PS),
        "PS_CHUNKING": bool(PS_CHUNKING),

        # physics & normalization
        "decimate": _to_serializable(decimate),
        "mass_si": _to_serializable(mass_si),
        "q_e": _to_serializable(q_e),
        "B_0": _to_serializable(B_0),
        "gamma": _to_serializable(gamma),
        "user_min_phase": _to_serializable(user_min_phase),

        # initial conditions (positions in RE units and velocity setup)
        "x_initial": _to_serializable(x_initial),
        "y_initial": _to_serializable(y_initial),
        "z_initial": _to_serializable(z_initial),
        "pitch_deg": _to_serializable(pitch_deg),
        "phi_deg": _to_serializable(phi_deg),

        # times / steps
        "norm_time": _to_serializable(norm_time),
        "ps_step": _to_serializable(ps_step),
        "rk4_step": _to_serializable(rk4_step),
        "rkg_step": _to_serializable(rkg_step),

        # PS & solver knobs
        "PS_order": int(PS_order),
        "tol": _to_serializable(tol),
        "rtol_rk45": _to_serializable(rtol_rk45),
        "atol_rk45": _to_serializable(atol_rk45),

        # charge/mass normalization used in RHS
        "qoverm": _to_serializable(qoverm),
    }


def get_run_params_constb(USE_RK45, USE_RK4, KE_particle, rtol_rk45, atol_rk45,
                          mass_si, q_e, B_0,
                          x_initial, y_initial, z_initial,
                          pitch_deg, phi_deg,
                          norm_time, ps_step, rk4_step,
                          PS_order, tol, qoverm):
    """Collect all knobs that define a unique constb run."""
    return {
        "USE_RK45": bool(USE_RK45),
        "USE_RK4":  bool(USE_RK4),

        "KE_particle": _to_serializable(KE_particle),
        "mass_si": _to_serializable(mass_si),
        "q_e": _to_serializable(q_e),
        "B_0": _to_serializable(B_0),

        "x_initial": _to_serializable(x_initial),
        "y_initial": _to_serializable(y_initial),
        "z_initial": _to_serializable(z_initial),
        "pitch_deg": _to_serializable(pitch_deg),
        "phi_deg": _to_serializable(phi_deg),

        "norm_time": _to_serializable(norm_time),
        "ps_step": _to_serializable(ps_step),
        "rk4_step": _to_serializable(rk4_step),

        "PS_order": int(PS_order),
        "tol": _to_serializable(tol),
        "rtol_rk45": _to_serializable(rtol_rk45),
        "atol_rk45": _to_serializable(atol_rk45),

        "qoverm": _to_serializable(qoverm),
    }


def get_run_params_hyperb(USE_RK45, USE_RK4, KE_particle, rtol_rk45, atol_rk45,
                         mass_si, q_e, B_0, delta,
                         x_initial, y_initial, z_initial,
                         pitch_deg, phi_deg,
                         norm_time, ps_step, rk4_step,
                         PS_order, tol, qoverm):
    """Collect all knobs that define a unique hyperb run."""
    return {
        "USE_RK45": bool(USE_RK45),
        "USE_RK4":  bool(USE_RK4),

        "KE_particle": _to_serializable(KE_particle),
        "mass_si": _to_serializable(mass_si),
        "q_e": _to_serializable(q_e),
        "B_0": _to_serializable(B_0),
        "delta": _to_serializable(delta),

        "x_initial": _to_serializable(x_initial),
        "y_initial": _to_serializable(y_initial),
        "z_initial": _to_serializable(z_initial),
        "pitch_deg": _to_serializable(pitch_deg),
        "phi_deg": _to_serializable(phi_deg),

        "norm_time": _to_serializable(norm_time),
        "ps_step": _to_serializable(ps_step),
        "rk4_step": _to_serializable(rk4_step),

        "PS_order": int(PS_order),
        "tol": _to_serializable(tol),
        "rtol_rk45": _to_serializable(rtol_rk45),
        "atol_rk45": _to_serializable(atol_rk45),

        "qoverm": _to_serializable(qoverm),
    }


# =====================================================================
# =================  Save / load  =====================================
# =====================================================================

# Compact h5 storage: only these rows are saved (pos, vel, B-field)
SAVE_ROWS = [0, 1, 2, 3, 4, 5, 14, 15, 16]
n_save = len(SAVE_ROWS)


def save_results_h5_dipoleb(h5_path, results, summary):
    """Write solver arrays and metadata to a new HDF5 cache file."""
    with h5py.File(h5_path, "w") as f:
        f.attrs["summary_json"] = json.dumps(summary)

        for k in ("ps", "rk4", "rk45", "rkg"):
            if k not in results or results[k] is None:
                continue
            grp = f.create_group(k)
            for name, val in results[k].items():
                if val is None:
                    continue
                if isinstance(val, np.ndarray):
                    grp.create_dataset(
                        name, data=val,
                        compression="gzip", compression_opts=1, shuffle=True)
                else:
                    grp.attrs[name] = val

        meta = results.get("meta", {})
        gmeta = f.create_group("meta")
        for mk, mv in meta.get("timing", {}).items():
            gmeta.attrs[f"timing_{mk}"] = float(mv)
        for sk in ("physical_time", "norm_time", "percent_c", "particle_label"):
            if sk in meta:
                gmeta.attrs[sk] = meta[sk]


def load_results_h5_dipoleb(h5_path):
    """Load solver arrays and metadata from an HDF5 cache file."""
    with h5py.File(h5_path, "r") as f:
        loaded = {"meta": {"timing": {}}}

        if "params_json" in f.attrs:
            loaded["params"] = json.loads(f.attrs["params_json"])
        else:
            loaded["params"] = None

        def _read_grp(name):
            if name not in f:
                return None
            g = f[name]
            out = {}
            for ds in g:
                out[ds] = g[ds][...]
            for k, v in g.attrs.items():
                out[k] = v
            return out

        for k in ("ps", "rk4", "rk45", "rkg"):
            loaded[k] = _read_grp(k)

        gmeta = f["meta"]
        for a in gmeta.attrs:
            if a.startswith("timing_"):
                loaded["meta"]["timing"][a.replace("timing_", "")] = gmeta.attrs[a]
            else:
                loaded["meta"][a] = gmeta.attrs[a]

        return loaded


def append_results_h5_dipoleb(h5_path, results, summary):
    """Append non-PS solver results and metadata to an existing HDF5 file.
    Ensures dictionary is written exactly once (for streaming PS files)."""
    with h5py.File(h5_path, "a") as f:
        if "summary_json" not in f.attrs:
            f.attrs["summary_json"] = json.dumps(summary, default=_to_serializable)

        if "meta" not in f:
            gmeta = f.create_group("meta")
        else:
            gmeta = f["meta"]

        for mk, mv in results["meta"]["timing"].items():
            gmeta.attrs[f"timing_{mk}"] = float(mv)

        for sk in ("physical_time", "norm_time", "percent_c", "particle_label"):
            if sk in results["meta"]:
                gmeta.attrs[sk] = results["meta"][sk]

        for k in ("rk4", "rk45", "rkg"):
            if results.get(k) is None:
                continue
            if k in f:
                del f[k]
            grp = f.create_group(k)
            for name, val in results[k].items():
                if isinstance(val, np.ndarray):
                    grp.create_dataset(
                        name, data=val,
                        compression="gzip", compression_opts=1, shuffle=True)
                else:
                    grp.attrs[name] = val

def save_results_h5_constb(h5_path, params, results):
    """Write solver arrays and metadata to a new HDF5 cache file."""
    with h5py.File(h5_path, "w") as f:
        f.attrs["params_json"] = json.dumps(params, sort_keys=True, default=_to_serializable)

        for k in ("ps", "rk4", "rk45"):
            if k in results and results[k] is not None:
                grp = f.create_group(k)
                for name, arr in results[k].items():
                    if arr is None:
                        continue
                    grp.create_dataset(name, data=arr, compression="gzip", compression_opts=1, shuffle=True)

        meta = results.get("meta", {})
        gmeta = f.create_group("meta")
        for mk, mv in meta.get("timing", {}).items():
            gmeta.attrs[f"timing_{mk}"] = float(mv)
        for sk in ("physical_time", "norm_time", "percent_c", "particle_label"):
            if sk in meta:
                gmeta.attrs[sk] = meta[sk]


def load_results_h5_constb(h5_path):
    """Load solver arrays and metadata from an HDF5 cache file."""
    with h5py.File(h5_path, "r") as f:
        loaded = {"meta": {"timing": {}}}
        loaded["params"] = json.loads(f.attrs["params_json"])

        def _read_grp(name):
            if name not in f:
                return None
            g = f[name]
            out = {}
            for ds in g:
                out[ds] = g[ds][...]
            return out

        for k in ("ps", "rk4", "rk45"):
            loaded[k] = _read_grp(k)

        gmeta = f["meta"]
        for a in gmeta.attrs:
            if a.startswith("timing_"):
                loaded["meta"]["timing"][a.replace("timing_", "")] = gmeta.attrs[a]
            else:
                loaded["meta"][a] = gmeta.attrs[a]

        return loaded

def save_results_h5_hyperb(h5_path, params, results):
    """Write solver arrays and metadata to a new HDF5 cache file."""
    with h5py.File(h5_path, "w") as f:
        f.attrs["params_json"] = json.dumps(params, sort_keys=True, default=_to_serializable)

        for k in ("ps", "rk4", "rk45", "rkg"):
            if k in results and results[k] is not None:
                grp = f.create_group(k)
                for name, arr in results[k].items():
                    if arr is None:
                        continue
                    grp.create_dataset(name, data=arr, compression="gzip", compression_opts=1, shuffle=True)

        meta = results.get("meta", {})
        gmeta = f.create_group("meta")
        for mk, mv in meta.get("timing", {}).items():
            gmeta.attrs[f"timing_{mk}"] = float(mv)
        for sk in ("physical_time", "norm_time", "percent_c", "particle_label"):
            if sk in meta:
                gmeta.attrs[sk] = meta[sk]


def load_results_h5_hyperb(h5_path):
    """Load solver arrays and metadata from an HDF5 cache file."""
    with h5py.File(h5_path, "r") as f:
        loaded = {"meta": {"timing": {}}}
        loaded["params"] = json.loads(f.attrs["params_json"])

        def _read_grp(name):
            if name not in f:
                return None
            g = f[name]
            out = {}
            for ds in g:
                out[ds] = g[ds][...]
            return out

        for k in ("ps", "rk4", "rk45", "rkg"):
            loaded[k] = _read_grp(k)

        gmeta = f["meta"]
        for a in gmeta.attrs:
            if a.startswith("timing_"):
                loaded["meta"]["timing"][a.replace("timing_", "")] = gmeta.attrs[a]
            else:
                loaded["meta"][a] = gmeta.attrs[a]

        return loaded


# =====================================================================
# =================  Field-specific summaries  ========================
# =====================================================================

def summary_txt_dipoleb(
    summary, run_folder, stem, dragt_log, bounce_results, drift_results,
    gyroperiods, norm_time, mass, cache_path,
    # Solver flags
    USE_PS, USE_RK4, USE_RK45, USE_RKG, PS_decimate,
    # Step sizes
    ps_step, rk4_step=None, rkg_step=None,
    # Energy drift arrays (already computed)
    rel_drift_ps=None, rel_drift_rk4=None, rel_drift_rk45=None, rel_drift_rkg=None,
    # Mu reference values
    mu0_ps=None, mu0_rk4=None, mu0_rk45=None, mu0_rkg=None,
    # Solver solutions (for mu tail computation)
    solution_rk4=None, solution_rkg=None, y_rk45_common=None,
    # PS storage info
    ps_store_stride=1,
    # npfloat type
    npfloat=np.float64,
    # Physics functions (injected to avoid circular imports)
    compute_mu_ps=None, compute_mu_rk=None, vector_potential=None,
):
    """Write the dipoleb summary text file, including tail-averaged energy
    and mu errors, Dragt diagnostics, and bounce/drift statistics."""
    # --- Tail fraction setup ---
    if gyroperiods < 1e6:
        TAIL_FRAC = 0.01
    else:
        TAIL_FRAC = 0.0001

    tail_start = (1.0 - TAIL_FRAC) * npfloat(norm_time)
    MAX_TAIL_STEPS = 500000

    # --- Build tail masks ---
    j0_ps = j0_rk4 = j0_rk45 = j0_rkg = 0

    if USE_PS:
        step_ps = ps_store_stride * ps_step
        _, j0_ps = _make_tail_mask(rel_drift_ps.size, step_ps, tail_start, MAX_TAIL_STEPS)

    if USE_RK45:
        _, j0_rk45 = _make_tail_mask(len(rel_drift_rk45), ps_step, tail_start, MAX_TAIL_STEPS)

    if USE_RK4:
        _, j0_rk4 = _make_tail_mask(len(rel_drift_rk4), rk4_step, tail_start, MAX_TAIL_STEPS)

    if USE_RKG:
        _, j0_rkg = _make_tail_mask(len(rel_drift_rkg), rkg_step, tail_start, MAX_TAIL_STEPS)

    # --- Write file ---
    output_filename = build_filename(summary, run_folder, stem,
                                     figure_tag="summary", ext="txt")

    with open(output_filename, "w") as f:
        f.write("=== Simulation Summary ===\n")
        write_dict(f, summary)
        f.write("\n")

        # --- Energy tail errors ---
        f.write("\n=== |delta E|/E0 (tail average) ===\n")
        if USE_RK45:
            summarize_to_file("RK45", rel_drift_rk45[j0_rk45:], f)
        if USE_RK4:
            summarize_to_file("RK4", rel_drift_rk4[j0_rk4:], f)
        if USE_RKG:
            summarize_to_file("RKG", rel_drift_rkg[j0_rkg:], f)
        if USE_PS:
            summarize_to_file("PS", rel_drift_ps[j0_ps:], f)

        # --- Mu tail errors ---
        f.write("\n=== |delta mu|/mu0 (tail average) ===\n")

        if USE_RK45:
            y_tail = y_rk45_common[:, j0_rk45:]
            mu_tail = compute_mu_rk(y_tail.T, mass)
            summarize_to_file("RK45", np.abs(mu_tail - mu0_rk45) / mu0_rk45, f)
            del y_tail, mu_tail
            gc.collect()

        if USE_RK4:
            y_tail = solution_rk4[:, j0_rk4:]
            mu_tail = compute_mu_rk(y_tail.T, mass)
            summarize_to_file("RK4", np.abs(mu_tail - mu0_rk4) / mu0_rk4, f)
            del y_tail, mu_tail
            gc.collect()

        if USE_RKG:
            r_tail = solution_rkg[j0_rkg:, 0:3]
            p_tail = solution_rkg[j0_rkg:, 3:6]

            A_tail = np.empty_like(r_tail)
            for i in range(len(r_tail)):
                A_tail[i] = vector_potential(r_tail[i])

            v_tail = p_tail - A_tail
            state_tail = np.hstack((r_tail, v_tail))

            mu_tail = compute_mu_rk(state_tail, mass)
            summarize_to_file("RKG", np.abs(mu_tail - mu0_rkg) / mu0_rkg, f)
            del r_tail, p_tail, A_tail, v_tail, state_tail, mu_tail
            gc.collect()

        if USE_PS:
            step_ps_store = ps_store_stride * ps_step

            with h5py.File(cache_path, "r") as ps_h5:
                ps_y = ps_h5["ps"]["y"]
                n_store = ps_y.shape[1]

                j0 = int(tail_start / step_ps_store)
                j0 = max(0, min(j0, n_store - 1))

                if n_store - j0 > MAX_TAIL_STEPS:
                    j0 = n_store - MAX_TAIL_STEPS

                y_tail = expand_h5_to_full(ps_y[:, j0:])

            mu_tail = compute_mu_ps(y_tail, mass)
            summarize_to_file("PS", np.abs(mu_tail - mu0_ps) / mu0_ps, f)
            del y_tail, mu_tail
            gc.collect()

        # --- Dragt diagnostics ---
        if dragt_log["L_eff"] is not None:
            f.write("\n=== Dragt Diagnostics ===\n")
            f.write(f"Dragt L-shell           : {dragt_log['L_eff']:.4f} R_E\n")
            f.write(f"W0^2                    : {dragt_log['W0_sq']:.8f}\n")
            f.write(f"Boundary status         : {dragt_log['boundary']}\n")
            f.write(f"mu^2 (sin^2 alpha_eq)   : {dragt_log['mu_sq']:.6f}\n")
            f.write(f"Orbit character          : {dragt_log['orbit_character']}\n")
            f.write(f"Adiabaticity (initial)  : {dragt_log['eps_initial']:.4f}\n")
            f.write(f"Adiabaticity (mean)     : {dragt_log['eps_mean']:.4f}\n")
            f.write(f"Adiabaticity (max)      : {dragt_log['eps_max']:.4f}\n")
            if dragt_log["hit_atmosphere"]:
                f.write(f"Atmosphere flag         : HIT (r_min = {dragt_log['hit_atm_r']:.4f} R_E)\n")
            else:
                f.write(f"Atmosphere flag         : CLEAR\n")

        # --- Bounce & drift ---
        if USE_PS:
            f.write("\n=== Bounce and Drift Motion ===\n")

            if bounce_results is None or bounce_results.get("full_mean_s") is None:
                f.write("Bounce: not detected / insufficient mirror crossings\n")
            else:
                f.write(f"Mirror crossings        : {bounce_results['n_crossings']}\n")
                f.write(f"Bounce period (s)       : {bounce_results['full_mean_s']:.6g}\n")
                f.write(f"Bounce frequency (Hz)   : {bounce_results['frequency_hz']:.6g}\n")

            if drift_results is None or drift_results.get("period_s") is None:
                f.write("Drift: not enough azimuthal phase to estimate\n")
            else:
                direction = drift_results.get("direction", 0)
                dir_str = "eastward" if direction > 0 else "westward"

                f.write(f"Drift period (s)        : {drift_results['period_s']:.6g}\n")
                f.write(f"Drift direction         : {dir_str}\n")

            f.write("\n")


def summary_txt_constb(
    output_filename, *,
    # Run identity
    stem=None, WRITE_DATA=False, READ_DATA=False,
    # Particle / field
    particle_type, KE_particle, mass, pitch_deg, phi_deg,
    tau_time, v_tau, gyro_radius_si,
    x_initial, y_initial, z_initial,
    vx_initial, vy_initial, vz_initial,
    Bfield, B_0,
    npfloat_name="float64",
    # Time / stepping
    norm_time, physical_time, gyroperiods,
    ps_step, rk4_step, steps_ps, steps_rk4=None,
    orders_used=None,
    # Solver flags
    USE_RK4=False, USE_RK45=False, USE_ANALYTICAL=False,
    # Timing dict
    timing=None,
    analytical_time=None,
    # Energy drift arrays (already computed, full length)
    rel_drift_ps=None, rel_drift_rk4=None, rel_drift_rk45=None,
):
    """Write a simulation summary text file for a constb run."""
    finalnum = max(1, int(steps_ps * 0.01))

    with open(output_filename, "w") as f:
        if WRITE_DATA or READ_DATA:
            f.write(f"Run Data: {stem}.h5\n\n")

        f.write("=== Simulation Summary ===\n")
        f.write("Initial Conditions:\n")
        f.write(f"  Particle      = {particle_type}\n")
        f.write(f"  Energy        = {KE_particle} eV\n")
        f.write(f"  mass          = {mass} kg\n")
        f.write(f"  pitch_deg     = {pitch_deg}\n")
        f.write(f"  phi_deg       = {phi_deg}\n")
        f.write(f"  tau_time      = {tau_time}\n")
        f.write(f"  v_tau         = {v_tau}\n")
        f.write(f"  gyroradius    = {gyro_radius_si}\n")
        f.write(f"  x_initial     = {x_initial}\n")
        f.write(f"  y_initial     = {y_initial}\n")
        f.write(f"  z_initial     = {z_initial}\n")
        f.write(f"  vx_initial    = {vx_initial}\n")
        f.write(f"  vy_initial    = {vy_initial}\n")
        f.write(f"  vz_initial    = {vz_initial}\n")
        f.write(f"  Bfield        = {Bfield}\n")
        f.write(f"  B_0           = {B_0} T\n")
        f.write(f"  float type    = {npfloat_name}\n\n")

        f.write("=== Timing Summary ===\n")
        if timing:
            f.write(f"  Run Time PS   = {timing['ps']:.2f} s\n")
            if USE_RK4 and "rk4" in timing:
                f.write(f"  Run Time RK4  = {timing['rk4']:.2f} s\n")
            if USE_RK45 and "rk45" in timing:
                f.write(f"  Run Time RK45 = {timing['rk45']:.2f} s\n")
        if USE_ANALYTICAL and analytical_time is not None:
            f.write(f"  Run Time Ana  = {analytical_time:.6f} s\n")
        f.write(f"  norm time     = {norm_time}\n")
        f.write(f"  physical time = {physical_time:.2e} s\n")
        f.write(f"  gyroperiods   = {gyroperiods}\n")
        f.write(f"  ps step size  = {ps_step}\n")
        f.write(f"  ps steps      = {steps_ps}\n")
        if USE_RK4:
            f.write(f"  rk4 step size = {rk4_step}\n")
            if steps_rk4 is not None:
                f.write(f"  rk4 steps     = {steps_rk4}\n")
        if orders_used is not None:
            f.write(f"  PS Orders     = max={orders_used.max()}, mean={orders_used.mean():.1f}\n")
        f.write("\n")

        f.write(f"=== |delta E|/E0 (last {finalnum} steps) ===\n")
        if USE_RK45 and rel_drift_rk45 is not None:
            summarize_to_file("RK45", rel_drift_rk45[-finalnum:], f)
        if USE_RK4 and rel_drift_rk4 is not None:
            summarize_to_file("RK4", rel_drift_rk4[-finalnum:], f)
        if rel_drift_ps is not None:
            summarize_to_file("PS", rel_drift_ps[-finalnum:], f)


def summary_txt_hyperb(
    output_filename, *,
    # Run identity
    stem=None, WRITE_DATA=False, READ_DATA=False,
    # Particle / field
    particle_type, KE_particle, mass_si, pitch_deg, phi_deg,
    tau_time, v_tau, gyro_radius_si,
    x_initial_si, y_initial_si, z_initial_si,
    vx_initial, vy_initial, vz_initial,
    delta, B_0, gamma,
    npfloat_name="float64",
    # Time / stepping
    norm_time, physical_time, gyroperiods,
    ps_step, rk4_step, steps_ps, steps_rk4=None,
    orders_used=None,
    # Solver flags
    USE_RK4=False, USE_RK45=False,
    # Timing dict
    timing=None,
    # Energy drift arrays (already computed, full length)
    rel_drift_ps=None, rel_drift_rk4=None, rel_drift_rk45=None,
):
    """Write a simulation summary text file for a hyperb run."""
    finalnum = max(1, int(steps_ps * 0.01))

    with open(output_filename, "w") as f:
        if WRITE_DATA or READ_DATA:
            f.write(f"Run Data: {stem}.h5\n\n")

        f.write("=== Simulation Summary ===\n")
        f.write("Initial Conditions:\n")
        f.write(f"  particle      = {particle_type}\n")
        f.write(f"  mass          = {mass_si} kg\n")
        f.write(f"  Energy        = {KE_particle} eV\n")
        f.write(f"  pitch_deg     = {pitch_deg}\n")
        f.write(f"  phi_deg       = {phi_deg}\n")
        f.write(f"  tau           = {tau_time} s\n")
        f.write(f"  v_tau         = {v_tau}\n")
        f.write(f"  gyroradius    = {gyro_radius_si} km\n")
        f.write(f"  x_initial     = {x_initial_si} km\n")
        f.write(f"  y_initial     = {y_initial_si} km\n")
        f.write(f"  z_initial     = {z_initial_si} km\n")
        f.write(f"  vx_initial    = {vx_initial}\n")
        f.write(f"  vy_initial    = {vy_initial}\n")
        f.write(f"  vz_initial    = {vz_initial}\n")
        f.write(f"  delta         = {delta} km\n")
        f.write(f"  gamma         = {gamma}\n")
        f.write(f"  B_0           = {B_0} T\n")
        f.write(f"  float type    = {npfloat_name}\n\n")

        f.write("=== Timing Summary ===\n")
        if timing:
            if USE_RK45 and "rk45" in timing:
                f.write(f"  Run Time RK45 = {timing['rk45']:.2f} s\n")
            if USE_RK4 and "rk4" in timing:
                f.write(f"  Run Time RK4  = {timing['rk4']:.2f} s\n")
            f.write(f"  Run Time PS   = {timing['ps']:.2f} s\n")
        if orders_used is not None:
            f.write(f"  PS Orders     = max={orders_used.max()}, mean={orders_used.mean():.1f}\n")
        f.write(f"  norm time     = {norm_time}\n")
        f.write(f"  physical time = {physical_time:.2e} s\n")
        f.write(f"  gyroperiods   = {gyroperiods}\n")
        if USE_RK4:
            f.write(f"  rk4 step size = {rk4_step}\n")
            if steps_rk4 is not None:
                f.write(f"  rk4 steps     = {steps_rk4}\n")
        f.write(f"  ps step size  = {ps_step}\n")
        f.write(f"  ps steps      = {steps_ps}\n\n")

        f.write(f"=== |delta E|/E0 (last {finalnum} steps) ===\n")
        if USE_RK45 and rel_drift_rk45 is not None:
            summarize_to_file("RK45", rel_drift_rk45[-finalnum:], f)
        if USE_RK4 and rel_drift_rk4 is not None:
            summarize_to_file("RK4", rel_drift_rk4[-finalnum:], f)
        if rel_drift_ps is not None:
            summarize_to_file("PS", rel_drift_ps[-finalnum:], f)


# =====================================================================
# =================  Dipoleb-only extras  =============================
# =====================================================================

def expand_h5_to_full(compact_arr):
    """Expand a 9-row compact h5 array back to 17-row full layout.
    If the array already has 17 rows, return it unchanged."""
    if compact_arr.shape[0] == 17:
        return compact_arr
    full = np.zeros((17, compact_arr.shape[1]), dtype=compact_arr.dtype)
    for i_new, i_old in enumerate(SAVE_ROWS):
        full[i_old, :] = compact_arr[i_new, :]
    return full


def _make_tail_mask(n_points, step_size, tail_start, max_tail_steps):
    """Build a boolean mask for the last fraction of a time series."""
    j0 = int(tail_start / step_size)
    j0 = max(0, min(j0, n_points - 1))

    if n_points - j0 > max_tail_steps:
        j0 = n_points - max_tail_steps

    mask = np.zeros(n_points, dtype=bool)
    mask[j0:] = True

    if not np.any(mask):
        NMIN = min(1000, n_points)
        mask[-NMIN:] = True
        j0 = n_points - NMIN

    return mask, j0


def master_csv(
    output_folder, stem, particle_type,
    KE_particle, x_initial, y_initial, z_initial, pitch_deg, phi_deg,
    dragt_log,
    method_records,
):
    """Build records and append to master_simulation_log.csv with duplicate detection."""
    records = []
    for method, steps, dt, e_drift, mu_drift in method_records:
        e = summarize(e_drift)
        mu = summarize(mu_drift)

        records.append({
            "run_id": stem,
            "particle": particle_type,
            "energy_keV": KE_particle,
            "x": x_initial,
            "y": y_initial,
            "z": z_initial,
            "pitch_deg": pitch_deg,
            "phi_deg": phi_deg,
            "L_eff": dragt_log["L_eff"],
            "eps_initial": dragt_log["eps_initial"],
            "eps_mean": dragt_log["eps_mean"],
            "eps_max": dragt_log["eps_max"],
            "W0_sq": dragt_log["W0_sq"],
            "boundary": dragt_log["boundary"],
            "mu_sq": dragt_log["mu_sq"],
            "orbit_character": dragt_log["orbit_character"],
            "hit_atmosphere": dragt_log["hit_atmosphere"],
            "hit_atm_r": dragt_log["hit_atm_r"],
            "steps": steps,
            "dt": dt,
            "method": method,
            "energy_mean_err": e["mean"],
            "energy_max_err": e["max"],
            "energy_rms_err": e["rms"],
            "mu_mean_err": mu["mean"],
            "mu_max_err": mu["max"],
            "mu_rms_err": mu["rms"],
        })

    df_new = pd.DataFrame(records)
    csv_path = f"{output_folder}/master_simulation_log.csv"
    dup_keys = ["energy_keV", "L_eff", "phi_deg", "pitch_deg", "particle", "method", "steps"]

    if os.path.exists(csv_path):
        df_existing = pd.read_csv(csv_path)
        for _, row in df_new.iterrows():
            mask = True
            for k in dup_keys:
                if row[k] is not None and k in df_existing.columns:
                    mask = mask & (df_existing[k] == row[k])
            df_existing = df_existing[~mask]
        df_out = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_out = df_new

    df_out.to_csv(csv_path, index=False)
