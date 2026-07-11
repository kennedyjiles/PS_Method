"""
writers.py — Consolidated I/O for all field types (dipoleb, constb, hyperb).

Shared utilities:
    _to_serializable   — JSON encoder for numpy scalars (including float128)
    run_hash           — deterministic SHA-1 (truncated to 6 hex chars) hash of a run-parameter dict
    h5_path_for        — build cache file path from a hash string + output folder
    stem_from_h5       — extract the canonical run stem from an h5 path (strips _full)
    build_filename     — assemble a figure/output path from stem + tag
    summarize          — mean / max / rms statistics for an error array
    summarize_to_file  — write summarize() output to an open file handle
    write_dict         — pretty-print a nested dict to a file handle

Field-specific run-param builders (saved as params_json in h5 for manual-
mode identity recovery — not used for cache hashing; that's physics_hash):
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
    expand_h5_to_full   — expand compact 9-row h5 array to full 17-row state
    _tail_start_index   — start index for tail-end sampling of a time series
    master_csv          — aggregate multi-run results into a CSV table
"""

import os
import gc
import re
import glob
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
    if isinstance(x, np.floating):
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


def h5_path_for(hash_str, output_folder):
    """Build the cache file path for a given physics hash.

    Original (full-run) h5 files carry a `_full` suffix to visually
    distinguish them from trimmed variants (`_first_<window>s.h5`,
    `_last_<window>s.h5`). The hash itself does NOT include the suffix.
    """
    return os.path.join(output_folder, f"{hash_str}_full.h5")


def stem_from_h5(h5_path):
    """Get the run stem from an h5 path, stripping the trailing `_full`
    suffix on original-run files so the run folder name is just the hash.

    Trimmed files (`<hash>_first_60.0s.h5` etc.) don't have `_full` and
    keep their full stem unchanged so the trim replot folder is named
    distinctly from the original run folder.
    """
    s = os.path.splitext(os.path.basename(h5_path))[0]
    if s.endswith("_full"):
        s = s[:-len("_full")]
    return s


# ---------------------------------------------------------------------------
# Segmented-checkpoint runs (see run_ps_streaming_with_decimation + driver)
# ---------------------------------------------------------------------------
# A long run is split into fixed-size segments, each written to its own
# cleanly-closed <hash>_segNNN.h5 (via a .building temp + atomic rename), then
# stitched into <hash>_full.h5 as an HDF5 Virtual Dataset. A crash costs at
# most the current segment; relaunching resumes from the last committed one.

def seg_path_for(hash_str, seg_index, output_folder):
    """Path for one checkpoint segment: <hash>_seg{NNN}.h5."""
    return os.path.join(output_folder, f"{hash_str}_seg{int(seg_index):03d}.h5")


def find_committed_segments(hash_str, output_folder):
    """Return sorted [(index, path)] of cleanly-committed segment files.

    A segment counts as committed only if it opens AND carries the
    ``ps/end_state`` handoff dataset. Half-written ``.building`` temps and
    crash-truncated files are skipped, so resume always restarts from the
    last fully-flushed boundary.
    """
    pattern = os.path.join(output_folder, f"{hash_str}_seg*.h5")
    rx = re.compile(rf"{re.escape(hash_str)}_seg(\d+)\.h5$")
    out = []
    for path in glob.glob(pattern):
        m = rx.search(os.path.basename(path))
        if not m:
            continue
        try:
            with h5py.File(path, "r") as f:
                if "ps/end_state" not in f:
                    continue
        except OSError:
            continue  # unreadable / truncated → treat as not committed
        out.append((int(m.group(1)), path))
    out.sort()
    return out


def contiguous_committed_segments(hash_str, output_folder):
    """Committed segments forming the unbroken prefix 0, 1, 2, … (stops at the
    first gap).

    Deleting any segment — trailing or middle — shortens this prefix, so resume
    refills from the gap and the VDS never stitches around a hole. Orphan
    segments past a gap are ignored (and get overwritten when the run refills).
    """
    by_idx = dict(find_committed_segments(hash_str, output_folder))
    out = []
    k = 0
    while k in by_idx:
        out.append((k, by_idx[k]))
        k += 1
    return out


def latest_committed_segment(hash_str, output_folder):
    """The highest-indexed committed segment present, or None.

    Unlike contiguous_committed_segments this does NOT require the earlier
    segments to be present — it's for the 'offload as you go' workflow, where
    completed segments are moved to other storage to free local space. Resume
    only needs the last segment's end_state to continue, so the highest local
    segment is a sufficient (and consistency-checked) restart point.
    """
    segs = find_committed_segments(hash_str, output_folder)
    return segs[-1] if segs else None


def vds_has_missing_sources(h5_path):
    """True iff *h5_path* is a VDS whose backing segment files are (partly) gone.

    A VDS silently returns fill-value (0) for absent source files, so a dangling
    stitch reads as corrupt-but-valid data (all-zero position columns → r=0 →
    NaNs downstream). Callers use this to refuse to trust such a file. Returns
    False for non-VDS files, unreadable files, or fully-intact VDSes.
    """
    if not os.path.exists(h5_path):
        return False
    folder = os.path.dirname(os.path.abspath(h5_path))
    try:
        with h5py.File(h5_path, "r") as f:
            if "ps/y" not in f or not f["ps/y"].is_virtual:
                return False
            for vs in f["ps/y"].virtual_sources():
                fn = vs.file_name
                if os.path.isabs(fn) and os.path.exists(fn):
                    continue
                if not os.path.exists(os.path.join(folder, os.path.basename(fn))):
                    return True
    except (OSError, KeyError, RuntimeError):
        return False
    return False


def synthesize_full_summary(hash_str, output_folder, full_path=None):
    """Reconstruct run-level ``summary_json`` + ``meta`` group on a stitched
    ``_full.h5`` from the per-segment summaries.

    The driver writes the authoritative summary via append_results_h5_dipoleb
    at run end; this exists for VDS files rebuilt OUTSIDE the driver (the
    scripts/build_vds.py CLI — e.g. after segments were moved to another
    drive), which would otherwise lack the identity metadata that manual
    loading and trim_h5 require.

    Identity fields are taken from segment 0's summary; span fields
    (gyroperiods, norm_time, physical_time, ps.steps) are summed over the
    stitched segments so a partial (contiguous-prefix) stitch is described
    accurately; ps.max_ps is the max and timing.ps the sum. Does nothing if
    the file already has a summary (driver flow stays authoritative) or if
    segments predate per-segment summaries.

    Returns True if a summary was written.
    """
    if full_path is None:
        full_path = h5_path_for(hash_str, output_folder)
    if not os.path.exists(full_path):
        return False
    segs = contiguous_committed_segments(hash_str, output_folder)
    if not segs:
        return False

    with h5py.File(full_path, "r") as f:
        if "summary_json" in f.attrs:
            return False  # driver-written summary present — leave it alone

    per_seg = []
    for _, path in segs:
        with h5py.File(path, "r") as f:
            if "summary_json" not in f.attrs:
                return False  # pre-2026-07 segments — can't synthesize
            per_seg.append(json.loads(f.attrs["summary_json"]))

    s = json.loads(json.dumps(per_seg[0]))  # deep copy
    meta = s["meta"]
    meta["stem"] = hash_str
    meta.pop("segment_index", None)
    for k in ("gyroperiods", "norm_time", "physical_time"):
        meta[k] = float(sum(p["meta"].get(k) or 0.0 for p in per_seg))
    meta["timing"] = {"ps": float(sum(p["meta"].get("timing", {}).get("ps") or 0.0
                                      for p in per_seg))}
    s["ps"]["steps"] = int(sum(p["ps"].get("steps") or 0 for p in per_seg))
    s["ps"]["max_ps"] = int(max(p["ps"].get("max_ps") or 0 for p in per_seg))

    with h5py.File(full_path, "a") as f:
        f.attrs["summary_json"] = json.dumps(s)
        gmeta = f.require_group("meta")
        gmeta.attrs["norm_time"] = meta["norm_time"]
        gmeta.attrs["physical_time"] = meta["physical_time"]
        gmeta.attrs["percent_c"] = float(meta.get("percent_c") or 0.0)
        gmeta.attrs["timing_ps"] = meta["timing"]["ps"]
    return True


def clear_building_segments(hash_str, output_folder):
    """Delete stray ``*.building`` temps left by a crashed segment/VDS write."""
    for pat in (f"{hash_str}_seg*.h5.building", f"{hash_str}_full.h5.building"):
        for path in glob.glob(os.path.join(output_folder, pat)):
            try:
                os.remove(path)
            except OSError:
                pass


def build_vds(hash_str, output_folder, verify_chain=True):
    """(Re)build <hash>_full.h5 as a Virtual Dataset over committed segments.

    The virtual ``ps/y`` / ``ps/orders`` datasets concatenate each segment's
    data along the time axis, so existing readers open <hash>_full.h5 and slice
    it exactly as a single-file run. Physics attrs are copied from the first
    segment; run-level stats (max_ps, mean_ps, hit_*) are aggregated.

    Source paths are stored as basenames (folder-relative) so the run folder
    stays movable. Refuses to overwrite a real (non-virtual) <hash>_full.h5 so
    a segmented test can never clobber a pre-existing single-file dataset.

    Returns the VDS path, or None if there are no committed segments.
    """
    # Only the unbroken 0,1,2,… prefix — never stitch around a missing segment.
    segs = contiguous_committed_segments(hash_str, output_folder)
    if not segs:
        return None

    n_save = y_dtype = o_dtype = base_attrs = None
    cols = []                     # (basename, ncols) per segment, in order
    total_cols = 0
    last_end_index = 0
    agg = dict(max_ps=0, sum_orders=0, count_orders=0, total_steps=0,
               total_substeps=0, total_rejections=0,
               hit_atmosphere=False, hit_atm_step=-1, hit_atm_r=0.0)
    prev_end = None
    for idx, path in segs:
        with h5py.File(path, "r") as f:
            ps = f["ps"]
            c = ps["y"].shape[1]
            if base_attrs is None:
                n_save = ps["y"].shape[0]
                y_dtype = ps["y"].dtype
                o_dtype = ps["orders"].dtype
                base_attrs = dict(ps.attrs)
            start_state = ps["start_state"][()]
            end_state = ps["end_state"][()]
            if verify_chain and prev_end is not None and not np.array_equal(prev_end, start_state):
                raise RuntimeError(
                    f"broken checkpoint chain: segment {idx} start_state does "
                    f"not match the previous segment's end_state")
            prev_end = end_state
            last_end_index = int(ps.attrs.get("end_global_index", last_end_index))
            agg["max_ps"] = max(agg["max_ps"], int(ps.attrs.get("max_ps", 0)))
            agg["sum_orders"]   += int(ps.attrs.get("sum_orders", 0))
            agg["count_orders"] += int(ps.attrs.get("count_orders", 0))
            agg["total_steps"]  += int(ps.attrs.get("steps", 0))
            agg["total_substeps"]   += int(ps.attrs.get("total_substeps", 0))
            agg["total_rejections"] += int(ps.attrs.get("total_rejections", 0))
            if bool(ps.attrs.get("hit_atmosphere", False)):
                s = int(ps.attrs.get("hit_atm_step", -1))
                r = float(ps.attrs.get("hit_atm_r", 0.0))
                if not agg["hit_atmosphere"]:
                    agg.update(hit_atmosphere=True, hit_atm_step=s, hit_atm_r=r)
                else:
                    agg["hit_atm_step"] = min(agg["hit_atm_step"], s)
                    agg["hit_atm_r"]    = min(agg["hit_atm_r"], r)
        cols.append((os.path.basename(path), c))
        total_cols += c

    full = h5_path_for(hash_str, output_folder)
    if os.path.exists(full):
        try:
            with h5py.File(full, "r") as f:
                if "ps/y" in f and not f["ps/y"].is_virtual:
                    raise RuntimeError(
                        f"refusing to overwrite non-virtual {os.path.basename(full)} "
                        f"(a real single-file run already lives there)")
        except OSError:
            pass  # unreadable existing file → safe to replace

    y_layout = h5py.VirtualLayout(shape=(n_save, total_cols), dtype=y_dtype)
    o_layout = h5py.VirtualLayout(shape=(total_cols,), dtype=o_dtype)
    col = 0
    for base, c in cols:
        y_layout[:, col:col + c] = h5py.VirtualSource(base, "ps/y", shape=(n_save, c))
        o_layout[col:col + c]    = h5py.VirtualSource(base, "ps/orders", shape=(c,))
        col += c

    # per-segment-only attrs that would be misleading on the stitched file
    for k in ("segment_index", "start_global_index", "end_global_index"):
        base_attrs.pop(k, None)

    tmp = full + ".building"
    with h5py.File(tmp, "w") as f:
        ps = f.create_group("ps")
        for k, v in base_attrs.items():
            ps.attrs[k] = v
        ps.attrs["t0"]                 = 0.0
        ps.attrs["steps"]              = int(agg["total_steps"])
        ps.attrs["max_ps"]             = int(agg["max_ps"])
        ps.attrs["mean_ps"]            = (agg["sum_orders"] / agg["count_orders"]
                                          if agg["count_orders"] > 0 else 0.0)
        ps.attrs["sum_orders"]         = int(agg["sum_orders"])
        ps.attrs["count_orders"]       = int(agg["count_orders"])
        ps.attrs["hit_atmosphere"]     = agg["hit_atmosphere"]
        ps.attrs["hit_atm_step"]       = agg["hit_atm_step"]
        ps.attrs["hit_atm_r"]          = agg["hit_atm_r"]
        ps.attrs["segmented"]          = True
        ps.attrs["n_segments"]         = len(segs)
        if "total_substeps" in base_attrs:      # adaptive runs only
            ps.attrs["total_substeps"]   = int(agg["total_substeps"])
            ps.attrs["total_rejections"] = int(agg["total_rejections"])
        ps.attrs["start_global_index"] = 0
        ps.attrs["end_global_index"]   = int(last_end_index)
        ps.create_virtual_dataset("y", y_layout)
        ps.create_virtual_dataset("orders", o_layout)
    os.replace(tmp, full)
    return full


def build_filename(output_folder, stem, figure_tag, ext="png"):
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

def get_run_params_constb(USE_RK45, USE_RK4, KE_particle, rtol_rk45, atol_rk45,
                          mass_si, q_e, B_0,
                          x_initial, y_initial, z_initial,
                          pitch_deg, phi_deg,
                          norm_time, ps_step, rk4_step,
                          PS_order, tol, charge_sign, dtype):
    """Collect all knobs that define a unique constb run.

    `dtype` is the numpy dtype name (``"float64"`` or ``"float128"``) the
    run is using. Stored in params_json so manual-mode loaders can recover
    the precision without inferring from ``tol``.
    """
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

        "charge_sign": _to_serializable(charge_sign),
        "dtype": str(dtype),
    }


def get_run_params_hyperb(USE_RK45, USE_RK4, KE_particle, rtol_rk45, atol_rk45,
                         mass_si, q_e, B_0, delta,
                         x_initial, y_initial, z_initial,
                         pitch_deg, phi_deg,
                         norm_time, ps_step, rk4_step,
                         PS_order, tol, charge_sign, dtype):
    """Collect all knobs that define a unique hyperb run.

    `dtype` is the numpy dtype name (``"float64"`` or ``"float128"``) the
    run is using. Stored in params_json so manual-mode loaders can recover
    the precision without inferring from ``tol``.
    """
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

        "charge_sign": _to_serializable(charge_sign),
        "dtype": str(dtype),
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
        f.attrs["summary_json"] = json.dumps(summary, default=_to_serializable)

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
    """Load solver arrays and metadata from an HDF5 cache file.

    Note: dipoleb's save writes ``summary_json``, not ``params_json``,
    so no params dict is returned. The cache filename is derived from
    a hash of the run-params dict in the driver, not from anything in
    the file itself, so consumers don't need it.
    """
    with h5py.File(h5_path, "r") as f:
        loaded = {"meta": {"timing": {}}}

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

        # Self-describing PS order summary so external-h5 comparison plots
        # can auto-detect the order without the consuming yml having to
        # spell it out. Both max and mean are written; plot labels use mean.
        if results.get("ps") and results["ps"].get("orders") is not None:
            _orders = np.asarray(results["ps"]["orders"])
            f["ps"].attrs["max_ps"]  = int(_orders.max())
            f["ps"].attrs["mean_ps"] = float(_orders.mean())

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
            for a in g.attrs:
                out[a] = g.attrs[a]
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

        # Self-describing PS order summary so external-h5 comparison plots
        # can auto-detect the order without the consuming yml having to
        # spell it out. Both max and mean are written; plot labels use mean.
        if results.get("ps") and results["ps"].get("orders") is not None:
            _orders = np.asarray(results["ps"]["orders"])
            f["ps"].attrs["max_ps"]  = int(_orders.max())
            f["ps"].attrs["mean_ps"] = float(_orders.mean())

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
            for a in g.attrs:
                out[a] = g.attrs[a]
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
    gyroperiods, norm_time, cache_path,
    # Solver flags
    USE_PS, USE_RK4, USE_RK45, USE_RKG,
    # Step sizes
    ps_step, rk4_step=None, rkg_step=None,
    # Energy drift arrays (already computed)
    rel_drift_ps=None, rel_drift_rk4=None, rel_drift_rk45=None, rel_drift_rkg=None,
    # P_phi drift arrays (already computed)
    rel_pphi_ps=None, rel_pphi_rk4=None, rel_pphi_rk45=None, rel_pphi_rkg=None,
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
        j0_ps = _tail_start_index(rel_drift_ps.size, step_ps, tail_start, MAX_TAIL_STEPS)

    if USE_RK45:
        j0_rk45 = _tail_start_index(len(rel_drift_rk45), ps_step, tail_start, MAX_TAIL_STEPS)

    if USE_RK4:
        j0_rk4 = _tail_start_index(len(rel_drift_rk4), rk4_step, tail_start, MAX_TAIL_STEPS)

    if USE_RKG:
        j0_rkg = _tail_start_index(len(rel_drift_rkg), rkg_step, tail_start, MAX_TAIL_STEPS)

    # --- Write file ---
    output_filename = build_filename(run_folder, stem,
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

        # --- P_phi (canonical angular momentum) tail errors ---
        # Same tail-window indices as the energy block — the arrays are sampled
        # at the same cadence per solver, so reusing j0_* is correct.
        f.write("\n=== |delta P_phi|/|P_phi_0| (tail average) ===\n")
        if USE_RK45 and rel_pphi_rk45 is not None:
            summarize_to_file("RK45", rel_pphi_rk45[j0_rk45:], f)
        if USE_RK4 and rel_pphi_rk4 is not None:
            summarize_to_file("RK4", rel_pphi_rk4[j0_rk4:], f)
        if USE_RKG and rel_pphi_rkg is not None:
            summarize_to_file("RKG", rel_pphi_rkg[j0_rkg:], f)
        if USE_PS and rel_pphi_ps is not None:
            summarize_to_file("PS", rel_pphi_ps[j0_ps:], f)

        # --- Mu tail errors ---
        f.write("\n=== |delta mu|/mu0 (tail average) ===\n")

        if USE_RK45:
            y_tail = y_rk45_common[:, j0_rk45:]
            mu_tail = compute_mu_rk(y_tail.T)
            summarize_to_file("RK45", np.abs(mu_tail - mu0_rk45) / mu0_rk45, f)
            del y_tail, mu_tail
            gc.collect()

        if USE_RK4:
            y_tail = solution_rk4[:, j0_rk4:]
            mu_tail = compute_mu_rk(y_tail.T)
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

            mu_tail = compute_mu_rk(state_tail)
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

            mu_tail = compute_mu_ps(y_tail)
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
    cache_path=None,                # actual h5 path (preferred over stem for filename)
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
            # Use the actual h5 filename (handles _full suffix and trimmed
            # variants correctly); fall back to stem-based name if not given.
            h5_name = os.path.basename(cache_path) if cache_path else f"{stem}.h5"
            f.write(f"Run Data: {h5_name}\n\n")

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
    cache_path=None,                # actual h5 path (preferred over stem for filename)
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
            h5_name = os.path.basename(cache_path) if cache_path else f"{stem}.h5"
            f.write(f"Run Data: {h5_name}\n\n")

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


def _tail_start_index(n_points, step_size, tail_start, max_tail_steps):
    """Start index for the last fraction of a time series, capped at max_tail_steps."""
    j0 = int(tail_start / step_size)
    j0 = max(0, min(j0, n_points - 1))
    if n_points - j0 > max_tail_steps:
        j0 = n_points - max_tail_steps
    # Defensive fallback: if j0 ended up out of bounds, use last NMIN points
    if j0 >= n_points or j0 < 0:
        NMIN = min(1000, n_points)
        j0 = max(0, n_points - NMIN)
    return j0


def master_csv(
    output_folder, stem, particle_type,
    ke_particle, x_initial, y_initial, z_initial, pitch_deg, phi_deg,
    gyroperiods,
    dragt_log,
    method_records,
    bounce_results=None,
    drift_results=None,
):
    """Build records and append to master_simulation_log.csv with duplicate detection."""
    # Trajectory-derived diagnostics (eps_*, hit_atm_*, bounce/drift) come
    # from the PS h5 only — they're meaningless for RK4/RK45/RKG rows. Blank
    # those on non-PS rows. IC-derived dragt fields (L_eff, W0_sq, boundary,
    # mu_sq, orbit_character) are properties of the run setup, not the
    # integrator, so they stay populated on every row.
    _b = bounce_results or {}
    _d = drift_results or {}

    records = []
    for method, steps, dt, e_drift, mu_drift in method_records:
        e = summarize(e_drift)
        mu = summarize(mu_drift)
        is_ps = (method == "PS")

        records.append({
            # --- run identity / setup (all rows) ---
            "run_id": stem,
            "particle": particle_type,
            "energy_eV": ke_particle,
            "gyroperiods": gyroperiods,
            "x": x_initial,
            "y": y_initial,
            "z": z_initial,
            "pitch_deg": pitch_deg,
            "phi_deg": phi_deg,
            # --- IC-derived dragt (all rows) ---
            "L_eff": dragt_log["L_eff"],
            "W0_sq": dragt_log["W0_sq"],
            "boundary": dragt_log["boundary"],
            "mu_sq": dragt_log["mu_sq"],
            "orbit_character": dragt_log["orbit_character"],
            # --- per-method run params + errors ---
            "steps": steps,
            "dt": dt,
            "method": method,
            "energy_mean_err": e["mean"],
            "energy_max_err": e["max"],
            "mu_mean_err": mu["mean"],
            "mu_max_err": mu["max"],
            # --- PS-only trajectory diagnostics (blank on non-PS rows) ---
            "eps_initial":        dragt_log["eps_initial"]    if is_ps else None,
            "eps_mean":           dragt_log["eps_mean"]       if is_ps else None,
            "eps_max":            dragt_log["eps_max"]        if is_ps else None,
            "hit_atmosphere":     dragt_log["hit_atmosphere"] if is_ps else None,
            "hit_atm_r":          dragt_log["hit_atm_r"]      if is_ps else None,
            "n_mirror_crossings": _b.get("n_crossings")       if is_ps else None,
            "bounce_period_s":    _b.get("full_mean_s")       if is_ps else None,
            "bounce_freq_hz":     _b.get("frequency_hz")      if is_ps else None,
            "drift_period_s":     _d.get("period_s")          if is_ps else None,
            "drift_direction":    _d.get("direction")         if is_ps else None,
        })

    df_new = pd.DataFrame(records)
    csv_path = os.path.join(output_folder, "master_simulation_log.csv")
    dup_keys = ["energy_eV", "L_eff", "phi_deg", "pitch_deg", "particle", "method", "steps"]

    if os.path.exists(csv_path):
        # Force string dtype on identifier columns so pandas doesn't auto-cast
        # values like "1e10" or "e25abc" (hex hashes) to scientific-notation floats.
        _str_cols = {
            "run_id":          str,
            "particle":        str,
            "boundary":        str,
            "orbit_character": str,
            "method":          str,
        }
        df_existing = pd.read_csv(csv_path, dtype=_str_cols)
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
