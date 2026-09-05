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
from configs.config_loader import load_config, compute_derived_dipoleb as compute_derived, copy_config_to_output, physics_hash


def _stamp_segment_summary(seg_build_path, summary_stub, hash_str, seg_index,
                           seg_steps, total_steps, seg_elapsed):
    """Write a per-segment ``summary_json`` onto a segment file.

    Makes each ``<hash>_segNNN.h5`` manually loadable on its own
    (``manual_h5_path``), which requires the run-identity summary that
    otherwise only lands on ``<hash>_full.h5`` at run end.

    Safety properties:
      * Called on the ``.building`` temp BEFORE the atomic commit — committed
        segment files are never reopened for write, and a crash mid-stamp only
        loses a temp that would be discarded anyway.
      * Root-level attr ONLY. Nothing is added to the ``ps`` group attrs,
        because build_vds copies those onto the stitched _full.h5 and a stray
        per-segment attr there would contaminate it.

    Span-dependent meta fields (gyroperiods, norm_time, physical_time) are
    scaled to the segment's share of the run; ``ps.steps``/``max_ps`` are the
    segment's own. The time axis of a lone segment still starts at 0
    (relative), matching the trimmed-file convention.
    """
    import copy as _copy
    s = _copy.deepcopy(summary_stub)
    frac = seg_steps / float(total_steps)
    meta = s["meta"]
    meta["stem"] = f"{hash_str}_seg{seg_index:03d}"
    for k in ("gyroperiods", "norm_time", "physical_time"):
        if meta.get(k):
            meta[k] = float(meta[k]) * frac
    meta["timing"] = {"ps": float(seg_elapsed)}
    meta["segment_index"] = int(seg_index)          # provenance
    with h5py.File(seg_build_path, "a") as f:
        s["ps"]["steps"] = int(seg_steps)
        s["ps"]["max_ps"] = int(f["ps"].attrs.get("max_ps", 0))
        f.attrs["summary_json"] = json.dumps(s)


def _run_ps_segments(base_args, hash_str, run_storage, steps_ps, seg_steps,
                     initial_pos_vel, dp, wr, run_fn, summary_stub=None,
                     offload=False):
    """Run PS as a sequence of cleanly-closed checkpoint segments.

    Each segment integrates ``seg_steps`` PS steps (the last one shorter),
    writing ``<hash>_segNNN.h5`` via a ``.building`` temp + atomic rename, then
    ``<hash>_full.h5`` is (re)built as a VDS over all committed segments.

    On (re)launch this resumes from the last committed segment's exact
    ``end_state``, so a crash costs at most one segment. Because the VDS is only
    written once every segment is done, a crashed run leaves no ``_full.h5`` and
    the driver's read-cache short-circuit won't mistake a partial run for a
    finished one — it re-enters here and resumes.

    Returns ``(max_ps, elapsed_seconds)`` like the single-run path.
    """
    import math
    n_segments = math.ceil(steps_ps / seg_steps)

    wr.clear_building_segments(hash_str, run_storage)

    # Resume point.  Default: the contiguous prefix from seg000 — a deleted
    # segment (trailing OR middle) shortens it, so we refill from the gap and the
    # local VDS never stitches around a hole.  Offload mode: the HIGHEST local
    # segment, even if earlier ones were moved to other storage — resume only
    # needs its end_state, and the consistency check below rejects an
    # inconsistent/halted one. In offload mode the local VDS is NOT stitched
    # (the full set no longer lives here); rebuild it at the destination with
    # scripts/build_vds.py once the segments are reunited.
    if offload:
        latest = wr.latest_committed_segment(hash_str, run_storage)
        resume_from = latest
    else:
        committed = wr.contiguous_committed_segments(hash_str, run_storage)
        resume_from = committed[-1] if committed else None

    cur_state = initial_pos_vel
    start_seg = 0
    max_ps_seen = 0
    if resume_from is not None:
        last_idx, last_path = resume_from
        with h5py.File(last_path, "r") as f:
            cur_state = f["ps/end_state"][()]
            resumed_index = int(f["ps"].attrs["end_global_index"])
            max_ps_seen = int(f["ps"].attrs.get("max_ps", 0))
        start_seg = last_idx + 1
        expected = min(start_seg * seg_steps, steps_ps)
        if resumed_index != expected:
            raise RuntimeError(
                f"resume mismatch: committed segment {last_idx} ends at global "
                f"step {resumed_index:,}, expected {expected:,} — refusing to "
                f"continue from an inconsistent checkpoint")
        _lbl = "Offload resume" if offload else "Resuming"
        _extra = " (earlier segments assumed offloaded)" if offload else ""
        print(f"  {_lbl}: continuing from segment {last_idx}{_extra} "
              f"at global step {resumed_index:,}/{steps_ps:,}\n")

    t_start = time.time()
    for seg in range(start_seg, n_segments):
        seg_global_start = seg * seg_steps
        this_seg_steps = min(seg_steps, steps_ps - seg_global_start)
        seg_final = wr.seg_path_for(hash_str, seg, run_storage)
        seg_build = seg_final + ".building"

        args = dict(base_args)
        args.update(
            cache_path=seg_build,
            initial_pos_vel_ps=cur_state,
            steps_ps=this_seg_steps,
            global_index_start=seg_global_start,
            total_steps=steps_ps,
            segment_index=seg,
        )
        print(f"  --- PS segment {seg + 1}/{n_segments} "
              f"(global steps {seg_global_start:,}–{seg_global_start + this_seg_steps:,}) ---")
        _t_seg = time.time()
        run_fn(**args)
        # Stamp per-segment identity summary on the temp BEFORE committing, so
        # the atomic rename publishes data + summary together (and committed
        # files are never reopened for write).
        if summary_stub is not None:
            _stamp_segment_summary(seg_build, summary_stub, hash_str, seg,
                                   this_seg_steps, steps_ps,
                                   time.time() - _t_seg)
        os.replace(seg_build, seg_final)   # atomic commit of the segment

        with h5py.File(seg_final, "r") as f:
            cur_state = f["ps/end_state"][()]
            _end_gi = int(f["ps"].attrs["end_global_index"])
            max_ps_seen = max(max_ps_seen, int(f["ps"].attrs.get("max_ps", 0)))

        # Adaptive integration can halt early (PS series can't converge at
        # dt_min). If a segment stops short of its target, don't fabricate the
        # remaining segments — stitch what we have and stop.
        if _end_gi < seg_global_start + this_seg_steps:
            print(f"  Segment {seg} halted at global step {_end_gi:,} "
                  f"(target {seg_global_start + this_seg_steps:,}) — stopping run.\n")
            break

    elapsed = time.time() - t_start

    if offload:
        # The full set no longer lives locally — don't build a holey VDS.
        print("  Offload mode: skipping local VDS stitch. Once all segments are\n"
              "  gathered on the destination drive, build the stitched _full.h5 with:\n"
              f"    python scripts/build_vds.py <destination>/_rawdata\n")
        return max_ps_seen, elapsed

    # Stitch only once all segments exist, so a partial run has no _full.h5.
    full = wr.build_vds(hash_str, run_storage)
    max_ps = max_ps_seen
    if full is not None:
        with h5py.File(full, "r") as f:
            max_ps = int(f["ps"].attrs.get("max_ps", max_ps_seen))
    return max_ps, elapsed


class _SkipSection(Exception):
    """Raised to skip an optional analysis section on purpose.

    Caught by its own `except` ahead of the generic handler so an intentional
    skip prints one line instead of being logged as an analysis FAILURE.
    """


def main(cfg_path, replot=False):
    """Run a dipole-B simulation from a YAML config file path.

    Parameters
    ----------
    cfg_path : str – path to a YAML config file.
    replot   : bool – if True, force READ_DATA=True (skip solvers,
               regenerate plots from cached h5 data).
    """

    cfg        = load_config(cfg_path)

    # --- Resolve manual_h5_path. For a relative path, try yml-relative first
    #     (so trim_h5's auto-companion yml can use just a basename — the h5 is
    #     a sibling of the yml). Fall back to the raw path (cwd-relative) for
    #     legacy yml configs that point at e.g. data/dipoleb/.../*.h5. ---
    _raw_manual = cfg.get("manual_h5_path")
    if _raw_manual and not os.path.isabs(_raw_manual):
        _yml_relative = os.path.join(
            os.path.dirname(os.path.abspath(cfg_path)), _raw_manual
        )
        if os.path.exists(_yml_relative):
            cfg["manual_h5_path"] = _yml_relative
        # else: leave as-is — interpreted relative to cwd at run time

    # --- Resolve float type BEFORE importing physics modules so @maybe_njit
    #     sees the correct type (float128 skips njit, float64 compiles).
    #     YAML governs the default. If a manual h5 is being loaded, its saved
    #     dtype takes precedence so physics modules import at the right
    #     precision and builtins.npfloat stays consistent throughout. ---
    manual_h5_path = cfg.get("manual_h5_path")
    USE_FLOAT128 = cfg.get("use_float128", False)

    if manual_h5_path and os.path.exists(manual_h5_path):
        try:
            with h5py.File(manual_h5_path, "r") as _f:
                if "summary_json" in _f.attrs:
                    _saved = json.loads(_f.attrs["summary_json"])["meta"].get("dtype")
                    if _saved:
                        _file_use_float128 = (np.dtype(_saved).type == np.float128)
                        if _file_use_float128 != USE_FLOAT128:
                            print(f"  NOTE: manual h5 file uses {_saved}; "
                                  f"overriding YAML use_float128={USE_FLOAT128}.")
                        USE_FLOAT128 = _file_use_float128
        except (OSError, KeyError, json.JSONDecodeError):
            pass  # fall through to YAML setting

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
    # Data-only: write the h5 then stop before plotting/analysis. Ignored when
    # replotting (the whole point of a replot is to (re)make the figures).
    DATA_ONLY       = bool(params.get("data_only", False)) and not replot
    USE_RK45        = params["USE_RK45"]
    USE_RK4         = params["USE_RK4"]
    USE_RKG         = params["USE_RKG"]
    USE_PS          = params["USE_PS"]
    USE_ADAPTIVE    = params["USE_ADAPTIVE"]
    ps_decimate     = params["ps_decimate"]
    y_initial       = params["y_initial"]
    z_initial       = params["z_initial"]
    USE_PLOT_TITLES = params["USE_PLOT_TITLES"]
    USE_FULL_PLOT   = params["USE_FULL_PLOT"]
    # Figure-only fast path: emit just <hash>_invariants.png and skip every other
    # plot plus the Dragt / mu-deviation / bounce-drift streams (they only feed
    # summary diagnostics). Implies paper-plot mode for everything else.
    INVARIANTS_ONLY = params.get("INVARIANTS_ONLY", False)
    if INVARIANTS_ONLY:
        USE_FULL_PLOT = False
    slice_mode      = params["slice_mode"]
    gyro_window     = params["gyro_window"]
    output_folder   = params["output_folder"]
    run_storage     = params["run_storage"]
    window_time     = params["window_time"]
    n_gyro          = params["n_gyro"]

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
    ke_particle  = params.get("ke_particle",  None)
    mass_si      = params.get("mass_si",      None)
    T_gyro       = params.get("T_gyro",       None)
    gyroperiods  = params.get("gyroperiods",  None)
    norm_time    = params.get("norm_time",    None)

    # --- Step sizes (set by all modes except legacy/manual) ---
    ps_step              = params.get("ps_step",  None)
    rk4_step             = params.get("rk4_step", None)
    rkg_step             = params.get("rkg_step", None)
    n_steps_per_gyro_ps  = params.get("n_steps_per_gyro_ps",  None)
    n_steps_per_gyro_rk4 = params.get("n_steps_per_gyro_rk4", None)
    n_steps_per_gyro_rkg = params.get("n_steps_per_gyro_rkg", None)

    # --- Optional overrides (only some modes set these) ---
    # compute_derived always populates these keys with its own defaults,
    # so no module-level fallback is needed.
    ps_order        = params["ps_order"]
    ps_chunk_steps  = params["ps_chunk_steps"]
    rtol_rk45       = params["rtol_rk45"]
    atol_rk45       = params["atol_rk45"]
    user_min_phase  = params["user_min_phase"]
    max_plot_points_local = params.get("max_plot_points", 1_000_000)  # not in base.yml
    cache_velocity_rtol   = params["cache_velocity_rtol"]
    plot_boundary_pad     = params["plot_boundary_pad"]
    # manual_h5_path already set at top of main() (used to resolve npfloat); reassign
    # here from params for consistency — value is the same.
    manual_h5_path  = params["manual_h5_path"]

    # --- Adaptive PS settings ---
    ps_adaptive = params["ps_adaptive"]

    # --- Dragt monitor ---
    dragt_monitor_rtol = params["dragt_monitor_rtol"]

    # --- Bounce/drift detection ---
    bounce_drift_cfg        = params["bounce_drift"]
    velocity_epsilon_scale  = bounce_drift_cfg["velocity_epsilon_scale"]
    min_gap_steps           = bounce_drift_cfg["min_gap_steps"]
    gap_gyro_fraction       = bounce_drift_cfg["gap_gyro_fraction"]

    r_atmosphere            = params["r_atmosphere"]   # atmospheric impact threshold (R_E)

    # === Misc Odds and Ends ===
    PS_CHUNKING = True     # PS data always streamed to disk in chunks (no in-memory option)
    WRITE_DATA  = True     # always write h5 (required by chunked streaming)
    ul.plt_config(scale=1)                        # config file for setting plot sizes and fonts (from Dr. W)
    # In manual mode the cache_path is the user-supplied h5 (not in run_storage),
    # so no fresh data ever lands in _rawdata — skip creating it to avoid an
    # empty stub folder in the trimmed-replot output tree.
    if not (manual_h5_path and os.path.exists(manual_h5_path)):
        os.makedirs(run_storage, exist_ok=True)    # raw data storage
    os.makedirs(output_folder, exist_ok=True)  # ensures file for the storage for images and text file exists
    plt.ioff()                                 # turn off interactive mode for plots
    if USE_FLOAT128 and USE_RKG:
        print("  NOTE: RKG disabled because use_float128=True "
              "(numba JIT off + implicit Newton + np.linalg.solve makes it impractically slow).")
        USE_RKG = False


    # --- Safety defaults for variables assigned only inside conditional
    #     branches (cache-reload, solver-execution).  Ensures no
    #     UnboundLocalError regardless of which path is taken. ---
    solution_rk4  = None
    solution_rk45 = None
    solution_rkg  = None
    y_rk45_common = None
    summary       = {}
    timing        = {}
    stem          = ""
    max_ps_value  = None
    steps_rk4     = None
    steps_rkg     = None
    rk4_y_initial  = None
    rk45_y_initial = None
    rkg_y_initial  = None

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
                ke_particle = meta["energy_eV"]
                pitch_deg = meta["pitch_deg"]
                phi_deg = meta["phi_deg"]
                x_initial = meta["x0"]
                y_initial = meta["y0"]
                z_initial = meta["z0"]
                B_0 = meta["B0_T"]
                gyroperiods = meta["gyroperiods"]
                norm_time = meta["norm_time"]
                # npfloat already resolved at the top of main() from this file's saved dtype.

                T_gyro = meta.get("T_gyro", 2.0 * np.pi * (x_initial**3))  # fallback for older h5 files

                # ---- Combine yml preference with h5 availability ----
                # The h5 dictates what data EXISTS (you can't plot what
                # wasn't run). The yml's solvers block lets the user DROP
                # methods from the plots without re-running. So
                # USE_X = yml_wants AND h5_has, matching constb's behavior.
                _yml_wants = {"ps": USE_PS, "rk4": USE_RK4,
                              "rk45": USE_RK45, "rkg": USE_RKG}
                _missing = [m for m in _yml_wants
                            if _yml_wants[m] and not summary[m]["enabled"]]
                if _missing:
                    print(f"  Note: yml requests {_missing} but the h5 has no "
                          f"data for {'these' if len(_missing) > 1 else 'it'}. Skipping.")
                _dropped = [m for m in _yml_wants
                            if not _yml_wants[m] and summary[m]["enabled"]]
                if _dropped:
                    print(f"  Note: yml drops {_dropped} from plots "
                          f"(h5 has data, not plotted).")

                # ---- PS config ----
                ps_cfg = summary["ps"]
                USE_PS = USE_PS and ps_cfg["enabled"]
                PS_CHUNKING = ps_cfg["streaming"]
                ps_step = ps_cfg["dt"]
                steps_ps = ps_cfg["steps"]
                ps_decimate = ps_cfg["decimate"]
                ps_chunk_steps = ps_cfg["chunksize"]
                n_steps_per_gyro_ps = ps_cfg["numberstepspergyro"]
                max_ps_value = ps_cfg["max_ps"]
                e0_ps = ps_cfg["E0"]
                mu0_ps = ps_cfg["mu0"]

                # ---- RK4 config ----
                rk4_cfg = summary["rk4"]
                USE_RK4 = USE_RK4 and rk4_cfg["enabled"]
                rk4_step = rk4_cfg["dt"]
                steps_rk4 = rk4_cfg["steps"]
                n_steps_per_gyro_rk4 = rk4_cfg["numberstepspergyro"]


                # ---- RK45 config ----
                rk45_cfg = summary["rk45"]
                USE_RK45 = USE_RK45 and rk45_cfg["enabled"]
                rtol_rk45 = rk45_cfg["rtol"]
                atol_rk45 = rk45_cfg["atol"]

                # ---- RKG config ----
                rkg_cfg = summary["rkg"]
                USE_RKG = USE_RKG and rkg_cfg["enabled"]
                rkg_step = rkg_cfg["dt"]
                steps_rkg = rkg_cfg["steps"]
                n_steps_per_gyro_rkg = rkg_cfg["numberstepspergyro"]


                # ---- Load solver data ------
                """
                Earlier editions of the code loaded everything into memory, for extended runs this has become untenable.
                Chunking allows files to be written and read in chunks which takes far less memory. Note, right now ONLY PS method
                does the chunking method. I have not tried to apply it to RK method until we find specific needs beccause it was a lot of work.
                """

                # === Detect trimmed file (independent of which solvers are enabled) ===
                # summary_json preserves ORIGINAL run identity verbatim. Any solver
                # group carrying a `trim_end` attr means the data spans only part of
                # that original run, so meta-derived values (gyroperiods, norm_time,
                # physical_time) are stale. Recompute from actual data length using
                # the highest-priority group available (PS > RK4 > RKG).
                trim_attr_grp = None
                for _g in ("ps", "rk4", "rkg"):
                    if _g in cached and "trim_end" in cached[_g].attrs and "y" in cached[_g]:
                        trim_attr_grp = _g
                        break
                if trim_attr_grp is not None:
                    _gsrc = cached[trim_attr_grp]
                    if trim_attr_grp == "ps":
                        _stride = int(_gsrc.attrs.get("decimate", 1)) or 1
                        _n_eff = _gsrc["y"].shape[1] * _stride
                    elif trim_attr_grp == "rkg":
                        _n_eff = _gsrc["y"].shape[0]
                    else:
                        _n_eff = _gsrc["y"].shape[1]
                    _dt = float(_gsrc.attrs["dt"])
                    norm_time = _n_eff * _dt
                    gyroperiods = norm_time / T_gyro
                    trim_end_label = _gsrc.attrs.get("trim_end", "trimmed")
                    trim_window_label = _gsrc.attrs.get("trim_window_s", "")
                    print(f"  Trimmed file: {trim_end_label} {trim_window_label}s "
                          f"(source: {_gsrc.attrs.get('trim_source', '?')}) — "
                          f"effective gyroperiods={gyroperiods:.4g}")
                    if trim_window_label:
                        stem = f"{stem}_{trim_end_label}_{trim_window_label}s"
                    else:
                        stem = f"{stem}_{trim_end_label}"

                # === PS (chunked — data stays on disk, read in slices later) ===
                if USE_PS and "ps" in cached:
                    # Defense in depth: catches accidental truncation even when the
                    # file isn't a trim_h5 product (no trim_end attr).
                    n_store_actual = cached["ps"]["y"].shape[1]
                    ps_store_stride = ps_decimate if ps_decimate > 1 else 1
                    steps_ps_actual = n_store_actual * ps_store_stride
                    if steps_ps_actual < steps_ps:
                        steps_ps = steps_ps_actual

                # === RK4 ===
                if USE_RK4 and "rk4" in cached:
                    n_actual_rk4 = cached["rk4"]["y"].shape[1]
                    if n_actual_rk4 < steps_rk4:
                        print(f"  Trimmed file detected (rk4): {n_actual_rk4:,} columns "
                              f"(original {steps_rk4:,} steps)")
                        steps_rk4 = n_actual_rk4
                    solution_rk4 = cached["rk4"]["y"][()]
                    if "y_initial" in cached["rk4"]:
                        rk4_y_initial = cached["rk4"]["y_initial"][()]

                # === RK45 ===
                if USE_RK45 and "rk45" in cached:
                    solution_rk45 = SimpleNamespace(t=cached["rk45"]["t"][()], y=cached["rk45"]["y"][()])
                    if "y_initial" in cached["rk45"]:
                        rk45_y_initial = cached["rk45"]["y_initial"][()]

                # === RKG ===
                if USE_RKG and "rkg" in cached:
                    # rkg stores y as (n_steps, n_dim) — axis 0 is time
                    n_actual_rkg = cached["rkg"]["y"].shape[0]
                    if n_actual_rkg < steps_rkg:
                        print(f"  Trimmed file detected (rkg): {n_actual_rkg:,} rows "
                              f"(original {steps_rkg:,} steps)")
                        steps_rkg = n_actual_rkg
                    solution_rkg = cached["rkg"]["y"][()]
                    if "y_initial" in cached["rkg"]:
                        rkg_y_initial = cached["rkg"]["y_initial"][()]

    # for file/plot naming
    if mass_si == m_e: particle_type = "Electron"
    elif mass_si == m_p: particle_type = "Proton"
    else: particle_type = "Particle"

    # +1 for protons, -1 for electrons. Stored at npfloat precision so it
    # composes correctly with the rest of the physics state; the Dragt
    # analysis casts to float() at its call sites where it needs float64.
    charge_sign = npfloat(-1) if mass_si == m_e else npfloat(1)

    # === Misc Conversions  ===
    ke_joules = ke_particle * evtoj                     # converting KE from eV to Joules
    gamma = 1.0 + ke_joules / (mass_si * spdlight**2)   # Lorentz factor
    mass = gamma * mass_si                              # Relativistic mass used for magnetic moment calculations

    v_si = spdlight * np.sqrt(1.0 - 1.0 / gamma**2)     # m/s
    tau_time = gamma * mass_si / (abs(q_e) * abs(B_0))  # this is tau0 from paper
    v_tau = v_si * tau_time / RE                        # dimensionless velocity

    physical_time = norm_time * abs(tau_time)           # actual physical time, t; normalized time =t/tau_time
    window_duration = window_time/tau_time              # converting window_time to dimensionless time
    # Old (paper version):
    # tol_local = npfloat(tol) * tau_time   # Scale tolerance by tau_0
    tol_local = npfloat(tol)                            # plain machine eps; advisor's relative test in ps_integrate handles scaling

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

    # |sin| because gyroradius is a magnitude — pitch sign only sets the
    # rotation direction (via v_perp's sign in the IC), not the radius.
    gyro_radius_si = (gamma * mass_si * v_si * np.abs(np.sin(pitch_rad)) / (np.abs(q_e) * (B_0 / x_initial**3)))
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
    e0_ps = npfloat(0.5) * (vx0*vx0 + vy0*vy0 + vz0*vz0)
    y0_ps = np.zeros((17, 1), dtype=npfloat)
    y0_ps[0:6, 0] = initial_pos_vel
    x0, y0, z0 = initial_pos_vel[0:3]
    r2 = x0*x0 + y0*y0 + z0*z0
    r5inv = r2**(-2.5)
    y0_ps[14, 0] = -3 * x0 * z0 * r5inv
    y0_ps[15, 0] = -3 * y0 * z0 * r5inv
    y0_ps[16, 0] = -(3*z0*z0 - r2) * r5inv
    mu0_ps = mp.compute_mu_ps(y0_ps)[0]


    # === Build parameter tracer & check cache ===
    """
    This first part is scanning the files already stored in 'run_storage' based on input parameters (not specifically
    lodaded legacy files) in the yml to see if we already have the data. If it finds the data, it will
    load relevant parameters. If it does not find a file, it will start running the solvers to get the needed data.
    Beware that these files can be GB size for dipole.
    """
    if not USE_MANUAL_FILE:
        cache_path = wr.h5_path_for(physics_hash(cfg), run_storage)
        # A stitched (VDS) cache whose segment files were deleted would silently
        # read back zeros. Detect that and ignore the cache so the run resumes
        # and rebuilds instead of loading corrupt data.
        _stale_vds = os.path.exists(cache_path) and wr.vds_has_missing_sources(cache_path)
        if _stale_vds:
            print(f"  Stale VDS: {os.path.basename(cache_path)} references missing "
                  f"segment files — ignoring cache and resuming the run.\n")
        if os.path.exists(cache_path) and READ_DATA and not _stale_vds:
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
                ke_particle = meta["energy_eV"]
                pitch_deg = meta["pitch_deg"]
                phi_deg = meta["phi_deg"]
                x_initial = meta["x0"]
                y_initial = meta["y0"]
                z_initial = meta["z0"]
                B_0 = meta["B0_T"]
                gyroperiods = meta["gyroperiods"]
                norm_time = meta["norm_time"]
                # npfloat already resolved at the top of main() from this file's saved dtype.

                # ---- Combine yml preference with h5 availability ----
                # The h5 dictates what data EXISTS (you can't plot what
                # wasn't run). The yml's solvers block lets the user DROP
                # methods from the plots without re-running. So
                # USE_X = yml_wants AND h5_has, matching constb's behavior.
                _yml_wants = {"ps": USE_PS, "rk4": USE_RK4,
                              "rk45": USE_RK45, "rkg": USE_RKG}
                _missing = [m for m in _yml_wants
                            if _yml_wants[m] and not summary[m]["enabled"]]
                if _missing:
                    print(f"  Note: yml requests {_missing} but the h5 has no "
                          f"data for {'these' if len(_missing) > 1 else 'it'}. Skipping.")
                _dropped = [m for m in _yml_wants
                            if not _yml_wants[m] and summary[m]["enabled"]]
                if _dropped:
                    print(f"  Note: yml drops {_dropped} from plots "
                          f"(h5 has data, not plotted).")

                # ---- PS config ----
                ps_cfg = summary["ps"]
                USE_PS = USE_PS and ps_cfg["enabled"]
                PS_CHUNKING = ps_cfg["streaming"]
                ps_step = ps_cfg["dt"]
                steps_ps = ps_cfg["steps"]
                ps_decimate = ps_cfg["decimate"]
                ps_chunk_steps = ps_cfg["chunksize"]
                n_steps_per_gyro_ps = ps_cfg["numberstepspergyro"]
                max_ps_value = ps_cfg["max_ps"]
                e0_ps = ps_cfg["E0"]
                mu0_ps = ps_cfg["mu0"]

                # ---- RK4 config ----
                rk4_cfg = summary["rk4"]
                USE_RK4 = USE_RK4 and rk4_cfg["enabled"]
                rk4_step = rk4_cfg["dt"]
                steps_rk4 = rk4_cfg["steps"]
                n_steps_per_gyro_rk4 = rk4_cfg["numberstepspergyro"]


                # ---- RK45 config ----
                rk45_cfg = summary["rk45"]
                USE_RK45 = USE_RK45 and rk45_cfg["enabled"]
                rtol_rk45 = rk45_cfg["rtol"]
                atol_rk45 = rk45_cfg["atol"]

                # ---- RKG config ----
                rkg_cfg = summary["rkg"]
                USE_RKG = USE_RKG and rkg_cfg["enabled"]
                rkg_step = rkg_cfg["dt"]
                steps_rkg = rkg_cfg["steps"]
                n_steps_per_gyro_rkg = rkg_cfg["numberstepspergyro"]


                # ---- Load solver data ------
                # === Detect trimmed file (independent of which solvers are enabled) ===
                # Same logic as manual-mode block above. Recompute norm_time /
                # gyroperiods / physical_time from actual data length so plots,
                # axes, and labels reflect the trim regardless of which solver
                # the user is replotting.
                trim_attr_grp = None
                for _g in ("ps", "rk4", "rkg"):
                    if _g in cached and "trim_end" in cached[_g].attrs and "y" in cached[_g]:
                        trim_attr_grp = _g
                        break
                if trim_attr_grp is not None:
                    _gsrc = cached[trim_attr_grp]
                    if trim_attr_grp == "ps":
                        _stride = int(_gsrc.attrs.get("decimate", 1)) or 1
                        _n_eff = _gsrc["y"].shape[1] * _stride
                    elif trim_attr_grp == "rkg":
                        _n_eff = _gsrc["y"].shape[0]
                    else:
                        _n_eff = _gsrc["y"].shape[1]
                    _dt = float(_gsrc.attrs["dt"])
                    norm_time = _n_eff * _dt
                    gyroperiods = norm_time / T_gyro
                    physical_time = norm_time * abs(tau_time)
                    trim_end_label = _gsrc.attrs.get("trim_end", "trimmed")
                    trim_window_label = _gsrc.attrs.get("trim_window_s", "")
                    print(f"  Trimmed file: {trim_end_label} {trim_window_label}s "
                          f"(source: {_gsrc.attrs.get('trim_source', '?')}) — "
                          f"effective gyroperiods={gyroperiods:.4g}")
                    if trim_window_label:
                        stem = f"{stem}_{trim_end_label}_{trim_window_label}s"
                    else:
                        stem = f"{stem}_{trim_end_label}"

                # === PS ===
                if USE_PS and "ps" in cached:
                    # Defense in depth for accidental truncation (no trim_end attr).
                    n_store_actual = cached["ps"]["y"].shape[1]
                    ps_store_stride = ps_decimate if ps_decimate > 1 else 1
                    steps_ps_actual = n_store_actual * ps_store_stride
                    if steps_ps_actual < steps_ps:
                        steps_ps = steps_ps_actual

                # === RK4 ===
                if USE_RK4 and "rk4" in cached:
                    n_actual_rk4 = cached["rk4"]["y"].shape[1]
                    if n_actual_rk4 < steps_rk4:
                        print(f"  Trimmed file detected (rk4): {n_actual_rk4:,} columns "
                              f"(original {steps_rk4:,} steps)")
                        steps_rk4 = n_actual_rk4
                    solution_rk4 = cached["rk4"]["y"][()]
                    if "y_initial" in cached["rk4"]:
                        rk4_y_initial = cached["rk4"]["y_initial"][()]

                # === RK45 ===
                if USE_RK45 and "rk45" in cached:
                    solution_rk45 = SimpleNamespace(t=cached["rk45"]["t"][()], y=cached["rk45"]["y"][()])
                    if "y_initial" in cached["rk45"]:
                        rk45_y_initial = cached["rk45"]["y_initial"][()]

                # === RKG ===
                if USE_RKG and "rkg" in cached:
                    # rkg stores y as (n_steps, n_dim) — axis 0 is time
                    n_actual_rkg = cached["rkg"]["y"].shape[0]
                    if n_actual_rkg < steps_rkg:
                        print(f"  Trimmed file detected (rkg): {n_actual_rkg:,} rows "
                              f"(original {steps_rkg:,} steps)")
                        steps_rkg = n_actual_rkg
                    solution_rkg = cached["rkg"]["y"][()]
                    if "y_initial" in cached["rkg"]:
                        rkg_y_initial = cached["rkg"]["y_initial"][()]
        else:
            print("No matching file or 'Read Data' skipped. Running solvers...\n")

            # PS-grid step count, also used as RK45's t_eval grid
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
                    ps_order=ps_order,
                    tol=tol_local,
                    charge_sign=charge_sign,
                    e0_ps=e0_ps,
                    mu0_ps=mu0_ps,
                    cache_path=cache_path,
                    write_data=True,
                    chunk_steps=ps_chunk_steps,
                    decimate=ps_decimate,
                    n_steps_per_gyro_ps=n_steps_per_gyro_ps,
                    user_min_phase=user_min_phase,
                    dragt_monitor=dragt_mon,
                    r_atmosphere=r_atmosphere,
                )
                # Pick the streaming function; both the adaptive and fixed-step
                # paths support segmented checkpointing identically.
                if USE_ADAPTIVE:
                    _stream_args.update(
                        order_low=ps_adaptive["order_low"],
                        order_high=ps_adaptive["order_high"],
                        grow_factor=ps_adaptive["grow_factor"],
                        shrink_factor=ps_adaptive["shrink_factor"],
                        steps_per_local_gyro=ps_adaptive["steps_per_local_gyro"],
                        min_fast_path_N=ps_adaptive["min_fast_path_N"],
                    )
                    _run_fn = run_ps_streaming_adaptive
                else:
                    _run_fn = dp.run_ps_streaming_with_decimation

                # Segmented checkpointing when ps_segment_gyroperiods is set (and
                # smaller than the whole run); otherwise the original single-file
                # streaming path, byte-for-byte unchanged.
                _seg_steps = params.get("ps_segment_steps", 0) or 0
                if _seg_steps and _seg_steps < steps_ps:
                    # Identity stub stamped (per-segment-patched) onto each
                    # segment so it can be manual-loaded standalone. Mirrors the
                    # run-end summary dict below (keep the two in sync); values
                    # coerced to plain Python types (stamp json.dumps has no
                    # numpy encoder). RK blocks are disabled — a lone segment
                    # carries PS data only, whatever else the run computed.
                    _summary_stub = {
                        "meta": {
                            "stem": wr.stem_from_h5(cache_path),
                            "particle": particle_type,
                            "mass_si": float(mass_si),
                            "q_e": float(q_e),
                            "energy_eV": float(ke_particle),
                            "pitch_deg": float(pitch_deg),
                            "phi_deg": float(phi_deg),
                            "x0": float(x_initial),
                            "y0": float(y_initial),
                            "z0": float(z_initial),
                            "B0_T": float(B_0),
                            "gyroperiods": float(gyroperiods),
                            "norm_time": float(norm_time),
                            "physical_time": float(physical_time),
                            "percent_c": float(v_si / spdlight),
                            "charge_sign": float(charge_sign),
                            "dtype": npfloat.__name__,
                            "tau0": float(tau_time),
                            "T_gyro": float(T_gyro),
                            "timing": {},
                        },
                        "ps": {
                            "enabled": True,
                            "dt": float(ps_step),
                            "steps": int(steps_ps),
                            "streaming": True,
                            "ordercap": int(ps_order),
                            "max_ps": None,
                            "chunksize": int(ps_chunk_steps),
                            "decimate": int(ps_decimate),
                            "numberstepspergyro": int(n_steps_per_gyro_ps),
                            "E0": float(e0_ps),
                            "mu0": float(mu0_ps),
                            "minphase": float(user_min_phase),
                            "tol": float(tol_local),
                        },
                        "rk4":  {"enabled": False, "dt": None, "steps": None,
                                 "numberstepspergyro": None},
                        "rk45": {"enabled": False, "rtol": None, "atol": None},
                        "rkg":  {"enabled": False, "dt": None, "steps": None,
                                 "numberstepspergyro": None},
                    }
                    max_ps, elapsed_ps = _run_ps_segments(
                        _stream_args, physics_hash(cfg), run_storage,
                        steps_ps, _seg_steps, initial_pos_vel, dp, wr, _run_fn,
                        summary_stub=_summary_stub,
                        offload=bool(params.get("ps_segment_offload", False)))
                else:
                    max_ps, elapsed_ps = _run_fn(**_stream_args)
                dragt_mon.summary()
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
                    args=(charge_sign,),
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
                    args=(charge_sign,),)
                end_time_rk4 = time.time()

            # ====== Run RKG ======
            if USE_RKG:
                # === Symplectic Implementations =====
                r0 = np.array([x_initial, y_initial, z_initial], dtype=npfloat)   # already normalized RE units
                v_tau_vec = np.array([vx_initial, vy_initial, vz_initial], dtype=npfloat)

                A0 = dp.vector_potential(r0)
                p0 = v_tau_vec + A0
                y0 = np.concatenate((r0, p0))   # for Hamiltonian in RKG

                steps_rkg = int(norm_time / rkg_step)
                steps_rkg = max(1, steps_rkg)

                start_time_rkg = time.time()
                solution_rkg, rkg_n_failed, rkg_avg_iters, rkg_max_iters = dp.rkgl4_hamiltonian(
                    dp.hamiltonian_rhs,
                    y0,
                    rkg_step,
                    steps_rkg,
                    args=(charge_sign,),
                )
                end_time_rkg = time.time()
                # Fixed-point (functional iteration) diagnostics for the implicit
                # RKG stage solve — matches Yugo & Iyemori (2007) / Calvo (2003).
                # Stopping tol 1e-15 on the stage-update, cap 100 sweeps — see
                # dp.rkgl4_hamiltonian_step_fp. (Tol must stay well below the
                # per-step truncation error or the residual floor drifts the energy.)
                print(f"  RKG fixed-point: avg {rkg_avg_iters:.2f} sweeps/step, "
                      f"max {rkg_max_iters} (cap 100), tol 1e-15")
                if rkg_n_failed > 0:
                    pct = 100.0 * rkg_n_failed / steps_rkg
                    print(f"  WARNING: {rkg_n_failed:,} of {steps_rkg:,} RKG steps "
                          f"({pct:.2f}%) hit the sweep cap without converging. "
                          f"Consider reducing rkg_step or increasing max_iter.")

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
                    "energy_eV": npfloat(ke_particle),
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
                    "ordercap": ps_order,
                    "max_ps": max_ps_value,
                    "numberstepspergyro": n_steps_per_gyro_ps,
                    "dt": ps_step,
                    "steps": steps_ps,
                    "streaming": True,
                    "chunksize": ps_chunk_steps,
                    "decimate": ps_decimate,
                    "tol": tol_local,
                    "minphase" : user_min_phase,
                    "E0": float(e0_ps),
                    "mu0": float(mu0_ps),
                    "t0": 0.0,
                })
                results["meta"]["timing"]["ps"] = end_time_ps - start_time_ps

            results["rk4"] = { "enabled": bool(USE_RK4),}
            if USE_RK4:
                results["rk4"].update({
                    "y": solution_rk4,
                    "numberstepspergyro": n_steps_per_gyro_rk4,
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
                    "numberstepspergyro": n_steps_per_gyro_rkg,
                    "dt": npfloat(rkg_step),
                    "steps": int(steps_rkg),
                    "t0": 0.0
                })
                results["meta"]["timing"]["rkg"] = end_time_rkg - start_time_rkg

            # =========================
            # ====== Save Results =====
            # =========================
            stem = wr.stem_from_h5(cache_path)
            timing = results["meta"]["timing"]
            results["meta"]["stem"]=stem
            if WRITE_DATA:
                summary = {
                    "meta": {
                        "stem": stem,
                        "particle": particle_type,
                        "mass_si": mass_si,
                        "q_e": q_e,
                        "energy_eV": float(ke_particle),
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
                        "charge_sign": float(charge_sign),
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
                        "ordercap": ps_order if USE_PS else None,
                        "max_ps": max_ps_value,
                        "chunksize": ps_chunk_steps if USE_PS else None,
                        "decimate": ps_decimate if USE_PS else None,
                        "numberstepspergyro": n_steps_per_gyro_ps if USE_PS else None,
                        "E0": float(e0_ps) if USE_PS else None,
                        "mu0": float(mu0_ps) if USE_PS else None,
                        "minphase": user_min_phase if USE_PS else None,
                        "tol": float(tol_local)
                    },
                    "rk4": {
                        "enabled": USE_RK4,
                        "dt": float(rk4_step) if USE_RK4 else None,
                        "steps": int(steps_rk4) if USE_RK4 else None,
                        "numberstepspergyro": n_steps_per_gyro_rk4 if USE_RK4 else None,
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
                        "numberstepspergyro": n_steps_per_gyro_rkg if USE_RKG else None,
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
        _, peak = tracemalloc.get_traced_memory()
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
        "NGYRO" : n_gyro,
        "gyroslice": gyro_window,
        "maxplotpoints": max_plot_points_local,
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
    if DEBUG: tracemalloc.start()

    ps_order_label = None # for plotting later
    ps_order_mean  = None

    # Preflight the VDS before anything reads it. A stitched <hash>_full.h5
    # references its segments by BASENAME, so moving the segments (or the
    # _full.h5) to another drive silently breaks the mapping — HDF5 returns the
    # 0.0 fill value rather than raising, and every figure below would come out
    # blank/NaN with no error. Fail here, with the fix, instead.
    if USE_PS and not DATA_ONLY:
        wr.check_vds_readable(cache_path)

    if USE_PS:
        with h5py.File(cache_path, "r") as ps_h5:
            ps_grp = ps_h5["ps"]
            # Label uses mean (typical work/step) — falls back to max for
            # h5 files written before mean tracking landed.
            ps_order_label = ul.ps_order_label_from_attrs(ps_grp.attrs)
            if "mean_ps" in ps_grp.attrs:
                ps_order_mean = float(ps_grp.attrs["mean_ps"])
            ps_order_max = int(ps_grp.attrs["max_ps"]) if "max_ps" in ps_grp.attrs else None

            # Skip the strided trajectory read under data_only — it's only for
            # plots we won't make, and reading through a (possibly moved) VDS is
            # exactly what data_only exists to avoid. Attrs above are cheap/safe.
            if USE_FULL_PLOT and not DATA_ONLY:
                ps_y_h5 = ps_grp["y"]
                # One sequential pass for all three rows, decimated on the way
                # in. Was three separate `ps_y_h5[i, ::stride]` reads, each of
                # which decompresses the whole VDS (3x the I/O on a 28 GB run),
                # with the stride derived from steps_ps (physical steps) rather
                # than the STORED column count — so with ps_decimate > 1 the
                # full-trajectory plots were silently thinned by a further
                # factor of ps_decimate below max_plot_points.
                x_ps_plot, y_ps_plot, z_ps_plot = ul.read_rows_decimated(
                    ps_y_h5, (0, 1, 2), 0, ps_y_h5.shape[1] - 1,
                    max_plot_points_local)

    if DEBUG:
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        logger.debug(f"Data access for plottings: {peak / 1024**2:.2f} MB\n")


    print(f"\n{'='*60}")
    print(f"  Run Statistics")
    print(f"{'='*60}")
    # === Timing Summary ===
    print(f"Particle        : {ke_particle:.1e} eV {particle_type}")
    if USE_RK45 and "rk45" in timing:
        print(f"Run Time RK45   : {timing['rk45']:.2f} s")
    if USE_RK4 and "rk4" in timing:
        print(f"Run Time RK4    : {timing['rk4']:.2f} s")
    if USE_RKG and "rkg" in timing:
        print(f"Run Time RKG    : {timing['rkg']:.2f} s")
    if USE_PS and "ps" in timing:
        print(f"Run Time PS     : {timing['ps']:.2f} s")

    print(f"Norm Time       : {norm_time:.2e}")
    print(f"Physical Time   : {physical_time:.2e} s")
    if ps_order_mean is not None:
        print(f"PS Orders       : max={ps_order_label}, mean={ps_order_mean:.1f}")
    else:
        print(f"PS Orders       : max={ps_order_label}")
    print(f"% of c          : {100*v_si/spdlight:.8f}")

    if DEBUG:
        logger.debug(f"Norm Time: {norm_time:.2e}")
        logger.debug(f"Physical Time   : {physical_time:.2e} s")
        logger.debug(f"ps_step: {ps_step}, norm_time: {norm_time}, steps_ps: {steps_ps}")
    print(f"{'='*60}")


    # === Create run-specific output subfolders ===
    # data/dipoleb/<config>/<run-hash>/figures/   ← plots
    # data/dipoleb/<config>/<run-hash>/           ← summary, config copy, log
    # data/dipoleb/<config>/_rawdata/             ← h5 trajectory files
    run_folder = os.path.join(output_folder, stem)
    fig_folder = os.path.join(run_folder, "figures")
    os.makedirs(fig_folder, exist_ok=True)

    # --- Redirect debug log to run folder ---
    if DEBUG:
        _log_path = os.path.join(run_folder, f"{stem}.log")
        dbg.redirect_logger(logger, _log_path)
        print(f"Debug log redirected to {_log_path}\n")

    # --- Copy config YAML to run folder (with git hash) ---
    copy_config_to_output(cfg_path, run_folder, cfg=cfg)

    # --- Data-only: h5/VDS + provenance config are written; stop before any
    # plotting or trajectory read-back. Replot later (this copied yml already
    # has manual_h5_path guidance) once the segments are reassembled. ---
    if DATA_ONLY:
        print(f"\n{'='*60}\ndata_only: trajectory written, skipping all plots/analysis.\n"
              f"  Replot later with: python run.py {os.path.join(run_folder, os.path.basename(cfg_path))}\n"
              f"{'='*60}\n")
        return

    # =====================================================
    # ============== Full Trajectory Plots ================
    # =====================================================
    plotbounds = x_initial + plot_boundary_pad

    if USE_FULL_PLOT:
        _traj_common = dict(
            run_folder=fig_folder, stem=stem,
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

    # The slice read only feeds slice_2d/slice_3d; skip it when neither is drawn.
    _sw = (dict.fromkeys(
        ("ps_x_slice","ps_y_slice","ps_z_slice","rk4_x_slice","rk4_y_slice","rk4_z_slice",
         "rkg_x_slice","rkg_y_slice","rkg_z_slice","rk45_x_slice","rk45_y_slice",
         "rk45_z_slice","ps_order_label"))
        if INVARIANTS_ONLY else ul.prepare_slice_dipoleb(
        slice_mode, window_duration, norm_time,
        USE_PS=USE_PS, cache_path=cache_path, ps_step=ps_step,
        steps_ps=steps_ps, ps_decimate=ps_decimate,
        max_plot_points=max_plot_points_local,
        USE_RK4=USE_RK4, solution_rk4=solution_rk4, rk4_step=rk4_step,
        USE_RKG=USE_RKG, solution_rkg=solution_rkg, rkg_step=rkg_step,
        USE_RK45=USE_RK45, y_rk45_common=y_rk45_common,
    ))
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
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        logger.info(f"Peak memory usage for slice analysis: {peak / 1024**2:.2f} MB")



    # =====================================================
    # ================ Trajectory Slice Plots =============
    # =====================================================
    _slice_common = dict(
        run_folder=fig_folder, stem=stem,
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

    if not INVARIANTS_ONLY:
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

    # ONE streaming pass over the PS VDS feeding the KE, P_phi and mu plots.
    # These three used to walk the whole file independently (~3x the I/O; on the
    # 2715-segment cec255 run that is ~18 h vs ~8 h). Decimation semantics inside
    # are unchanged, so every downstream number is identical.
    _ps_fused = None
    if USE_PS:
        _ps_fused = ea.compute_ps_invariants_fused(
            cache_path=cache_path, e0_ps=e0_ps, ps_y_initial=initial_pos_vel,
            ps_step=ps_step, ps_decimate=ps_decimate,
            max_plot_points=max_plot_points_local,
            energy_stride=max(1, steps_ps // max_plot_points_local),
            charge_sign=charge_sign,
        )

    _ke = ea.compute_ke_errors(
        T_gyro, n_ps=steps_ps, max_plot_points=max_plot_points_local,
        ps_fused=_ps_fused,
        USE_PS=USE_PS, cache_path=cache_path, ps_step=ps_step,
        ps_decimate=ps_decimate, e0_ps=e0_ps,
        USE_RK4=USE_RK4, solution_rk4=solution_rk4, rk4_step=rk4_step,
        rk4_y_initial=rk4_y_initial,
        USE_RKG=USE_RKG, solution_rkg=solution_rkg, rkg_step=rkg_step,
        rkg_y_initial=rkg_y_initial,
        USE_RK45=USE_RK45, y_rk45_common=y_rk45_common,
        rk45_y_initial=rk45_y_initial,
        USE_EXTERNAL_H5_ps=USE_EXTERNAL_H5_ps,   external_h5_ps=external_h5_ps,
        USE_EXTERNAL_H5_rk4=USE_EXTERNAL_H5_rk4, external_h5_rk4=external_h5_rk4,
        USE_EXTERNAL_H5_rk45=USE_EXTERNAL_H5_rk45, external_h5_rk45=external_h5_rk45,
        USE_EXTERNAL_H5_rkg=USE_EXTERNAL_H5_rkg,   external_h5_rkg=external_h5_rkg,
        vector_potential_func=dp.vector_potential,
        charge_sign=charge_sign,
        load_results_h5_func=wr.load_results_h5_dipoleb,
    )

    time_factor    = _ke["time_factor"]
    energy_stride  = _ke["energy_stride"]
    rel_drift_ps   = _ke["rel_drift_ps"]
    rel_drift_rk4  = _ke["rel_drift_rk4"]
    rel_drift_rk45 = _ke["rel_drift_rk45"]
    rel_drift_rkg  = _ke["rel_drift_rkg"]

    if not INVARIANTS_ONLY:
        dplt.ke_error(
            run_folder=fig_folder, stem=stem,
            particle_type=particle_type, ps_order_label=ps_order_label,
            USE_PLOT_TITLES=USE_PLOT_TITLES, time_factor=time_factor, norm_time=norm_time,
            ps_data=_ke["ke_ps"], rk4_data=_ke["ke_rk4"],
            rk45_data=_ke["ke_rk45"], rkg_data=_ke["ke_rkg"],
            ext_ps_data=_ke["ke_ext_ps"], ext_rk4_data=_ke["ke_ext_rk4"],
            ext_rk45_data=_ke["ke_ext_rk45"], ext_rkg_data=_ke["ke_ext_rkg"],
            envelope=True,   # True = plot max-per-bin upper envelope (cleaner RKG band)
        )

    if DEBUG:
        _, peak = tracemalloc.get_traced_memory()
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

    # =========================================================
    # PLOT RELATIVE ERROR OF CANONICAL ANGULAR MOMENTUM
    # =========================================================
    # Runs right after ke_error (and BEFORE the Dragt analysis, which streams the
    # whole VDS and is a likely crash point) so both error plots are safely on
    # disk before that risk.
    # Computes P_phi drift for every enabled solver (mirrors compute_ke_errors),
    # then plots them all on a single log-log axis using the same color /
    # linestyle scheme as the kinetic-energy plot.
    # Produced whenever there is anything to plot — a local solver OR an
    # external h5 overlay. Independent of USE_FULL_PLOT (paper plot), matching
    # ke_error, so external-only comparison runs still get a P_phi plot.
    if (USE_PS or USE_RK4 or USE_RK45 or USE_RKG
            or USE_EXTERNAL_H5_ps or USE_EXTERNAL_H5_rk4
            or USE_EXTERNAL_H5_rk45 or USE_EXTERNAL_H5_rkg):
        _pphi = ea.compute_pphi_errors(
            T_gyro, n_ps=steps_ps, max_plot_points=max_plot_points_local,
            ps_fused=_ps_fused,
            USE_PS=USE_PS, cache_path=cache_path, ps_step=ps_step,
            ps_decimate=ps_decimate, ps_y_initial=initial_pos_vel,
            USE_RK4=USE_RK4, solution_rk4=solution_rk4, rk4_step=rk4_step,
            rk4_y_initial=rk4_y_initial,
            USE_RKG=USE_RKG, solution_rkg=solution_rkg, rkg_step=rkg_step,
            rkg_y_initial=rkg_y_initial,
            USE_RK45=USE_RK45, y_rk45_common=y_rk45_common,
            rk45_y_initial=rk45_y_initial,
            USE_EXTERNAL_H5_ps=USE_EXTERNAL_H5_ps,   external_h5_ps=external_h5_ps,
            USE_EXTERNAL_H5_rk4=USE_EXTERNAL_H5_rk4, external_h5_rk4=external_h5_rk4,
            USE_EXTERNAL_H5_rk45=USE_EXTERNAL_H5_rk45, external_h5_rk45=external_h5_rk45,
            USE_EXTERNAL_H5_rkg=USE_EXTERNAL_H5_rkg,   external_h5_rkg=external_h5_rkg,
            vector_potential_func=dp.vector_potential,
            charge_sign=charge_sign,
            load_results_h5_func=wr.load_results_h5_dipoleb,
        )
        if not INVARIANTS_ONLY:
            dplt.pphi_error(
                run_folder=fig_folder, stem=stem,
                particle_type=particle_type, ps_order_label=ps_order_label,
                USE_PLOT_TITLES=USE_PLOT_TITLES,
                time_factor=time_factor, norm_time=norm_time,
                ylabel_str=_pphi["ylabel"],
                ps_data=_pphi["pphi_ps"], rk4_data=_pphi["pphi_rk4"],
                rk45_data=_pphi["pphi_rk45"], rkg_data=_pphi["pphi_rkg"],
                ext_ps_data=_pphi["pphi_ext_ps"], ext_rk4_data=_pphi["pphi_ext_rk4"],
                ext_rk45_data=_pphi["pphi_ext_rk45"], ext_rkg_data=_pphi["pphi_ext_rkg"],
            )

    # =========================================================
    # PLOT RELATIVE ERROR OF MAGNETIC MOMENT (full run)
    # =========================================================
    # Third invariant plot in the ke_error / pphi_error family: |Δμ_n|/μ_0
    # over the WHOLE run on log-log axes (the mu_deviation plot further down
    # covers only the gyro window). Same chunked-h5 reading pattern as the
    # other two so it stays memory-safe on large VDS files, and runs BEFORE
    # the Dragt analysis for the same crash-safety reason as pphi_error.
    # Whole-run ceiling on |dmu|/mu0, captured from the array that already
    # feeds the MUerror plot. None when PS did not run. Distinct from the
    # tail-window mu_max_err in the summary, which covers only the last
    # 0.01% of the run.
    mu_max_run = None
    if (USE_PS or USE_RK4 or USE_RK45 or USE_RKG
            or USE_EXTERNAL_H5_ps or USE_EXTERNAL_H5_rk4
            or USE_EXTERNAL_H5_rk45 or USE_EXTERNAL_H5_rkg):
        _mue = ea.compute_mu_errors(
            T_gyro, n_ps=steps_ps, max_plot_points=max_plot_points_local,
            ps_fused=_ps_fused,
            USE_PS=USE_PS, cache_path=cache_path, ps_step=ps_step,
            ps_decimate=ps_decimate, ps_y_initial=initial_pos_vel,
            USE_RK4=USE_RK4, solution_rk4=solution_rk4, rk4_step=rk4_step,
            rk4_y_initial=rk4_y_initial,
            USE_RKG=USE_RKG, solution_rkg=solution_rkg, rkg_step=rkg_step,
            rkg_y_initial=rkg_y_initial,
            USE_RK45=USE_RK45, y_rk45_common=y_rk45_common,
            rk45_y_initial=rk45_y_initial,
            USE_EXTERNAL_H5_ps=USE_EXTERNAL_H5_ps,   external_h5_ps=external_h5_ps,
            USE_EXTERNAL_H5_rk4=USE_EXTERNAL_H5_rk4, external_h5_rk4=external_h5_rk4,
            USE_EXTERNAL_H5_rk45=USE_EXTERNAL_H5_rk45, external_h5_rk45=external_h5_rk45,
            USE_EXTERNAL_H5_rkg=USE_EXTERNAL_H5_rkg,   external_h5_rkg=external_h5_rkg,
            vector_potential_func=dp.vector_potential,
            charge_sign=charge_sign,
            load_results_h5_func=wr.load_results_h5_dipoleb,
        )
        # Capture BEFORE any plotting, so a figure failure cannot cost the
        # number. rel_mu_ps is the uniform-decimation full-run array; the
        # log-spaced "mu_ps" tuple is plot-only and must not be used here.
        if _mue.get("rel_mu_ps") is not None:
            mu_max_run = float(np.max(_mue["rel_mu_ps"]))
        if not INVARIANTS_ONLY:
            dplt.mu_error(
                run_folder=fig_folder, stem=stem,
                particle_type=particle_type, ps_order_label=ps_order_label,
                USE_PLOT_TITLES=USE_PLOT_TITLES,
                time_factor=time_factor, norm_time=norm_time,
                ylabel_str=_mue["ylabel"],
                ps_data=_mue["mu_ps"], rk4_data=_mue["mu_rk4"],
                rk45_data=_mue["mu_rk45"], rkg_data=_mue["mu_rkg"],
                ext_ps_data=_mue["mu_ext_ps"], ext_rk4_data=_mue["mu_ext_rk4"],
                ext_rk45_data=_mue["mu_ext_rk45"], ext_rkg_data=_mue["mu_ext_rkg"],
            )

        # ---- Invariants overlay: E, P_phi, mu for the PS run on ONE axis ----
        # Reuses the three log-spaced PS series already computed above (no
        # extra h5 pass). PS-only by design: colours encode the quantity here.
        #
        # PAPER PLOT — deliberately NOT gated on USE_FULL_PLOT: this figure is
        # wanted in every mode. Do not "tidy" it into the full_plot gate; the
        # only condition is USE_PS, because all three series it draws are PS
        # series and there is nothing to plot without them.
        if USE_PS:
            dplt.invariants_overlay(
                run_folder=fig_folder, stem=stem,
                particle_type=particle_type, ps_order_label=ps_order_label,
                USE_PLOT_TITLES=USE_PLOT_TITLES, time_factor=time_factor,
                ke_data=_ke["ke_ps"],
                pphi_data=_pphi["pphi_ps"],
                mu_data=_mue["mu_ps"],
            )

    # ==================================================
    # ======== Dragt Analysis + Poincaré Plots =========
    # ==================================================
    from functools import partial as _partial
    # In paper-plot mode (full_plot=false) the Dragt *analysis* still runs — it
    # populates dragt_log (summary writer) plus adiabaticity/crossing stats — but
    # no Dragt figures are emitted. Feed no-op plot funcs to skip only the images.
    def _noop_plot(*_a, **_k):
        return None
    _dragt_plot = (lambda f: _partial(f, use_titles=USE_PLOT_TITLES)) if USE_FULL_PLOT \
        else (lambda f: _noop_plot)
    # Best-effort: the Dragt stream walks the whole VDS and can fail on a very
    # long run. Default dragt_log keeps the summary writer's key access safe, and
    # a failure here still lets the summary / CSV be written below.
    # Must carry EVERY key the summary / CSV writers read, or a skipped-or-failed
    # Dragt section takes the writers down with it — which is exactly what the
    # try/except below exists to prevent. Keep in sync with the identical default
    # in dipoleb_dragt_analysis.run_section.
    dragt_log = {
        "L_eff": None, "W0_sq": None, "boundary": None, "mu_sq": None,
        "orbit_character": None, "eps_initial": None, "eps_mean": None,
        "eps_max": None, "n_eq_crossings": None,
        "hit_atmosphere": False, "hit_atm_r": None,
    }
    try:
        if INVARIANTS_ONLY:
            raise _SkipSection("invariants_only: skipping Dragt analysis")
        dragt_log, _ = df.run_section(
            x_initial, y_initial, z_initial,
            vx_initial, vy_initial, v_tau,
            charge_sign, gamma,
            USE_PS=USE_PS, cache_path=cache_path,
            ps_step=ps_step, time_factor=time_factor,
            cache_velocity_rtol=cache_velocity_rtol,
            fig_folder=fig_folder, stem=stem,
            poincare_func=_dragt_plot(dplt.poincare),
            gyrophase_mu_func=_dragt_plot(dplt.gyrophase_mu),
            polar_phase_space_func=_dragt_plot(dplt.polar_phase_space),
            meridian_plane_func=_dragt_plot(dplt.meridian_plane),
            adiabaticity_func=_dragt_plot(dplt.adiabaticity),
            meridian_plane_RE_func=_dragt_plot(dplt.meridian_plane_RE),
        )
    except _SkipSection as _s:
        print(f"  {_s}")
    except Exception as _e:
        logger.exception("Dragt analysis failed; continuing with default dragt_log.")
        print(f"\n  WARNING: Dragt analysis failed ({type(_e).__name__}: {_e});\n"
              f"  summary will omit Dragt diagnostics.\n")


    # ============================================================
    # ================ Magnetic Moment Deviations ================
    # ============================================================
    if DEBUG: tracemalloc.start()

    mu_rk4_result = mu_rkg_result = mu_rk45_result = mu_ps_result = None

    # Best-effort: the PS μ pass streams the whole VDS. On failure keep whatever
    # per-solver results completed (all pre-set to None above) so the summary
    # still writes; ps_order_label retains its earlier value.
    try:
        if INVARIANTS_ONLY:
            raise _SkipSection("invariants_only: skipping mu-deviation analysis")
        if USE_RK4:
            mu_rk4_result = mp.compute_mu_deviation_rk(
                solution_rk4, steps_rk4, rk4_step,
                n_gyro, n_steps_per_gyro_rk4, gyro_window, time_factor,
                solver_type="rk4", y_initial=rk4_y_initial)

        if USE_RKG:
            mu_rkg_result = mp.compute_mu_deviation_rk(
                solution_rkg, steps_rkg, rkg_step,
                n_gyro, n_steps_per_gyro_rkg, gyro_window, time_factor,
                solver_type="rkg", y_initial=rkg_y_initial,
                charge_sign=charge_sign)

        if USE_RK45:
            mu_rk45_result = mp.compute_mu_deviation_rk(
                y_rk45_common, steps_ps, ps_step,
                n_gyro, n_steps_per_gyro_ps, gyro_window, time_factor,
                solver_type="rk45", y_initial=rk45_y_initial)

        if USE_PS:
            mu_ps_result = mp.compute_mu_deviation_ps(
                cache_path, steps_ps, ps_step, ps_decimate,
                n_gyro, n_steps_per_gyro_ps, mu0_ps,
                gyro_window, time_factor, max_plot_points=max_plot_points_local)
            ps_order_label = mu_ps_result["ps_order_label"]
    except _SkipSection as _s:
        print(f"  {_s}")
    except Exception as _e:
        logger.exception("Magnetic-moment analysis failed; continuing with partial μ results.")
        print(f"\n  WARNING: μ analysis failed ({type(_e).__name__}: {_e});\n"
              f"  summary will use whatever μ results completed.\n")

    # --- Unpack mu0 values needed by the summary writer ---
    mu0_rk4  = mu_rk4_result["mu0"]  if mu_rk4_result  else None
    mu0_rkg  = mu_rkg_result["mu0"]  if mu_rkg_result  else None
    mu0_rk45 = mu_rk45_result["mu0"] if mu_rk45_result else None

    # --- Equatorial-crossing markers for the μ figures (black dots) ---
    # Taken from the first available solver's μ result (crossings are a
    # physical event, so any solver marks essentially the same times).
    # To DISABLE the dots: comment out the eq_data= line in either plot
    # call below (the plots default to eq_data=None).
    _mu_eq_src = next((r for r in (mu_ps_result, mu_rk4_result,
                                   mu_rk45_result, mu_rkg_result)
                       if r is not None and "eq_t" in r), None)
    _eq_mudrift = (_mu_eq_src["eq_t"], _mu_eq_src["eq_mudrift"]) if _mu_eq_src else None
    _eq_mushape = (_mu_eq_src["eq_t"], _mu_eq_src["eq_mu_ratio"]) if _mu_eq_src else None

    # mu_deviation is diagnostic-only — skip the image in paper mode. The μ
    # computation above runs either way (its mu0 / mudrift feed the summary writer).
    if USE_FULL_PLOT:
        dplt.mu_deviation(
            fig_folder, stem, particle_type, ps_order_label,
            USE_PLOT_TITLES,
            ps_data=(mu_ps_result["t"], mu_ps_result["mudrift_plot"]) if mu_ps_result else None,
            rk4_data=(mu_rk4_result["t"], mu_rk4_result["mudrift"]) if mu_rk4_result else None,
            rk45_data=(mu_rk45_result["t"], mu_rk45_result["mudrift"]) if mu_rk45_result else None,
            rkg_data=(mu_rkg_result["t"], mu_rkg_result["mudrift"]) if mu_rkg_result else None,
            eq_data=_eq_mudrift,   # equatorial-crossing dots — comment out to disable
        )

    # Instantaneous μ/μ₀ shape over the same window (companion to mu_deviation).
    # PAPER PLOT: produced in both full_plot modes, like ke_error / pphi_error /
    # mu_error — it shows the shape of μ along the orbit, not a diagnostic
    # conservation error. Only invariants_only suppresses it, and it must: that
    # mode skips the μ analysis, so the data would all be None and the figure empty.
    if not INVARIANTS_ONLY:
        dplt.mu_shape(
            fig_folder, stem, particle_type, ps_order_label,
            USE_PLOT_TITLES,
            ps_data=(mu_ps_result["t"], mu_ps_result["mu_ratio_plot"]) if mu_ps_result else None,
            rk4_data=(mu_rk4_result["t"], mu_rk4_result["mu_ratio"]) if mu_rk4_result else None,
            rk45_data=(mu_rk45_result["t"], mu_rk45_result["mu_ratio"]) if mu_rk45_result else None,
            rkg_data=(mu_rkg_result["t"], mu_rkg_result["mu_ratio"]) if mu_rkg_result else None,
            # eq_data=_eq_mushape,   # equatorial-crossing dots — comment out to disable
            # ^ DISABLED for the decimated 20-year run: the markers are sample-level
            #   (no interpolation), so at 20 steps/gyroperiod with decimate=5 one can be
            #   a quarter gyroperiod -- 90 deg of gyrophase -- early. RE-ENABLE before
            #   rebuilding the Section 3.1 demo figure, whose caption promises the dots.
        )


    if DEBUG:
        _, peak = tracemalloc.get_traced_memory()
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

    # Pre-set so the summary writer's `ps_store_stride if USE_PS else 1` is
    # always safe, even if the bounce/drift stream below fails early.
    ps_store_stride = ps_decimate if ps_decimate > 1 else 1

    # Best-effort: the PS bounce/drift stream walks the whole VDS. On failure,
    # bounce_results/drift_results stay None (summary writer tolerates that).
    try:
        if INVARIANTS_ONLY:
            raise _SkipSection("invariants_only: skipping bounce/drift analysis")
        if USE_PS:
            print(f"\n{'='*60}")
            print(f"  Bounce/Drift Statistics")
            print(f"{'='*60}")

            v_eps = npfloat(velocity_epsilon_scale) * v_tau
            user_min_gap = max(min_gap_steps, int(gap_gyro_fraction * T_gyro / ps_step))

            bounce_state = bd.init_bounce_stream_state()
            drift_state  = bd.init_drift_stream_state()

            ps_store_stride = ps_decimate if ps_decimate > 1 else 1
            dt_store = ps_step * ps_store_stride

            with h5py.File(cache_path, "r") as ps_h5:
                ps_y = ps_h5["ps"]["y"]
                n_store = ps_y.shape[1]

                for j0_chunk in range(0, n_store, ps_chunk_steps):
                    j1 = min(j0_chunk + ps_chunk_steps, n_store)

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
    except _SkipSection as _s:
        print(f"  {_s}")
    except Exception as _e:
        logger.exception("PS bounce/drift analysis failed; continuing without bounce/drift stats.")
        print(f"\n  WARNING: bounce/drift analysis failed ({type(_e).__name__}: {_e});\n"
              f"  summary will omit bounce/drift statistics.\n")

    if DEBUG:
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        logger.info(f"Peak memory usage for bounce/drift analysis: {peak / 1024**2:.2f} MB")

    # ===========================================================
    # ========= Write Summary Output to File & CVS Log ==========
    # ===========================================================

    if DEBUG: tracemalloc.start()

    wr.summary_txt_dipoleb(
        summary=summary, run_folder=run_folder, stem=stem,
        dragt_log=dragt_log, bounce_results=bounce_results, drift_results=drift_results,
        gyroperiods=gyroperiods, norm_time=norm_time, cache_path=cache_path,
        USE_PS=USE_PS, USE_RK4=USE_RK4, USE_RK45=USE_RK45, USE_RKG=USE_RKG,
        ps_step=ps_step,
        rk4_step=rk4_step if USE_RK4 else None,
        rkg_step=rkg_step if USE_RKG else None,
        rel_drift_ps=rel_drift_ps if USE_PS else None,
        rel_drift_rk4=rel_drift_rk4 if USE_RK4 else None,
        rel_drift_rk45=rel_drift_rk45 if USE_RK45 else None,
        rel_drift_rkg=rel_drift_rkg if USE_RKG else None,
        rel_pphi_ps=_pphi.get("rel_pphi_ps")   if USE_PS   else None,
        rel_pphi_rk4=_pphi.get("rel_pphi_rk4") if USE_RK4  else None,
        rel_pphi_rk45=_pphi.get("rel_pphi_rk45") if USE_RK45 else None,
        rel_pphi_rkg=_pphi.get("rel_pphi_rkg") if USE_RKG  else None,
        mu0_ps=mu0_ps if USE_PS else None,
        mu0_rk4=mu0_rk4 if USE_RK4 else None,
        mu0_rk45=mu0_rk45 if USE_RK45 else None,
        mu0_rkg=mu0_rkg if USE_RKG else None,
        solution_rk4=solution_rk4 if USE_RK4 else None,
        solution_rkg=solution_rkg if USE_RKG else None,
        y_rk45_common=y_rk45_common if USE_RK45 else None,
        ps_store_stride=ps_store_stride if USE_PS else 1,
        energy_stride=energy_stride if USE_PS else 1,
        npfloat=npfloat,
        compute_mu_ps=mp.compute_mu_ps,
        compute_mu_rk=mp.compute_mu_rk,
        vector_potential=dp.vector_potential,
        mu_max_run=mu_max_run,
    )

    if DEBUG:
        if USE_RK4:logger.debug(f"  rk4 step size = {rk4_step}")
        if USE_RKG: logger.debug(f"  rkg step size = {rkg_step}")
        if USE_RK4: logger.debug(f"  rk4 steps     = {steps_rk4}")
        if USE_RKG: logger.debug(f"  rkg steps     = {steps_rkg}")
        if USE_PS: logger.debug(f"  ps steps      = {steps_ps}")

    # === Write to master simulation log CSV ===
    _method_records = []
    # mudrift may be missing if the μ analysis above failed — pass None then
    # (summarize() and the CSV writer tolerate it, leaving μ error cols blank).
    if USE_RK4:  _method_records.append(("RK4",  steps_rk4, rk4_step, rel_drift_rk4,  mu_rk4_result["mudrift"]  if mu_rk4_result  else None))
    if USE_RK45: _method_records.append(("RK45", steps_ps,  ps_step,  rel_drift_rk45, mu_rk45_result["mudrift"] if mu_rk45_result else None))
    if USE_RKG:  _method_records.append(("RKG",  steps_rkg, rkg_step, rel_drift_rkg,  mu_rkg_result["mudrift"]  if mu_rkg_result  else None))
    if USE_PS:   _method_records.append(("PS",   steps_ps,  ps_step,  rel_drift_ps,   mu_ps_result["mudrift"]   if mu_ps_result   else None))

    wr.master_csv(
        output_folder=output_folder, stem=stem, particle_type=particle_type,
        ke_particle=ke_particle, x_initial=x_initial, y_initial=y_initial,
        z_initial=z_initial, pitch_deg=pitch_deg, phi_deg=phi_deg,
        gyroperiods=gyroperiods,
        dragt_log=dragt_log,
        method_records=_method_records,
        bounce_results=bounce_results,
        drift_results=drift_results,
        mu_max_run=mu_max_run,
    )

    print(f"\nRun Complete → {run_folder}")
    print(f"  Figures → {fig_folder}")


    if DEBUG:
        _, peak = tracemalloc.get_traced_memory()
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
