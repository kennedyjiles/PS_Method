"""
Trim a large dipoleb HDF5 file to keep only the first and/or last
N seconds of PS trajectory data.

Usage:
    python scripts/trim_h5.py path/to/run.h5 --window 60

    Produces:
        path/to/run_first_60.0s.h5
        path/to/run_last_60.0s.h5

    Each trimmed file preserves the full h5 structure (metadata, attrs,
    PS group layout) so it can be fed directly back into the plotting
    and analysis pipeline.

Options:
    --window SECONDS    Duration to keep at each end (physical seconds)
    --first-only        Only produce the "first" file
    --last-only         Only produce the "last" file
    --dry-run           Print what would be done without writing files
"""

import sys
import os
import argparse
import json
import h5py
import numpy as np
import yaml


def get_ps_timing(ps_grp):
    """Extract timing info from the PS group attributes."""
    dt = float(ps_grp.attrs["dt"])                  # normalized step size
    decimate = int(ps_grp.attrs.get("decimate", 1))
    store_stride = decimate if decimate > 1 else 1
    dt_store = dt * store_stride                     # normalized time between stored columns
    n_cols = ps_grp["y"].shape[1]
    return dt, store_stride, dt_store, n_cols


def write_trimmed(src_path, dst_path, ps_grp_src, col_start, col_end,
                  window_s, end_label, f_src, physical_time=None):
    """Write a trimmed h5 file containing columns [col_start, col_end).

    Parameters
    ----------
    src_path : str
        Original file path (for provenance).
    dst_path : str
        Output file path.
    ps_grp_src : h5py.Group
        Source PS group (opened for reading).
    col_start, col_end : int
        PS-relative column range to extract (half-open).
    window_s : float
        Requested window in seconds (for metadata).
    end_label : str
        "first" or "last".
    f_src : h5py.File
        Source file handle (for copying root attrs, meta, etc.).
    physical_time : float, optional
        Total physical duration of the source run. If provided, RK groups
        are sliced based on each method's own column count (accounts for
        different dt across solvers). Falls back to PS-relative slicing
        when None.
    """
    n_keep = col_end - col_start
    y_src = ps_grp_src["y"]
    n_rows = y_src.shape[0]
    ps_n_cols = y_src.shape[1]

    with h5py.File(dst_path, "w") as f_dst:
        # --- Copy root-level attrs ---
        for attr_name in f_src.attrs:
            f_dst.attrs[attr_name] = f_src.attrs[attr_name]

        # --- PS group ---
        ps_dst = f_dst.create_group("ps")

        # Copy all PS attrs, then patch the ones that change
        for attr_name in ps_grp_src.attrs:
            ps_dst.attrs[attr_name] = ps_grp_src.attrs[attr_name]

        # Patch step count to reflect trimmed size
        ps_dst.attrs["steps_trimmed"] = n_keep
        ps_dst.attrs["steps_original"] = int(ps_grp_src.attrs["steps"])

        # Trim info for provenance and time reconstruction
        dt_store = float(ps_grp_src.attrs["dt"])
        decimate = int(ps_grp_src.attrs.get("decimate", 1))
        store_stride = decimate if decimate > 1 else 1
        dt_store_norm = dt_store * store_stride

        ps_dst.attrs["trim_end"] = end_label
        ps_dst.attrs["trim_window_s"] = window_s
        ps_dst.attrs["trim_col_start"] = col_start
        ps_dst.attrs["trim_col_end"] = col_end
        ps_dst.attrs["trim_t0_norm"] = col_start * dt_store_norm
        ps_dst.attrs["trim_source"] = os.path.basename(src_path)

        # --- Copy y dataset in chunks to avoid loading all into memory ---
        chunk_cols = min(200_000, n_keep)
        y_dst = ps_dst.create_dataset(
            "y",
            shape=(n_rows, n_keep),
            dtype=y_src.dtype,
            chunks=(n_rows, min(chunk_cols, n_keep)),
            compression="gzip",
            compression_opts=1,
            shuffle=True,
        )

        for j0 in range(0, n_keep, chunk_cols):
            j1 = min(j0 + chunk_cols, n_keep)
            y_dst[:, j0:j1] = y_src[:, col_start + j0 : col_start + j1]

        # --- Copy orders dataset (trimmed to same range) ---
        if "orders" in ps_grp_src:
            orders_src = ps_grp_src["orders"]
            orders_data = orders_src[col_start:col_end]
            ps_dst.create_dataset(
                "orders", data=orders_data,
                compression="gzip", compression_opts=1, shuffle=True,
            )

        # --- Copy meta group ---
        if "meta" in f_src:
            f_src.copy("meta", f_dst)

        # --- Trim RK groups using the same column range as PS ---
        # rk4, rk45, rkg are all integrated on the PS fixed grid (rk45 via
        # t_eval=t_common in dipoleb.py), so column indices map 1:1 to PS.
        # Convention matches PS: t0 stays 0, the trimmed file restarts its
        # time axis at zero. Original-run offset is recorded for provenance.
        for grp_name in ("rk4", "rk45", "rkg"):
            if grp_name not in f_src:
                continue
            src_grp = f_src[grp_name]
            dst_grp = f_dst.create_group(grp_name)

            for attr_name in src_grp.attrs:
                dst_grp.attrs[attr_name] = src_grp.attrs[attr_name]

            # Disabled solvers (e.g. flux_map runs are PS-only) leave the
            # group as a header-only placeholder. Nothing to trim.
            if "y" not in src_grp:
                continue

            y_src_rk = src_grp["y"]
            # rkg stores y as (n_steps, n_dim); rk4/rk45 store (n_dim, n_steps).
            time_axis = 0 if grp_name == "rkg" else 1
            n_cols_grp = y_src_rk.shape[time_axis]

            # When a method runs at a different dt than PS its column count
            # differs too. Use physical-time correspondence (window_s of
            # total physical_time) to derive each method's own slice rather
            # than reusing PS column indices, so all groups represent the
            # same physical time window.
            if physical_time and physical_time > 0 and n_cols_grp != ps_n_cols:
                n_window_grp = max(1, int(np.ceil(window_s / physical_time * n_cols_grp)))
                if end_label == "last":
                    cs = max(0, n_cols_grp - n_window_grp)
                    ce = n_cols_grp
                else:
                    cs = 0
                    ce = min(n_window_grp, n_cols_grp)
                print(f"  Note: {grp_name} has {n_cols_grp:,} cols (PS has "
                      f"{ps_n_cols:,}) — sliced [{cs:,}..{ce:,}) per its own dt.")
            else:
                cs = min(col_start, n_cols_grp)
                ce = min(col_end, n_cols_grp)
            n_keep_grp = ce - cs

            if grp_name == "rkg":
                y_data = y_src_rk[cs:ce, :]
                y_initial = np.array(y_src_rk[0, :])
            else:
                y_data = y_src_rk[:, cs:ce]
                y_initial = np.array(y_src_rk[:, 0])

            dst_grp.create_dataset(
                "y", data=y_data,
                compression="gzip", compression_opts=1, shuffle=True,
            )

            # Source initial state (always col/row 0 of the original run).
            # Lets downstream baseline computations (mu0, etc.) use the true
            # IC instead of the trim-window-start state.
            dst_grp.create_dataset("y_initial", data=y_initial)

            # rk45 carries its own absolute-time array; rebase so trimmed
            # file's t starts at 0, matching PS/RK4/RKG plot convention.
            if grp_name == "rk45" and "t" in src_grp:
                t_src = src_grp["t"][cs:ce]
                t0_orig = float(t_src[0]) if t_src.size else 0.0
                dst_grp.create_dataset(
                    "t", data=(t_src - t0_orig),
                    compression="gzip", compression_opts=1, shuffle=True,
                )
                dst_grp.attrs["trim_t0_norm"] = t0_orig
            elif "dt" in src_grp.attrs:
                dst_grp.attrs["trim_t0_norm"] = cs * float(src_grp.attrs["dt"])

            if "steps" in src_grp.attrs:
                dst_grp.attrs["steps_original"] = int(src_grp.attrs["steps"])
                dst_grp.attrs["steps"] = n_keep_grp
            dst_grp.attrs["steps_trimmed"] = n_keep_grp
            dst_grp.attrs["trim_end"] = end_label
            dst_grp.attrs["trim_window_s"] = window_s
            dst_grp.attrs["trim_col_start"] = cs
            dst_grp.attrs["trim_col_end"] = ce
            dst_grp.attrs["trim_source"] = os.path.basename(src_path)


def write_companion_yml(dst_h5_path, src_summary, base_yml_path, end_label="last"):
    """Write a self-contained yml next to a trimmed h5.

    The yml is generated from dipoleb's base.yml as a starting point, then
    has its identity parameters (particle, energy, pitch, position, etc.)
    overridden from the source h5's summary_json. It carries
    ``base_config: none`` so load_config does NOT merge with base.yml at
    runtime — identity comes entirely from this snapshot, matching the h5.

    The ``manual_h5_path`` is written as the basename of the trimmed h5,
    so dipoleb.py resolves it relative to the yml's own directory.

    Plotting defaults are aligned with the trim end: a ``_first`` trim gets
    ``slice_mode: first`` / ``gyro_window: first`` so the slice plot shows
    the data the file actually contains; a ``_last`` trim gets ``last`` /
    ``last``. Override in the yml if you want the opposite view.
    """
    with open(base_yml_path) as f:
        config = yaml.safe_load(f)

    meta = src_summary["meta"]
    config["particle"]    = str(meta["particle"]).lower()
    config["energy_eV"]   = float(meta["energy_eV"])
    config["pitch_deg"]   = float(meta["pitch_deg"])
    config["phi_deg"]     = float(meta["phi_deg"])
    config["x_initial"]   = float(meta["x0"])
    config["y_initial"]   = float(meta["y0"])
    config["z_initial"]   = float(meta["z0"])
    config["gyroperiods"] = float(meta["gyroperiods"])
    config["use_float128"] = (str(meta.get("dtype", "float64")) == "float128")

    plotting = config.setdefault("plotting", {})
    plotting["slice_mode"]  = end_label
    plotting["gyro_window"] = end_label

    config["base_config"]    = "none"
    config["manual_h5_path"] = os.path.basename(dst_h5_path)

    # Hoist the most-edited fields to the top so they're visible at first
    # glance: base_config (self-contained marker), manual_h5_path (points
    # at the trimmed h5), output_root (where outputs land).
    _top = ("base_config", "manual_h5_path", "output_root")
    ordered = {k: config[k] for k in _top if k in config}
    for k, v in config.items():
        if k not in _top:
            ordered[k] = v

    yml_path = os.path.splitext(dst_h5_path)[0] + ".yml"
    header = (
        f"# Auto-generated by trim_h5 alongside {os.path.basename(dst_h5_path)}.\n"
        f"# Self-contained — base.yml is NOT merged at load time.\n"
        f"# Identity parameters come from the source run's summary_json.\n"
        f"#\n"
        f"# Run with:\n"
        f"#   python run.py {yml_path}\n"
        f"# (Paths under data/ auto-trigger replot mode — no --replot needed.)\n"
        f"#\n"
        f"# manual_h5_path is already set to {os.path.basename(dst_h5_path)} so this\n"
        f"# yml replots the trim without re-running solvers. To customize:\n"
        f"#   • Edit plotting / slice / output fields below.\n"
        f"#   • Toggle solvers (e.g. solvers.rk4: false) to drop a method from\n"
        f"#     plots without triggering a fresh solver run.\n"
        f"#   • Identity / step-size fields are overridden from the h5 — editing\n"
        f"#     them here has no effect.\n\n"
    )
    with open(yml_path, "w") as f:
        f.write(header)
        yaml.dump(ordered, f, sort_keys=False, default_flow_style=False)
    return yml_path


def main():
    parser = argparse.ArgumentParser(
        description="Trim a dipoleb HDF5 file to first/last N seconds.")
    parser.add_argument("h5_path", help="Path to the source HDF5 file")
    parser.add_argument("--window", type=float, required=True,
                        help="Duration to keep at each end (physical seconds)")
    parser.add_argument("--first-only", action="store_true",
                        help="Only produce the 'first' file")
    parser.add_argument("--last-only", action="store_true",
                        help="Only produce the 'last' file")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print plan without writing files")
    args = parser.parse_args()

    if args.first_only and args.last_only:
        print("Error: --first-only and --last-only are mutually exclusive.")
        sys.exit(1)

    src = args.h5_path
    if not os.path.isfile(src):
        print(f"Error: file not found: {src}")
        sys.exit(1)

    with h5py.File(src, "r") as f:
        if "ps" not in f:
            print("Error: no 'ps' group in this HDF5 file.")
            sys.exit(1)

        ps_grp = f["ps"]
        dt, store_stride, dt_store_norm, n_cols = get_ps_timing(ps_grp)

        # Get physical time for conversion
        physical_time = None
        if "meta" in f and "physical_time" in f["meta"].attrs:
            physical_time = float(f["meta"].attrs["physical_time"])

        if physical_time is None:
            # Try to get from params
            if "params_json" in f.attrs:
                params = json.loads(f.attrs["params_json"])
                physical_time = params.get("physical_time")

        if physical_time is None or physical_time <= 0:
            print("Error: cannot determine physical_time from h5 metadata.")
            print("       Ensure the file has meta/physical_time or params_json.")
            sys.exit(1)

        # Convert seconds to columns
        # total_physical_time corresponds to n_cols stored columns
        secs_per_col = physical_time / n_cols
        n_window = int(np.ceil(args.window / secs_per_col))

        if n_window >= n_cols:
            print(f"Warning: requested window ({args.window:.2f} s) covers the "
                  f"entire file ({physical_time:.2f} s, {n_cols:,} columns).")
            print("         Nothing to trim.")
            sys.exit(0)

        # --- Report ---
        src_size_mb = os.path.getsize(src) / (1024**2)
        trim_frac = n_window / n_cols
        est_size = src_size_mb * trim_frac

        print(f"Source:          {src}")
        print(f"Source size:     {src_size_mb:,.1f} MB")
        print(f"Total columns:   {n_cols:,}")
        print(f"Physical time:   {physical_time:.4g} s")
        print(f"Secs per column: {secs_per_col:.4g} s")
        print(f"Window request:  {args.window:.4g} s  →  {n_window:,} columns")
        print(f"Trim fraction:   {trim_frac:.2%} of original per file")
        print(f"Est. size/file:  ~{est_size:,.1f} MB")
        print()

        do_first = not args.last_only
        do_last = not args.first_only

        # Place trimmed files in a trimmed/ subfolder next to the original
        src_dir = os.path.dirname(src)
        src_stem = os.path.splitext(os.path.basename(src))[0]
        if src_stem.endswith("_full"):
            src_stem = src_stem[:-len("_full")]
        trim_dir = os.path.join(src_dir, "trimmed")

        if do_first:
            col_start = 0
            col_end = min(n_window, n_cols)
            dst_first = os.path.join(trim_dir, f"{src_stem}_first_{args.window}s.h5")
            print(f"FIRST: columns [{col_start:,} .. {col_end:,})  →  {dst_first}")
            t_start_s = col_start * secs_per_col
            t_end_s = col_end * secs_per_col
            print(f"       time [{t_start_s:.4g} .. {t_end_s:.4g}] s")

        if do_last:
            col_start_last = max(0, n_cols - n_window)
            col_end_last = n_cols
            dst_last = os.path.join(trim_dir, f"{src_stem}_last_{args.window}s.h5")
            print(f"LAST:  columns [{col_start_last:,} .. {col_end_last:,})  →  {dst_last}")
            t_start_s = col_start_last * secs_per_col
            t_end_s = col_end_last * secs_per_col
            print(f"       time [{t_start_s:.4g} .. {t_end_s:.4g}] s")

        print()

        if args.dry_run:
            print("(dry run — no files written)")
            return

        # Create trimmed/ folder
        os.makedirs(trim_dir, exist_ok=True)

        # --- Resolve base.yml for the companion yml writer ---
        # scripts/trim_h5.py → project_root → configs/dipoleb/base.yml
        _project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        _base_yml = os.path.join(_project_root, "configs", "dipoleb", "base.yml")

        # Pull source summary once for the companion yml(s)
        _src_summary = None
        if "summary_json" in f.attrs:
            _src_summary = json.loads(f.attrs["summary_json"])

        # --- Write ---
        written_ymls = []
        if do_first:
            print(f"Writing {dst_first} ...")
            write_trimmed(src, dst_first, ps_grp, 0, min(n_window, n_cols),
                          args.window, "first", f, physical_time=physical_time)
            out_size = os.path.getsize(dst_first) / (1024**2)
            print(f"  Done: {out_size:,.1f} MB")
            if _src_summary is not None and os.path.isfile(_base_yml):
                yml_first = write_companion_yml(dst_first, _src_summary, _base_yml,
                                                end_label="first")
                print(f"  Wrote companion yml: {os.path.basename(yml_first)}")
                written_ymls.append(yml_first)
            else:
                print("  (skipped companion yml — source missing summary_json or base.yml not found)")

        if do_last:
            print(f"Writing {dst_last} ...")
            write_trimmed(src, dst_last, ps_grp,
                          max(0, n_cols - n_window), n_cols,
                          args.window, "last", f, physical_time=physical_time)
            out_size = os.path.getsize(dst_last) / (1024**2)
            print(f"  Done: {out_size:,.1f} MB")
            if _src_summary is not None and os.path.isfile(_base_yml):
                yml_last = write_companion_yml(dst_last, _src_summary, _base_yml,
                                               end_label="last")
                print(f"  Wrote companion yml: {os.path.basename(yml_last)}")
                written_ymls.append(yml_last)
            else:
                print("  (skipped companion yml — source missing summary_json or base.yml not found)")

    print("\nTrimming complete.")

    if written_ymls:
        # Show paths relative to cwd if possible — easier to copy-paste.
        cwd = os.getcwd()
        rel_ymls = [
            os.path.relpath(p, cwd) if p.startswith(cwd) else p
            for p in written_ymls
        ]
        print("\nNext steps — replot from a trimmed file:")
        for p in rel_ymls:
            print(f"    python run.py {p}")
        print("\nNotes:")
        print("  • Paths under data/ auto-trigger replot mode (no --replot needed).")
        print("  • The companion yml is self-contained (base_config: none) — identity")
        print("    fields match the source h5. Edit plotting / output overrides freely")

if __name__ == "__main__":
    main()
