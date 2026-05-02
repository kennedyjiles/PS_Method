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


def get_ps_timing(ps_grp):
    """Extract timing info from the PS group attributes."""
    dt = float(ps_grp.attrs["dt"])                  # normalized step size
    decimate = int(ps_grp.attrs.get("decimate", 1))
    store_stride = decimate if decimate > 1 else 1
    dt_store = dt * store_stride                     # normalized time between stored columns
    n_cols = ps_grp["y"].shape[1]
    return dt, store_stride, dt_store, n_cols


def get_norm_time_factor(f):
    """Get the normalization factor (seconds per normalized time unit)
    from the h5 metadata.  Returns norm_time in seconds."""
    # Try meta group first (always present after append_results)
    if "meta" in f and "norm_time" in f["meta"].attrs:
        return float(f["meta"].attrs["norm_time"])
    # Fall back to params_json
    if "params_json" in f.attrs:
        params = json.loads(f.attrs["params_json"])
        if "norm_time" in params:
            return float(params["norm_time"])
    return None


def seconds_to_columns(window_s, norm_time_s, dt_store_norm):
    """Convert a physical time window (seconds) to a column count.

    Parameters
    ----------
    window_s : float
        Desired window in physical seconds.
    norm_time_s : float
        Physical duration of the full simulation in seconds.
    dt_store_norm : float
        Normalized time between stored columns.

    Returns
    -------
    n_cols : int
        Number of stored columns that span the requested window.
    """
    # norm_time attr is total normalized time for the full run.
    # But we need the *factor*: seconds_per_normalized_unit.
    # From dipoleb.py: physical_time = norm_time * tau_time
    # where tau_time = mass / (q * B0).
    # But we don't need tau_time explicitly — we have:
    #   total_physical_seconds = norm_time_s  (meta/physical_time)
    #   total_normalized_time  = n_cols_total * dt_store_norm
    # So: tau_factor = total_physical_seconds / total_normalized_time
    #     window_normalized = window_s / tau_factor
    #     n_cols = window_normalized / dt_store_norm
    # Simplifying:
    #     n_cols = window_s / (total_physical_seconds / n_cols_total)
    #            = window_s * n_cols_total / total_physical_seconds
    # But we need total_physical_seconds and n_cols_total passed in.
    # We handle this in the caller.
    raise NotImplementedError("Use seconds_to_columns_direct instead")


def write_trimmed(src_path, dst_path, ps_grp_src, col_start, col_end,
                  window_s, end_label, f_src):
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
        Column range to extract (half-open).
    window_s : float
        Requested window in seconds (for metadata).
    end_label : str
        "first" or "last".
    f_src : h5py.File
        Source file handle (for copying root attrs, meta, etc.).
    """
    n_keep = col_end - col_start
    y_src = ps_grp_src["y"]
    n_rows = y_src.shape[0]

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

        # --- Copy any RK groups verbatim (unlikely for massive runs) ---
        for grp_name in ("rk4", "rk45", "rkg"):
            if grp_name in f_src:
                f_src.copy(grp_name, f_dst)


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

        norm_time = None
        if "meta" in f and "norm_time" in f["meta"].attrs:
            norm_time = float(f["meta"].attrs["norm_time"])

        if physical_time is None:
            # Try to get from params
            if "params_json" in f.attrs:
                params = json.loads(f.attrs["params_json"])
                physical_time = params.get("physical_time")
                norm_time = params.get("norm_time")

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

        # --- Write ---
        if do_first:
            print(f"Writing {dst_first} ...")
            write_trimmed(src, dst_first, ps_grp, 0, min(n_window, n_cols),
                          args.window, "first", f)
            out_size = os.path.getsize(dst_first) / (1024**2)
            print(f"  Done: {out_size:,.1f} MB")

        if do_last:
            print(f"Writing {dst_last} ...")
            write_trimmed(src, dst_last, ps_grp,
                          max(0, n_cols - n_window), n_cols,
                          args.window, "last", f)
            out_size = os.path.getsize(dst_last) / (1024**2)
            print(f"  Done: {out_size:,.1f} MB")

    print("\nTrimming complete.")


if __name__ == "__main__":
    main()
