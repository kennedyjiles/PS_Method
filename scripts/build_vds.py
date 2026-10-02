"""
Rebuild <hash>_full.h5 as an HDF5 Virtual Dataset over <hash>_segNNN.h5 files
without re-running the solver (e.g. after segments were offloaded to another
drive and the auto-stitch missed them).

Usage:
    python scripts/build_vds.py <_rawdata folder> [--hash HASH] [--force]

--hash is auto-detected when the folder holds segments for one run only.
Only the unbroken seg000, seg001, ... prefix is stitched; a gap is reported
and later segments are ignored. Segments are referenced by basename, so the
result must stay in the same folder (or be reached via HDF5_VDS_PREFIX).
Refuses to overwrite a non-virtual _full.h5. The rebuilt file holds only the
ps group plus a summary synthesized from the per-segment summaries, so any
rk4/rk45/rkg data the driver appended to an existing _full.h5 would be LOST:
the script refuses in that case unless --force is given.
"""
import os
import re
import sys
import glob
import argparse

# Make `ps_method` importable when run as `python scripts/build_vds.py`.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ps_method import writers as wr


def detect_hashes(folder):
    """Return the sorted set of run hashes that have segment files in *folder*."""
    rx = re.compile(r"^(.*)_seg\d+\.h5$")
    hashes = set()
    for p in glob.glob(os.path.join(folder, "*_seg*.h5")):
        m = rx.match(os.path.basename(p))
        if m:
            hashes.add(m.group(1))
    return sorted(hashes)


def main():
    parser = argparse.ArgumentParser(
        description="(Re)build <hash>_full.h5 as a VDS over a folder of segments.")
    parser.add_argument("folder", help="the _rawdata folder holding the segments")
    parser.add_argument("--force", action="store_true",
                        help="rebuild even if the existing _full.h5 holds rk4/rk45/rkg data (it is lost)")
    parser.add_argument("--hash", default=None,
                        help="run hash (auto-detected if the folder has just one)")
    args = parser.parse_args()

    folder = args.folder
    if not os.path.isdir(folder):
        print(f"Error: not a directory: {folder}")
        sys.exit(1)

    hash_str = args.hash
    if hash_str is None:
        found = detect_hashes(folder)
        if not found:
            print(f"Error: no <hash>_segNNN.h5 files found in {folder}")
            sys.exit(1)
        if len(found) > 1:
            print("Error: multiple runs' segments in this folder — pass --hash:")
            for h in found:
                n = len(wr.find_committed_segments(h, folder))
                print(f"    --hash {h}   ({n} segments)")
            sys.exit(1)
        hash_str = found[0]
        print(f"Auto-detected hash: {hash_str}")

    committed = wr.find_committed_segments(hash_str, folder)
    contiguous = wr.contiguous_committed_segments(hash_str, folder)
    print(f"Segments present:   {len(committed)}  "
          f"(contiguous prefix: {len(contiguous)})")
    if not contiguous:
        print("Error: no contiguous segments to stitch (is seg000 present?).")
        sys.exit(1)
    if len(contiguous) < len(committed):
        gap = len(contiguous)
        print(f"  WARNING: gap after seg{gap-1:03d} — segments >= seg{gap:03d} are "
              f"orphaned and will NOT be stitched. Refill or move them back.")

    # The rebuilt file carries only the ps group. If the driver already
    # appended RK solver data to the existing _full.h5, rebuilding destroys it.
    existing = os.path.join(folder, f"{hash_str}_full.h5")
    if os.path.exists(existing) and not args.force:
        import h5py
        try:
            with h5py.File(existing, "r") as f:
                rk_groups = [g for g in ("rk4", "rk45", "rkg") if g in f and "y" in f[g]]
        except OSError:
            rk_groups = []
        if rk_groups:
            print(f"Error: {existing} holds solver data for {rk_groups}, which a rebuild "
                  f"would discard. Re-run with --force to rebuild anyway.")
            sys.exit(1)

    try:
        full = wr.build_vds(hash_str, folder)
    except RuntimeError as e:
        print(f"Error: {e}")
        sys.exit(1)

    # Rebuilt outside the driver, the VDS lacks the run-level summary_json/meta
    # that manual loading and trim_h5 need — synthesize them from the
    # per-segment summaries (no-op if already present or segments predate them).
    synthesized = wr.synthesize_full_summary(hash_str, folder, full)

    # Report result + health.
    import h5py
    with h5py.File(full, "r") as f:
        cols = f["ps/y"].shape[1]
        nseg = int(f["ps"].attrs.get("n_segments", len(contiguous)))
        has_summary = "summary_json" in f.attrs
    missing = wr.vds_has_missing_sources(full)
    print(f"\nBuilt: {full}")
    print(f"  segments stitched: {nseg}")
    print(f"  total columns:     {cols:,}")
    print(f"  sources intact:    {'yes' if not missing else 'NO — some are missing!'}")
    if synthesized:
        print(f"  summary_json:      synthesized from per-segment summaries")
    elif has_summary:
        print(f"  summary_json:      present")
    else:
        print(f"  summary_json:      MISSING — segments predate per-segment summaries;")
        print(f"                     manual load / trim_h5 of this file won't work.")
    print("\nVerify:  python scripts/inspect_hdf5.py " + full)


if __name__ == "__main__":
    main()
