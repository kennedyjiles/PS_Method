"""
(Re)build the stitched <hash>_full.h5 Virtual Dataset over a folder of
segment files — without re-running the solver.

WHEN YOU NEED THIS
------------------
Segmented runs (ps_segment_gyroperiods > 0) write one <hash>_segNNN.h5 per
segment and auto-stitch them into <hash>_full.h5 at the end of the run. That
auto-stitch scans the folder the run wrote to. So you need to rebuild by hand
whenever the segments and the VDS get separated, e.g.:

  * You moved segments to an external drive *during* the run, so the auto-built
    _full.h5 only covered the segments still in the local folder.
  * You gathered segments from several places into one folder and want a fresh
    stitch over the complete set.
  * You deleted _full.h5 and just want to regenerate the (tiny) pointer file.

The VDS is only pointers — this is fast and copies no trajectory data. It
records each segment by *basename*, so the resulting _full.h5 must live in the
SAME folder as its _segNNN.h5 files (that's how HDF5 resolves them).

USAGE
-----
    python scripts/build_vds.py <folder> [--hash HASH]

    <folder>   the _rawdata folder holding the <hash>_segNNN.h5 files
    --hash     the run hash (the NNN-less stem). Optional: if omitted and the
               folder holds segments for exactly one hash, it's auto-detected.

EXAMPLES
    # after moving all segments to the drive, stitch them there:
    python scripts/build_vds.py /Volumes/Jiles_thesi/data/dipoleb/run/_rawdata

    # be explicit when several runs share a folder:
    python scripts/build_vds.py path/to/_rawdata --hash a51bff

NOTES
    * Only the unbroken 0,1,2,… segment prefix is stitched; a gap (missing
      middle segment) stops the stitch there and is reported.
    * It refuses to overwrite a real (non-virtual) <hash>_full.h5, so it can't
      clobber a single-file run that happens to share the name.
    * Verify afterwards with:  python scripts/inspect_hdf5.py <folder>/<hash>_full.h5
      (look for "VIRTUAL: N/N sources present").
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
    if len(contiguous) < len(committed):
        gap = len(contiguous)
        print(f"  WARNING: gap after seg{gap-1:03d} — segments >= seg{gap:03d} are "
              f"orphaned and will NOT be stitched. Refill or move them back.")
    if not contiguous:
        print("Error: no contiguous segments to stitch (is seg000 present?).")
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
