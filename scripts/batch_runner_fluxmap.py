#!/usr/bin/env python3
"""
Dwell Map Batch Runner
======================
Preset-driven parameter sweep for building meridional dwell-occupancy maps
from orbit integrations.  Sweeps a fine L-shell grid at each energy, producing
h5 trajectories that flux_map_builder.py consumes.

Presets cover 10 MeV – 400 MeV protons; L-shell and pitch ranges are
configured per preset.  Custom scans override any axis from the CLI.

Output lands under  data/dipoleb/fluxmap_<tag>/  (one folder per preset or
custom group name).  Folder name pattern is kept as `fluxmap_*` for backward
compatibility with existing data — only the user-facing terminology has
shifted from "flux" to "dwell occupancy" to reflect what the maps actually
are (trajectory dwell time per Cartesian bin, without the toroidal Jacobian
or pitch-angle integration required for a true omnidirectional flux).

# After runs complete, build the dwell map:
python scripts/plots/flux_map_builder.py --group fluxmap_10mev
python scripts/plots/flux_map_builder.py --group fluxmap_all --per-energy
"""

import os
import sys
import copy
import json
import argparse
import subprocess
import time
import yaml
import numpy as np
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

# ─── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT  = os.path.dirname(SCRIPT_DIR)
PROGRESS_PATH = os.path.join(SCRIPT_DIR, "batch_progress_fluxmap.json")
BATCH_TMP_DIR = os.path.join(SCRIPT_DIR, "batch_tmp")
BASE_YML      = os.path.join(PROJECT_ROOT, "configs", "dipoleb", "base.yml")


# ═══════════════════════════════════════════════════════════════════════════════
#  Presets — each defines one energy's L-sweep for the dwell map
# ═══════════════════════════════════════════════════════════════════════════════
L_start = 1.1   # Avoid L=1 singularity (x=1 is just outside the surface)
L_step = 0.1
pitch_deg = 15.0
gyroperiods = 1e5
# Note: phi_deg (the initial GYROPHASE, set in base.yml = 90°) is intentionally
# not overridden here. It only sets where in the gyration circle each particle
# starts; over 1e5 gyroperiods it averages out and does not affect the dwell map.


PRESETS = {
    # ── Single-energy sweeps ──────────────────────────────────────────────────
    "10mev": {
        "description": "10 MeV proton, L = 1.0 – 3.8, dL = 0.1",
        "energy_eV":   10e6,
        "L_start":     L_start,
        "L_stop":      3.8,
        "L_step":      L_step,
        "pitch_deg":   pitch_deg,
        "gyroperiods": gyroperiods,
    },
    "15mev": {
        "description": "15 MeV proton, L = 1.0 – 3.6, dL = 0.1",
        "energy_eV":   15e6,
        "L_start":     L_start,
        "L_stop":      3.6,
        "L_step":      L_step,
        "pitch_deg":   pitch_deg,
        "gyroperiods": gyroperiods,
    },
    "20mev": {
        "description": "20 MeV proton, L = 1.0 – 3.5, dL = 0.1",
        "energy_eV":   20e6,
        "L_start":     L_start,
        "L_stop":      3.5,
        "L_step":      L_step,
        "pitch_deg":   pitch_deg,
        "gyroperiods": gyroperiods,
    },
    "30mev": {
        "description": "30 MeV proton, L = 1.0 – 3.2, dL = 0.1",
        "energy_eV":   30e6,
        "L_start":     L_start,
        "L_stop":      3.2,
        "L_step":      L_step,
        "pitch_deg":   pitch_deg,
        "gyroperiods": gyroperiods,
    },
    "50mev": {
        "description": "50 MeV proton, L = 1.0 – 3.0, dL = 0.1",
        "energy_eV":   50e6,
        "L_start":     L_start,
        "L_stop":      3.0,
        "L_step":      L_step,
        "pitch_deg":   pitch_deg,
        "gyroperiods": gyroperiods,
    },
        "60mev": {
        "description": "60 MeV proton, L = 1.0 – 2.9, dL = 0.1",
        "energy_eV":   60e6,
        "L_start":     L_start,
        "L_stop":      2.9,
        "L_step":      L_step,
        "pitch_deg":   pitch_deg,
        "gyroperiods": gyroperiods,
    },
        "80mev": {
        "description": "80 MeV proton, L = 1.0 – 2.8, dL = 0.1",
        "energy_eV":   80e6,
        "L_start":     L_start,
        "L_stop":      2.8,
        "L_step":      L_step,
        "pitch_deg":   pitch_deg,
        "gyroperiods": gyroperiods,
    },
    "100mev": {
        "description": "100 MeV proton, L = 1.0 – 2.7, dL = 0.1",
        "energy_eV":   100e6,
        "L_start":     L_start,
        "L_stop":      2.7,
        "L_step":      L_step,
        "pitch_deg":   pitch_deg,
        "gyroperiods": gyroperiods,
    },
    "200mev": {
        "description": "200 MeV proton, L = 1.0 – 2.4, dL = 0.1",
        "energy_eV":   200e6,
        "L_start":     L_start,
        "L_stop":      2.4,
        "L_step":      L_step,
        "pitch_deg":   pitch_deg,
        "gyroperiods": gyroperiods,
    },
    "400mev": {
        "description": "400 MeV proton, L = 1.0 – 2.1, dL = 0.1",
        "energy_eV":   400e6,
        "L_start":     L_start,
        "L_stop":      2.1,
        "L_step":      L_step,
        "pitch_deg":   pitch_deg,
        "gyroperiods": gyroperiods,
    },
}

# ── Meta-presets (expand to multiple single-energy presets) ───────────────────
META_PRESETS = {
    "all":    ["10mev", "15mev", "20mev", "30mev", "50mev", "60mev", "80mev", "100mev", "200mev", "400mev"],
    "low":    ["10mev", "15mev", "20mev", "30mev"],
    "medium": ["50mev", "60mev", "80mev"],
    "high":   ["100mev", "200mev", "400mev"],
}


# ═════════════════════════════════════════════════════2══════════════════════════
#  Run-list construction
# ═══════════════════════════════════════════════════════════════════════════════

def _L_grid(L_start, L_stop, L_step):
    """Build inclusive L-shell array, rounded to avoid float noise."""
    vals = np.arange(L_start, L_stop + L_step / 2, L_step)
    return [round(v, 4) for v in vals]


def expand_preset(name):
    """Resolve a preset (or meta-preset) into a list of (preset_name, preset_dict) pairs."""
    if name in META_PRESETS:
        pairs = []
        for sub in META_PRESETS[name]:
            pairs.extend(expand_preset(sub))
        return pairs
    if name in PRESETS:
        return [(name, PRESETS[name])]
    raise ValueError(f"Unknown preset: {name!r}. Use --list-presets to see options.")


def build_runs_from_preset(preset_dict, group):
    """Convert a single preset dict into a list of run dicts."""
    Ls = _L_grid(preset_dict["L_start"], preset_dict["L_stop"], preset_dict["L_step"])
    runs = []
    for L in Ls:
        runs.append({
            "energy_eV":   float(preset_dict["energy_eV"]),
            "x_initial":   L,
            "pitch_deg":   float(preset_dict["pitch_deg"]),
            "gyroperiods": float(preset_dict["gyroperiods"]),
            "group":       group,
        })
    return runs


def build_chaotic_rerun_list(group):
    """Read the group's master CSV and return run dicts for CHAOTIC orbits."""
    import pandas as pd

    csv_path = os.path.join(PROJECT_ROOT, "data", "dipoleb", group, "master_simulation_log.csv")
    if not os.path.exists(csv_path):
        print(f"  No master CSV at {csv_path}")
        return []

    df = pd.read_csv(csv_path, dtype={"run_id": str})
    if "orbit_character" not in df.columns:
        print("  No orbit_character column in master CSV.")
        return []

    mask = df["orbit_character"].fillna("").str.upper() == "CHAOTIC"
    chaotic = df[mask]
    if chaotic.empty:
        print("  No chaotic orbits found.")
        return []

    # Deduplicate by physics parameters (same cell may have multiple solver rows)
    param_cols = ["energy_eV", "L_eff", "pitch_deg"]
    existing = [c for c in param_cols if c in chaotic.columns]
    chaotic = chaotic.drop_duplicates(subset=existing)

    runs = []
    for _, row in chaotic.iterrows():
        # Use x (launch position), NOT L_eff (gyroradius-corrected)
        L_val = float(row["x"])
        runs.append({
            "energy_eV":      float(row["energy_eV"]),
            "x_initial":      round(L_val, 4),
            "pitch_deg":      float(row["pitch_deg"]),
            "gyroperiods":    float(row["gyroperiods"]),
            "group":          group,
            "use_adaptive":   True,
            "clean_before_run": True,   # wipe old fixed-step data
        })

    print(f"  Found {len(runs)} chaotic orbits to rerun with adaptive stepping.")
    return runs


def build_runs_custom(energy_eV, L_start, L_stop, L_step, pitch_deg, gyroperiods, group):
    """Build run list from CLI overrides."""
    Ls = _L_grid(L_start, L_stop, L_step)
    runs = []
    for L in Ls:
        runs.append({
            "energy_eV":   float(energy_eV),
            "x_initial":   L,
            "pitch_deg":   float(pitch_deg),
            "gyroperiods": float(gyroperiods),
            "group":       group,
        })
    return runs


# ═══════════════════════════════════════════════════════════════════════════════
#  Config writing + execution (mirrors batch_runner.py pattern)
# ═══════════════════════════════════════════════════════════════════════════════

def run_key(run):
    return f"E{run['energy_eV']:.1e}_L{run['x_initial']:.2f}_P{run['pitch_deg']:.1f}"


def write_config(run, config_path):
    """Write a minimal per-worker YAML that overrides base.yml."""
    cfg = {
        "base_config":  os.path.abspath(BASE_YML),
        "batch_group":  run["group"],
        "energy_eV":    float(run["energy_eV"]),
        "x_initial":    float(run["x_initial"]),
        "pitch_deg":    float(run["pitch_deg"]),
        "gyroperiods":  float(run["gyroperiods"]),
        "particle":     "proton",
        "read_data":    False,
        "solvers":      {"adaptive": bool(run.get("use_adaptive", False))},
    }
    with open(config_path, "w") as f:
        f.write("# Auto-generated dwell-map batch config — merged with base.yml\n")
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)


def execute_single_run(run):
    """Run one dipoleb.py subprocess. Returns (key, status, elapsed_min, error)."""
    key = run_key(run)

    run_folder = os.path.join(PROJECT_ROOT, "data", "dipoleb",
                              run["group"], f"fm_{key}")

    # Clean old run folder before rerun (e.g. chaotic reruns with adaptive)
    if run.get("clean_before_run", False):
        import shutil
        if os.path.isdir(run_folder):
            shutil.rmtree(run_folder)

    config_path = os.path.join(BATCH_TMP_DIR, f"fm_{key}.yml")
    write_config(run, config_path)

    start = time.time()
    try:
        result = subprocess.run(
            [sys.executable, "dipoleb.py", config_path],
            cwd=PROJECT_ROOT,
            timeout=3600 * 12,   # 12-hour timeout per cell
            capture_output=True,
            text=True,
        )
        elapsed = (time.time() - start) / 60

        if os.path.exists(config_path):
            os.remove(config_path)

        if result.returncode == 0:
            # A clean exit is NOT proof of success: some runs (e.g. a
            # particle lost almost immediately) exit 0 without ever
            # writing their master_simulation_log.csv row. Verify the
            # output actually exists, otherwise this run would be marked
            # "completed" and silently skipped on every future --resume.
            per_run_csv = os.path.join(run_folder, "master_simulation_log.csv")
            if os.path.exists(per_run_csv):
                return (key, "completed", elapsed, None)
            tail = "\n".join((result.stdout or "").strip().split("\n")[-4:])
            return (key, "no_output", elapsed,
                    "exit 0 but no master_simulation_log.csv written. "
                    "Last stdout:\n" + tail)
        err = "\n".join(result.stderr.strip().split("\n")[-5:])
        return (key, "failed", elapsed, err)

    except subprocess.TimeoutExpired:
        elapsed = (time.time() - start) / 60
        if os.path.exists(config_path):
            os.remove(config_path)
        return (key, "timeout", elapsed, "12h timeout exceeded")
    except Exception as e:
        elapsed = (time.time() - start) / 60
        if os.path.exists(config_path):
            os.remove(config_path)
        return (key, "error", elapsed, str(e))


# ═══════════════════════════════════════════════════════════════════════════════
#  Progress tracking
# ═══════════════════════════════════════════════════════════════════════════════

def load_progress():
    if os.path.exists(PROGRESS_PATH):
        with open(PROGRESS_PATH, "r") as f:
            return json.load(f)
    return {"completed": [], "failed": [], "started": None}


def save_progress(progress):
    with open(PROGRESS_PATH, "w") as f:
        json.dump(progress, f, indent=2)


# ═══════════════════════════════════════════════════════════════════════════════
#  Parallel execution
# ═══════════════════════════════════════════════════════════════════════════════

def run_parallel(runs, n_workers, dry_run=False):
    total = len(runs)

    if dry_run:
        print(f"\nDRY RUN — would launch {total} runs with {n_workers} workers.\n")
        for i, run in enumerate(runs, 1):
            E_MeV = run["energy_eV"] / 1e6
            print(f"  [{i:>4d}/{total}] {run_key(run)}  "
                  f"E={E_MeV:.0f} MeV  L={run['x_initial']:.2f}  "
                  f"pitch={run['pitch_deg']:.0f}°  "
                  f"gyros={run['gyroperiods']:.0e}")
        return 0, 0

    progress = load_progress()
    progress["started"] = datetime.now().isoformat()
    save_progress(progress)

    print(f"\nLaunching {n_workers} workers for {total} runs...\n")
    n_done = n_fail = 0
    completed = 0

    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futures = {ex.submit(execute_single_run, r): r for r in runs}
        try:
            for fut in as_completed(futures):
                key, status, elapsed, err = fut.result()
                completed += 1
                tag = "ok " if status == "completed" else "FAIL"
                print(f"  [{completed:>4d}/{total}] {tag} {key}  {elapsed:5.1f} min")

                if status == "completed":
                    n_done += 1
                    if key not in progress["completed"]:
                        progress["completed"].append(key)
                    # A success supersedes any prior failure: drop it from
                    # the failed list so the two lists stay mutually exclusive.
                    if key in progress["failed"]:
                        progress["failed"].remove(key)
                else:
                    n_fail += 1
                    if err:
                        for line in err.split("\n")[-3:]:
                            print(f"          {line}")
                    if key not in progress["failed"]:
                        progress["failed"].append(key)
                    # A failure supersedes any prior success: drop it from
                    # the completed list so --resume (which only checks
                    # 'completed') will re-attempt it instead of skipping.
                    if key in progress["completed"]:
                        progress["completed"].remove(key)

                if completed % 10 == 0:
                    save_progress(progress)

        except KeyboardInterrupt:
            print(f"\n{'─'*65}")
            print(f"  Interrupted! Completed: {n_done}  Failed: {n_fail}  "
                  f"Remaining: {total - completed}")
            print(f"  Progress saved. Resume with --resume")
            ex.shutdown(wait=False, cancel_futures=True)
            save_progress(progress)
            sys.exit(0)

    save_progress(progress)
    return n_done, n_fail


# ═══════════════════════════════════════════════════════════════════════════════
#  CSV consolidation
# ═══════════════════════════════════════════════════════════════════════════════

def consolidate_csv_logs(group):
    """
    Merge per-cell master_simulation_log.csv into one at the group level.

    Takes the UNION of columns across all per-cell files (so a schema change
    in one cell — e.g. adding eps_median — doesn't truncate older cells' data).

    Detects and warns about schema drift: when per-cell CSVs disagree on
    column sets it means some cells were written by a different writer
    version, which is worth knowing about.  The merge still works correctly
    in that case (missing columns become NaN).
    """
    import glob
    import pandas as pd

    group_dir = os.path.join(PROJECT_ROOT, "data", "dipoleb", group)
    if not os.path.isdir(group_dir):
        print(f"  No {group}/ directory — nothing to consolidate.")
        return

    pattern = os.path.join(group_dir, "*", "master_simulation_log.csv")
    csv_files = sorted(glob.glob(pattern))

    if not csv_files:
        print(f"  No per-cell CSVs found under {group}/.")
        return

    frames = []
    schemas = {}    # frozenset(columns) -> count of files with that schema
    for f in csv_files:
        try:
            # run_id is a hex hash (e.g. "2755e9", "052699"); keep it a string
            # so pandas doesn't coerce numeric-looking hashes to float/int.
            df_one = pd.read_csv(f, dtype={"run_id": str})
            schemas[frozenset(df_one.columns)] = schemas.get(frozenset(df_one.columns), 0) + 1
            frames.append(df_one)
        except Exception as e:
            print(f"  Warning: could not read {f}: {e}")

    if not frames:
        return

    # Schema-drift warning — alerts the user when per-cell CSVs have different
    # column sets, usually meaning some cells were processed with a newer or
    # older writer version.  The pd.concat below still merges them correctly
    # via outer join (union of columns, NaN for missing), but knowing about
    # the drift helps debug later inconsistencies.
    if len(schemas) > 1:
        union = set().union(*schemas.keys())
        print(f"  WARNING: {len(schemas)} different per-cell schemas detected:")
        for cols, count in sorted(schemas.items(), key=lambda kv: -kv[1]):
            missing = sorted(union - set(cols))
            print(f"    [{count:4d} files]  missing columns: {missing if missing else 'none'}")
        print(f"  Taking column union; rows missing a column will have NaN there.")

    df = pd.concat(frames, ignore_index=True)
    dup_keys = ["energy_eV", "L_eff", "phi_deg", "pitch_deg", "particle", "method", "steps"]
    existing_cols = [k for k in dup_keys if k in df.columns]
    df = df.drop_duplicates(subset=existing_cols, keep="last")

    master_path = os.path.join(group_dir, "master_simulation_log.csv")
    df.to_csv(master_path, index=False)
    print(f"  Consolidated {len(csv_files)} per-cell CSVs → {master_path}  "
          f"({len(df)} rows × {len(df.columns)} cols)")


# ═══════════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    _default_workers = max(1, (os.cpu_count() or 2) - 1)

    parser = argparse.ArgumentParser(
        description="Dwell map batch runner — fine L-shell sweeps for "
                    "meridional dwell-occupancy maps from orbit integrations.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Presets:
  10mev, 15mev, 20mev, 30mev, 50mev, 60mev, 80mev,
  100mev, 200mev, 400mev                       — single-energy sweeps
  low    = 10 + 15 + 20 + 30 MeV
  medium = 50 + 60 + 80 MeV
  high   = 100 + 200 + 400 MeV
  all    = all ten energies

Examples:
  %(prog)s --list-presets                          Show preset details
  %(prog)s 10mev --dry-run                         Preview the 10 MeV sweep
  %(prog)s 10mev                                   Run the 10 MeV sweep
  %(prog)s all --resume                            Run all energies, skip completed
  %(prog)s --energy 4e7 --L-start 2.0 --L-stop 5.0 --L-step 0.02 --group fluxmap_40mev

After runs finish, build the map:
  python scripts/plots/flux_map_builder.py --group fluxmap_10mev
        """)

    parser.add_argument("preset", nargs="?", default=None,
                        help="Preset name (see --list-presets). "
                             "Omit for custom scan with --energy etc.")
    parser.add_argument("--list-presets", action="store_true", dest="list_presets",
                        help="Show all available presets and exit.")

    # Custom scan overrides
    parser.add_argument("--energy", type=float, default=None,
                        help="Energy in eV for custom scan (e.g. 4e7 = 40 MeV).")
    parser.add_argument("--L-start", type=float, default=1.0, dest="L_start",
                        help="Starting L-shell (default: 1.0).")
    parser.add_argument("--L-stop", type=float, default=7.0, dest="L_stop",
                        help="Ending L-shell, inclusive (default: 7.0).")
    parser.add_argument("--L-step", type=float, default=0.05, dest="L_step",
                        help="L-shell increment (default: 0.05).")
    parser.add_argument("--pitch", type=float, default=89.0,
                        help="Pitch angle in degrees (default: 89.0).")
    parser.add_argument("--gyroperiods", type=float, default=1e5,
                        help="Gyroperiods per run (default: 1e5).")
    parser.add_argument("--group", type=str, default=None,
                        help="Output group name (default: fluxmap_<preset> or fluxmap_custom).")

    # Solver
    parser.add_argument("--adaptive", action="store_true",
                        help="Use adaptive PS stepping (default: fixed-step).")

    # Execution
    parser.add_argument("--workers", type=int, default=_default_workers,
                        help=f"Parallel workers (default: {_default_workers}).")
    parser.add_argument("--dry-run", action="store_true", dest="dry_run",
                        help="Print run plan without launching dipoleb.")
    parser.add_argument("--resume", action="store_true",
                        help="Skip runs already in batch_progress_fluxmap.json.")
    parser.add_argument("--consolidate-only", action="store_true", dest="consolidate_only",
                        help="Just rebuild the group's master CSV without running.")
    parser.add_argument("--rerun-chaotic", action="store_true", dest="rerun_chaotic",
                        help="Only rerun orbits classified as CHAOTIC in the group's "
                             "master CSV.  Pair with --adaptive to upgrade accuracy.")

    args = parser.parse_args()

    # ── List presets ─────────────────────────────────────────────────
    if args.list_presets:
        print("\nSingle-energy presets:")
        print(f"  {'Name':<10s}  {'Energy':<12s}  {'L range':<16s}  {'dL':<6s}  {'Pitch':<7s}  Gyros")
        print(f"  {'─'*10}  {'─'*12}  {'─'*16}  {'─'*6}  {'─'*7}  {'─'*8}")
        for name, p in PRESETS.items():
            E_str = f"{p['energy_eV']/1e6:.0f} MeV"
            L_str = f"{p['L_start']:.1f} – {p['L_stop']:.1f}"
            n_cells = len(_L_grid(p["L_start"], p["L_stop"], p["L_step"]))
            print(f"  {name:<10s}  {E_str:<12s}  {L_str:<16s}  {p['L_step']:<6.2f}  "
                  f"{p['pitch_deg']:<7.0f}  {p['gyroperiods']:.0e}  ({n_cells} cells)")
        print(f"\nMeta-presets:")
        for name, subs in META_PRESETS.items():
            total = sum(len(_L_grid(PRESETS[s]["L_start"], PRESETS[s]["L_stop"],
                                     PRESETS[s]["L_step"])) for s in subs)
            print(f"  {name:<10s}  {', '.join(subs)}  ({total} cells total)")
        return

    # ── Consolidate-only ─────────────────────────────────────────────
    if args.consolidate_only:
        group = args.group
        if not group and args.preset:
            group = f"fluxmap_{args.preset}"
        if not group:
            raise SystemExit("--consolidate-only requires --group or a preset name.")
        print(f"Consolidating {group}/ ...")
        consolidate_csv_logs(group)
        return

    # ── Build run list ───────────────────────────────────────────────
    all_runs = []

    if args.rerun_chaotic:
        # Chaotic-rerun mode: read master CSV instead of building from presets
        group = args.group
        if not group and args.preset:
            group = f"fluxmap_{args.preset}"
        if not group:
            raise SystemExit("--rerun-chaotic requires --group or a preset name.")
        print(f"Scanning {group}/master_simulation_log.csv for chaotic orbits...")
        all_runs = build_chaotic_rerun_list(group)

    elif args.preset:
        pairs = expand_preset(args.preset)
        for pname, pdict in pairs:
            group = args.group or f"fluxmap_{args.preset}"
            # Allow CLI overrides on top of presets
            p = copy.deepcopy(pdict)
            if args.energy is not None:
                p["energy_eV"] = args.energy
            if args.L_start != 1.0:
                p["L_start"] = args.L_start
            if args.L_stop != 7.0:
                p["L_stop"] = args.L_stop
            if args.L_step != 0.05:
                p["L_step"] = args.L_step
            if args.pitch != 89.0:
                p["pitch_deg"] = args.pitch
            if args.gyroperiods != 1e5:
                p["gyroperiods"] = args.gyroperiods

            runs = build_runs_from_preset(p, group)
            all_runs.extend(runs)

    elif args.energy is not None:
        group = args.group or "fluxmap_custom"
        all_runs = build_runs_custom(
            args.energy, args.L_start, args.L_stop, args.L_step,
            args.pitch, args.gyroperiods, group)
    else:
        raise SystemExit("Specify a preset name or --energy for a custom scan. "
                         "See --list-presets.")

    # ── Stamp adaptive flag onto every run ──────────────────────────
    if args.adaptive:
        for r in all_runs:
            r["use_adaptive"] = True
            # Wipe any stale fixed-step h5 so the adaptive result is the
            # only one in the folder (otherwise the dwell map builder may pick
            # the old fixed-step h5 depending on glob order).
            r["clean_before_run"] = True

    # ── Determine group (for summary / consolidation) ────────────────
    group = all_runs[0]["group"] if all_runs else "fluxmap"

    # ── Resume filtering ─────────────────────────────────────────────
    if args.rerun_chaotic:
        # Chaotic reruns: remove their keys from "completed" so --resume
        # won't skip them, and so the NEW adaptive result replaces the old.
        progress = load_progress()
        rerun_keys = {run_key(r) for r in all_runs}
        before_len = len(progress["completed"])
        progress["completed"] = [k for k in progress["completed"]
                                 if k not in rerun_keys]
        removed = before_len - len(progress["completed"])
        if removed:
            save_progress(progress)
            print(f"Cleared {removed} chaotic keys from progress (will be re-completed).\n")

    if args.resume:
        progress = load_progress()
        before = len(all_runs)
        all_runs = [r for r in all_runs if run_key(r) not in progress["completed"]]
        skipped = before - len(all_runs)
        if skipped:
            print(f"Resuming: skipping {skipped} already-completed cells.\n")

    if not all_runs:
        print("Nothing to do — all cells already completed (or empty preset).")
        return

    # ── Collect summary info ─────────────────────────────────────────
    energies = sorted(set(r["energy_eV"] for r in all_runs))
    L_vals   = sorted(set(r["x_initial"] for r in all_runs))

    print(f"{'='*65}")
    print(f"  DWELL MAP SWEEP   group: {group}")
    print(f"  Output: data/dipoleb/{group}/")
    print(f"  Total runs: {len(all_runs)}   Workers: {args.workers}")
    print(f"  Energies: {[f'{e/1e6:.0f} MeV' for e in energies]}")
    print(f"  L range:  {L_vals[0]:.2f} – {L_vals[-1]:.2f}  ({len(L_vals)} values)")
    print(f"  Pitch:    {all_runs[0]['pitch_deg']:.0f}°")
    print(f"  Gyros:    {all_runs[0]['gyroperiods']:.0e}")
    print(f"  Stepping: {'adaptive' if args.adaptive else 'fixed-step'}")
    print(f"{'='*65}")

    os.makedirs(BATCH_TMP_DIR, exist_ok=True)

    # ── Execute ──────────────────────────────────────────────────────
    n_done = n_fail = 0
    try:
        n_done, n_fail = run_parallel(all_runs, args.workers, dry_run=args.dry_run)
    finally:
        if not args.dry_run:
            if os.path.isdir(BATCH_TMP_DIR) and not os.listdir(BATCH_TMP_DIR):
                os.rmdir(BATCH_TMP_DIR)
            print("\nConsolidating per-cell CSVs...")
            try:
                consolidate_csv_logs(group)
            except Exception as e:
                print(f"  Consolidation failed: {e}")
                print(f"  Retry: {sys.argv[0]} --consolidate-only --group {group}")

    if args.dry_run:
        return

    print(f"\n{'='*65}")
    print(f"  DWELL MAP BATCH COMPLETE — {group}")
    print(f"  Completed: {n_done}  Failed: {n_fail}")
    print(f"  Master CSV: data/dipoleb/{group}/master_simulation_log.csv")
    print(f"\n  Next step — build the dwell map:")
    print(f"    python scripts/plots/flux_map_builder.py --group {group}")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
