#!/usr/bin/env python3
"""
Batch Runner
=============
Orchestrates a parameter sweep of dipoleb.py across (energy, L-shell, pitch angle)
and consolidates results into a single master CSV per batch group.

Supports parallel execution on multi-core machines (e.g. M4 Max with 16 cores).
Each worker gets its own minimal YAML config file (merged with base.yml at load time)
to avoid race conditions.

This script does NOT modify dipoleb.py or any core library files. It writes a
minimal YAML config per run, then calls `python dipoleb.py <path/to/config.yml>`.
dipoleb.py's __main__ block detects the .yml extension and passes it to load_config(),
which auto-merges with base.yml via the `base_config` key.

Output layout:
    data/dipoleb/<group>/<run_key>/           per-run output (plots, h5, etc.)
    data/dipoleb/<group>/master_simulation_log.csv   consolidated CSV for the batch

Usage:
    python scripts/batch_runner.py --phase 1 --workers 10
    python scripts/batch_runner.py --phase 1 --workers 10 --adaptive
    python scripts/batch_runner.py --phase 1 --workers 10 --dry-run
    python scripts/batch_runner.py --phase 1 --workers 10 --resume
    python scripts/batch_runner.py --phase 1 --group my_sweep --workers 10

"""

import os
import sys
import json
import argparse
import subprocess
import time
import yaml
import pandas as pd
import numpy as np
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

# ─── Paths ───────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
PROGRESS_PATH = os.path.join(SCRIPT_DIR, "batch_progress.json")
BATCH_TMP_DIR = os.path.join(SCRIPT_DIR, "batch_tmp")
BASE_YML = os.path.join(PROJECT_ROOT, "configs", "dipoleb", "base.yml")


ENERGIES_EV = {
    1: [5e8],
    2: [7e8, 1e9, 3e9, 5e9, 7e9, 1e10],
    3: [1e8],
}

# L-shells (equatorial launch distance in R_E)
# L_FINE   = list(np.arange(1.0, 4.1, 0.5))
L_FINE  = list(np.arange(3.51, 3.6, 0.01))
L_INNER   = [4.5, 5.0]

L_SHELLS = {
    1: L_FINE,
    2: L_FINE,
    3: L_INNER,
}

# Pitch angles (degrees)
PITCH_ANGLES = [89.0]

# Simulation duration: number of characteristic gyroperiods
GYROPERIODS = {
    1: 1e5,
    2: 1e5,
    3: 1e5,
}

# Default batch group names per phase
DEFAULT_GROUPS = {
    1: "flux_map",
    2: "flux_map",
    3: "flux_map",
}


def build_run_list(phase):
    """Generate list of run dicts for a given phase."""
    runs = []
    for energy in ENERGIES_EV[phase]:
        for L in L_SHELLS[phase]:
            for pitch in PITCH_ANGLES:
                runs.append({
                    "energy_eV": energy,
                    "x_initial": round(L, 2),
                    "pitch_deg": pitch,
                    "gyroperiods": GYROPERIODS[phase],
                    "phase": phase,
                })
    return runs


def build_chaotic_rerun_list(group):
    """
    Read the consolidated master CSV for a batch group, find runs where
    orbit_character is 'chaotic', and build a run list to re-run them
    with adaptive stepping.
    """
    csv_path = os.path.join(PROJECT_ROOT, "data", "dipoleb", group, "master_simulation_log.csv")
    if not os.path.exists(csv_path):
        print(f"  No consolidated CSV at {csv_path}")
        return []

    df = pd.read_csv(csv_path)
    if "orbit_character" not in df.columns:
        print("  No orbit_character column in CSV.")
        return []

    mask = df["orbit_character"].fillna("").str.upper() == "CHAOTIC"
    chaotic = df[mask]
    if chaotic.empty:
        return []

    # Deduplicate by physics parameters (same run may have multiple solver rows)
    param_cols = ["energy_eV", "x", "pitch_deg", "phi_deg"]
    existing = [c for c in param_cols if c in chaotic.columns]
    chaotic = chaotic.drop_duplicates(subset=existing)

    runs = []
    for _, row in chaotic.iterrows():
        runs.append({
            "energy_eV": float(row["energy_eV"]),
            "x_initial": float(row["x"]),
            "pitch_deg": float(row["pitch_deg"]),
            "phi_deg":   float(row["phi_deg"]) if "phi_deg" in row and pd.notna(row["phi_deg"]) else 0.0,
            "gyroperiods": float(row["gyroperiods"]),
            "phase": "rerun",
            "use_adaptive": True,
        })

    print(f"  Found {len(runs)} chaotic runs to re-run with adaptive stepping.")
    return runs


def run_key(run):
    """Unique string key for a run, used for progress tracking."""
    key = f"E{run['energy_eV']:.0e}_L{run['x_initial']:.2f}_P{run['pitch_deg']:.1f}"
    if run.get("phi_deg", 0.0) != 0.0:
        key += f"_phi{run['phi_deg']:.1f}"
    return key


def load_progress():
    if os.path.exists(PROGRESS_PATH):
        with open(PROGRESS_PATH, "r") as f:
            return json.load(f)
    return {"completed": [], "failed": [], "started": None}


def save_progress(progress):
    with open(PROGRESS_PATH, "w") as f:
        json.dump(progress, f, indent=2)



def write_config(run, config_path, group):
    """
    Write a minimal per-worker YAML config file.

    The YAML contains only the parameters that differ from base.yml, plus a
    base_config key so load_config() can find and merge with the base defaults.
    """
    cfg = {
        "base_config": os.path.abspath(BASE_YML),
        "batch_group": group,
        "energy_eV":   float(run["energy_eV"]),
        "x_initial":   float(run["x_initial"]),
        "pitch_deg":   float(run["pitch_deg"]),
        "phi_deg":     float(run.get("phi_deg", 0.0)),
        "gyroperiods": float(run["gyroperiods"]),
        "particle":    "proton",
        "read_data":   False,
        "solvers": {
            "adaptive": run.get("use_adaptive", False),
        },
    }

    with open(config_path, "w") as f:
        f.write(f"# Auto-generated batch config -- merged with base.yml at load time\n")
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)


# Module-level variable set by main() so execute_single_run can access it
_batch_group = "flux_map"


def execute_single_run(run):
    """
    Execute one dipoleb batch run.
    Called by worker processes -- must be a top-level function for pickling.
    Returns (run_key, status, elapsed_min, error_msg).
    """
    key = run_key(run)

    config_path = os.path.join(BATCH_TMP_DIR, f"{key}.yml")
    write_config(run, config_path, _batch_group)

    start = time.time()
    try:
        result = subprocess.run(
            [sys.executable, "dipoleb.py", config_path],
            cwd=PROJECT_ROOT,
            timeout=3600 * 24,   # 24-hour timeout
            capture_output=True,
            text=True,
        )
        elapsed = (time.time() - start) / 60

        # Clean up config file
        if os.path.exists(config_path):
            os.remove(config_path)

        if result.returncode == 0:
            return (key, "completed", elapsed, None)
        else:
            err = "\n".join(result.stderr.strip().split("\n")[-5:])
            return (key, "failed", elapsed, err)

    except subprocess.TimeoutExpired:
        elapsed = (time.time() - start) / 60
        if os.path.exists(config_path):
            os.remove(config_path)
        return (key, "timeout", elapsed, "24-hour timeout exceeded")

    except Exception as e:
        elapsed = (time.time() - start) / 60
        if os.path.exists(config_path):
            os.remove(config_path)
        return (key, "failed", elapsed, str(e))


def run_parallel(runs, max_workers, dry_run=False):
    """Execute runs in parallel using ProcessPoolExecutor."""
    progress = load_progress()
    n_done = 0
    n_fail = 0
    total = len(runs)

    print(f"Launching {max_workers} parallel workers for {total} runs...\n")

    if dry_run:
        for i, run in enumerate(runs):
            key = run_key(run)
            E_MeV = run["energy_eV"] / 1e6
            print(f"  [{i+1:>4d}/{total}] {key}  "
                  f"E={E_MeV:.0f} MeV  L={run['x_initial']:.2f}  "
                  f"alpha={run['pitch_deg']:.0f} deg")
        return

    # Submit all runs to the pool
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_run = {}
        for run in runs:
            future = executor.submit(execute_single_run, run)
            future_to_run[future] = run

        try:
            for future in as_completed(future_to_run):
                run = future_to_run[future]
                key, status, elapsed, err = future.result()

                n_done += 1 if status == "completed" else 0
                n_fail += 1 if status != "completed" else 0

                completed_total = n_done + n_fail

                if status == "completed":
                    print(f"  [{completed_total:>4d}/{total}] ok {key}  "
                          f"{elapsed:.1f} min")
                    progress["completed"].append(key)
                else:
                    print(f"  [{completed_total:>4d}/{total}] FAIL {key}  "
                          f"{status} after {elapsed:.1f} min")
                    if err:
                        for line in err.split("\n")[-3:]:
                            print(f"           {line}")
                    progress["failed"].append(key)

                # Save progress periodically (every 10 completions)
                if completed_total % 10 == 0:
                    save_progress(progress)

        except KeyboardInterrupt:
            print(f"\n{'─'*65}")
            print(f"  Interrupted! Cancelling remaining runs...")
            executor.shutdown(wait=False, cancel_futures=True)
            print(f"  Completed: {n_done}  Failed: {n_fail}  "
                  f"Remaining: {total - n_done - n_fail}")
            print(f"  Progress saved. Resume with --resume")
            save_progress(progress)
            sys.exit(0)

    save_progress(progress)
    return n_done, n_fail


def run_sequential(runs, dry_run=False):
    """Execute runs one at a time (original behavior)."""
    progress = load_progress()
    n_done = 0
    n_fail = 0
    total = len(runs)

    try:
        for i, run in enumerate(runs):
            key = run_key(run)
            E_MeV = run["energy_eV"] / 1e6
            print(f"[{i+1:>4d}/{total}] {key}  "
                  f"E={E_MeV:.0f} MeV  L={run['x_initial']:.2f}  "
                  f"alpha={run['pitch_deg']:.0f} deg")

            if dry_run:
                continue

            key, status, elapsed, err = execute_single_run(run)

            if status == "completed":
                print(f"         ok completed in {elapsed:.1f} min")
                progress["completed"].append(key)
                n_done += 1
            else:
                print(f"         FAIL {status} after {elapsed:.1f} min")
                if err:
                    for line in err.split("\n")[-3:]:
                        print(f"           {line}")
                progress["failed"].append(key)
                n_fail += 1

            save_progress(progress)

    except KeyboardInterrupt:
        print(f"\n{'─'*65}")
        print(f"  Interrupted. Completed: {n_done}  Failed: {n_fail}  "
              f"Remaining: {total - n_done - n_fail}")
        print(f"  Resume with --resume")
        save_progress(progress)
        sys.exit(0)

    save_progress(progress)
    return n_done, n_fail


def main():
    global _batch_group

    parser = argparse.ArgumentParser(
        description="Batch parameter sweep runner for dipoleb.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --phase 1 --dry-run                        Preview Phase 1
  %(prog)s --phase 1 --workers 10                     Phase 1, fixed-step, 10 cores
  %(prog)s --phase 2 --workers 8 --adaptive           Phase 2, adaptive stepping
  %(prog)s --phase 1 --workers 10 --resume            Resume interrupted Phase 1
  %(prog)s --single E1e+07_L2.00_P89.0                Run one specific case
  %(prog)s --phase 1 --group my_sweep --workers 10    Custom group name
        """)
    parser.add_argument("--phase", type=int, default=1, choices=[1, 2, 3],
                        help="Which phase to run (default: 1)")
    parser.add_argument("--group", type=str, default=None,
                        help="Batch group name for output directory "
                             "(default: flux_map). All runs land in "
                             "data/dipoleb/<group>/.")
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of parallel workers (default: 1 = sequential). "
                             "Recommended: 8-10 for M4 Max, 4-6 for M1/M2.")
    parser.add_argument("--adaptive", action="store_true",
                        help="Use adaptive PS stepping (default: fixed-step).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print run plan without executing")
    parser.add_argument("--resume", action="store_true",
                        help="Skip runs already completed in previous session")
    parser.add_argument("--single", type=str, default=None,
                        help="Run a single case by key, e.g. 'E1e+07_L2.00_P89.0'")
    parser.add_argument("--rerun-chaotic", action="store_true",
                        help="Read the consolidated CSV, find runs flagged as chaotic, "
                             "and re-run them with adaptive stepping.")
    args = parser.parse_args()

    # ── Resolve group name ─────────────────────────────────────────
    group = args.group or DEFAULT_GROUPS[args.phase]
    _batch_group = group

    # ── Rerun-chaotic mode ─────────────────────────────────────────
    if args.rerun_chaotic:
        runs = build_chaotic_rerun_list(group)
        if not runs:
            print("No chaotic runs found in consolidated CSV. Nothing to re-run.")
            sys.exit(0)
        mode_label = f"RERUN CHAOTIC -- group: {group}"
        # Force adaptive for the re-runs
        for r in runs:
            r["use_adaptive"] = True
    else:
        # ── Normal build ───────────────────────────────────────────
        runs = build_run_list(args.phase)
        mode_label = f"BATCH SWEEP -- Phase {args.phase}"

    progress = load_progress() if args.resume else {"completed": [], "failed": [], "started": None}

    # Filter out completed runs
    if args.resume:
        before = len(runs)
        runs = [r for r in runs if run_key(r) not in progress["completed"]]
        skipped = before - len(runs)
        if skipped > 0:
            print(f"Resuming: skipping {skipped} already-completed runs.\n")

    # Single-run mode
    if args.single:
        all_runs = build_run_list(args.phase)
        runs = [r for r in all_runs if run_key(r) == args.single]
        if not runs:
            print(f"No match for '{args.single}'")
            sys.exit(1)

    if not runs:
        print("All runs already completed! Nothing to do.")
        sys.exit(0)

    # Summary
    stepping_label = "adaptive" if args.adaptive else "fixed-step"
    print(f"{'='*65}")
    print(f"  {mode_label}  [{stepping_label}]  group: {group}")
    print(f"  Runs: {len(runs)}   Workers: {args.workers}")
    print(f"  Energies: {sorted(set(r['energy_eV']/1e6 for r in runs))} MeV")
    print(f"  L-shells: {sorted(set(r['x_initial'] for r in runs))}")
    print(f"  Pitch angles: {sorted(set(r['pitch_deg'] for r in runs))}")
    print(f"{'='*65}")
    if args.dry_run:
        print("  *** DRY RUN -- no simulations will execute ***\n")
    print()

    # Create directories
    os.makedirs(BATCH_TMP_DIR, exist_ok=True)

    progress["started"] = datetime.now().isoformat()

    # Set use_adaptive flag for each run (passed through YAML config to dipoleb.py)
    if not args.rerun_chaotic:
        for r in runs:
            r["use_adaptive"] = args.adaptive

    # Execute
    if args.workers > 1:
        n_done, n_fail = run_parallel(runs, args.workers, dry_run=args.dry_run)
    else:
        n_done, n_fail = run_sequential(runs, dry_run=args.dry_run)

    if args.dry_run:
        return

    # Clean up temp directory
    if os.path.exists(BATCH_TMP_DIR) and not os.listdir(BATCH_TMP_DIR):
        os.rmdir(BATCH_TMP_DIR)

    # Consolidate per-run CSVs into one master at the group level
    print("\nConsolidating per-run CSVs...")
    consolidate_csv_logs(group)

    # Final summary
    print(f"\n{'='*65}")
    print(f"  BATCH COMPLETE -- Phase {args.phase}  group: {group}")
    print(f"  Completed: {n_done}  Failed: {n_fail}")
    if n_fail > 0:
        print(f"  Failed runs logged in {PROGRESS_PATH}")
    print(f"  Output: data/dipoleb/{group}/")
    print(f"{'='*65}")


def consolidate_csv_logs(group):
    """
    After all runs complete, collect the per-run master_simulation_log.csv files
    from each run folder and merge them into a single master CSV at the group
    level (e.g. data/dipoleb/flux_map/master_simulation_log.csv).

    Each dipoleb.py run writes its own CSV in its output_folder. This function
    gathers them, concatenates, deduplicates, and writes one combined file.
    """
    import pandas as pd
    import glob

    group_dir = os.path.join(PROJECT_ROOT, "data", "dipoleb", group)
    if not os.path.isdir(group_dir):
        print(f"  No {group}/ directory found -- nothing to consolidate.")
        return

    # Find all per-run CSVs
    pattern = os.path.join(group_dir, "*", "master_simulation_log.csv")
    csv_files = sorted(glob.glob(pattern))

    if not csv_files:
        print(f"  No per-run CSVs found in {group}/ -- nothing to consolidate.")
        return

    # Concatenate all per-run CSVs
    frames = []
    for f in csv_files:
        try:
            frames.append(pd.read_csv(f))
        except Exception as e:
            print(f"  Warning: could not read {f}: {e}")

    if not frames:
        return

    df = pd.concat(frames, ignore_index=True)

    # Deduplicate
    dup_keys = ["energy_eV", "L_eff", "phi_deg", "pitch_deg", "particle", "method", "steps"]
    existing_cols = [k for k in dup_keys if k in df.columns]
    df = df.drop_duplicates(subset=existing_cols, keep="last")

    # Write consolidated CSV at the group level
    master_path = os.path.join(group_dir, "master_simulation_log.csv")
    df.to_csv(master_path, index=False)
    print(f"  Consolidated {len(csv_files)} run CSVs -> {master_path} ({len(df)} rows)")


if __name__ == "__main__":
    main()
