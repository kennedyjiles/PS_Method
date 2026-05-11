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
    python scripts/batch_runner_protons.py                # phase 1, n_cores-1 workers
    python scripts/batch_runner_protons.py --workers 4    # cap workers
    python scripts/batch_runner_protons.py --dry-run      # preview without running
    python scripts/batch_runner_protons.py --resume       # skip already-completed runs
    python scripts/batch_runner_protons.py --adaptive     # use adaptive PS stepping

Output always lands in data/dipoleb/proton/ (DEFAULT_GROUPS in this script).
"""

import os
import sys
import copy
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


# ─── Phase grid for proton method-comparison table (PS / RK4 / RK45 / RKG) ──
# All cells at L=5; varies (energy, pitch). Each cell runs all 4 solvers,
# producing one CSV row per solver (4 rows × 8 cells = 32 method-rows).
# Phases 2–4 are reserved for future extensions (additional L-shells,
# more energies, etc.) — leave empty for now.

# Phase 1 — 4 energies × 2 pitches at L=5 = 8 cells
_PHASE_1_CELLS = [
    (1e4, 5),   # 10 keV   proton @ L=5
    (1e5, 5),   # 100 keV
    (1e6, 5),   # 1 MeV
    (1e7, 5),   # 10 MeV
]
_PHASE_2_CELLS = []
_PHASE_3_CELLS = []
_PHASE_4_CELLS = []

CELLS = {
    1: _PHASE_1_CELLS,
    2: _PHASE_2_CELLS,
    3: _PHASE_3_CELLS,
    4: _PHASE_4_CELLS,
}

GYROPERIODS = {
    1: None,
    2: None,
    3: None,
    4: None,
}

STEPS_PER_GYRO_PS = {
    1: 65,
    2: 65,
    3: 65,
    4: 65,
}

# Pitch angles (degrees) — every (E, L) cell is run at each pitch in this list.
# Phase 1 has 4 cells × 2 pitches = 8 runs.
PITCH_ANGLES = [90.0, 30.0]

# Particle per phase
PARTICLE = {
    1: "proton",
    2: "proton",
    3: "proton",
    4: "proton",
}

# Default batch group names per phase — all four phases share one group so
# the consolidated CSV holds the full table.
DEFAULT_GROUPS = {
    1: "proton",
    2: "proton",
    3: "proton",
    4: "proton",
}

# Per-phase overrides applied on top of base.yml. Sweep params (energy, L,
# pitch, gyroperiods, particle, read_data, solvers.adaptive, plus phi_deg
# when the run dict carries it, plus steps_per_gyro.ps) are applied AFTER
# these and win on conflict — so use OVERRIDES for the constants you want
# to vary between tables (user_min_phase, phi_deg, ps_chunk_steps, plotting
# toggles, etc.), not for sweep axes. Nested dicts are deep-merged.
#
# solvers override enables RK4 / RK45 / RKG alongside PS for the four-method
# comparison rows in the table. (base.yml has rk4/rk45/rkg false.)
OVERRIDES = {
    1: {"gyroperiods": None, "total_steps": 1.0e7,
        "solvers": {"rk4": True, "rk45": True, "rkg": True},
    },
    2: {"phi_deg": 0.0, "use_gyroradius_L_correction": False, "user_min_phase": 0.00001},
    3: {"phi_deg": 0.0, "use_gyroradius_L_correction": False, "user_min_phase": 0.000001},
    4: {"phi_deg": 0.0, "use_gyroradius_L_correction": False, "user_min_phase": 0.0000001},
}


# Per-cell solver skips. Use to suppress a solver that hangs / fails / takes
# unbounded time for a specific (energy_eV, L, pitch_deg) cell. The disable
# wins over OVERRIDES[phase].solvers, so the listed solvers will be False
# for that cell only — every other cell still runs them.
#
# Key:   (energy_eV, x_initial, pitch_deg)
# Value: dict of solver flags to force False (True wouldn't make sense here)
SKIP_SOLVERS = {
    (1e4, 5, 30.0): {"rk45": False},   # RK45 cannot resolve 10 keV proton at 30° pitch (runs indefinitely)
}


def _deep_merge(base, override):
    """In-place deep merge of override into base; override wins on conflict.
    Mirrors configs.config_loader._deep_merge so batch_runner stays
    standalone (no cross-package import)."""
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v


def build_run_list(phase):
    """Generate list of run dicts for a given phase.

    Each phase carries an explicit list of (energy_eV, L) cells in CELLS,
    and uniform GYROPERIODS / STEPS_PER_GYRO_PS for the whole tier. Pitch
    is taken from the global PITCH_ANGLES.

    Per-cell solver skips from SKIP_SOLVERS (keyed by (E, L, pitch)) are
    attached to the run dict so write_config can apply them.
    """
    runs = []
    gyros          = GYROPERIODS[phase]
    steps_per_gyro = STEPS_PER_GYRO_PS.get(phase)        # None = use base.yml default
    for energy, L in CELLS[phase]:
        for pitch in PITCH_ANGLES:
            run = {
                "energy_eV":  energy,
                "x_initial":  round(L, 2),
                "pitch_deg":  pitch,
                "phase":      phase,
            }
            # Only include gyroperiods if set. None means the phase is
            # driven by total_steps (set via OVERRIDES) and we shouldn't
            # write a gyroperiods value to the per-worker yml.
            if gyros is not None:
                run["gyroperiods"] = gyros
            if steps_per_gyro is not None:
                run["steps_per_gyro_ps"] = steps_per_gyro
            skip_key = (energy, L, pitch)
            if skip_key in SKIP_SOLVERS:
                run["solver_skips"] = SKIP_SOLVERS[skip_key]
            runs.append(run)
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
    Phase-keyed OVERRIDES are deep-merged in first; sweep params win on
    conflict so the table axes (energy/L/pitch/phi/gyroperiods) are never
    masked by an OVERRIDES entry.
    """
    # Start from a deep copy of the phase's overrides (could be empty)
    cfg = copy.deepcopy(OVERRIDES.get(run["phase"], {}))

    # sweep solvers — adaptive (always set) plus any per-cell skips. Skips
    # land here so the deep_merge below makes them WIN over OVERRIDES[phase],
    # cleanly disabling the listed solvers for this one cell.
    sweep_solvers = {"adaptive": run.get("use_adaptive", False)}
    if "solver_skips" in run:
        sweep_solvers.update(run["solver_skips"])

    sweep = {
        "base_config": os.path.abspath(BASE_YML),
        "batch_group": group,
        "energy_eV":   float(run["energy_eV"]),
        "x_initial":   float(run["x_initial"]),
        "pitch_deg":   float(run["pitch_deg"]),
        "particle":    PARTICLE.get(run["phase"], "proton"),
        "read_data":   False,
        "solvers":     sweep_solvers,
    }
    # Only include gyroperiods in sweep when it's actually set per-run;
    # otherwise let OVERRIDES (e.g. {"gyroperiods": null, "total_steps": N})
    # or base.yml decide. Same pattern as phi_deg below.
    if "gyroperiods" in run:
        sweep["gyroperiods"] = float(run["gyroperiods"])
    # Only include phi_deg in sweep if it's actually a per-run value;
    # otherwise let OVERRIDES (or base.yml) decide.
    if "phi_deg" in run:
        sweep["phi_deg"] = float(run["phi_deg"])
    # Phase 4 sets per-cell steps_per_gyro_ps; otherwise base.yml default applies.
    if run.get("steps_per_gyro_ps") is not None:
        sweep["steps_per_gyro"] = {"ps": int(run["steps_per_gyro_ps"])}
    _deep_merge(cfg, sweep)

    with open(config_path, "w") as f:
        f.write(f"# Auto-generated batch config -- merged with base.yml at load time\n")
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)


# Module-level variable set by main() so execute_single_run can access it
_batch_group = "proton"


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
  %(prog)s --dry-run                          Preview Phase 1 (default)
  %(prog)s                                    Run Phase 1 with n_cores-1 workers
  %(prog)s --workers 4                        Cap to 4 workers
  %(prog)s --adaptive                         Use adaptive PS stepping
  %(prog)s --resume                           Skip cells already completed
  %(prog)s --single E1e+04_L5.00_P30.0        Run one specific cell
        """)
    parser.add_argument("--phase", type=int, default=1, choices=[1, 2, 3, 4],
                        help="Which proton sweep phase to run (default: 1). "
                             "Phase 1 = the method-comparison table "
                             "(4 energies × 2 pitches at L=5 = 8 cells). "
                             "Phases 2-4 reserved for future extensions. "
                             "All phases share group 'proton'.")
    # NOTE: --group intentionally removed. Output group is fixed by
    # DEFAULT_GROUPS[phase] in this runner so proton runs always land in
    # data/dipoleb/proton/ (not the walt folder, which the walt runner owns).
    # Default to (n_cores - 1) so the machine stays responsive without
    # requiring the user to look up their core count.
    _default_workers = max(1, (os.cpu_count() or 2) - 1)
    parser.add_argument("--workers", type=int, default=_default_workers,
                        help=f"Number of parallel workers "
                             f"(default: n_cores-1 = {_default_workers} on this machine). "
                             f"Pass 1 to disable parallelism.")
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

    # ── Resolve group name (always from DEFAULT_GROUPS — no CLI override) ──
    group = DEFAULT_GROUPS[args.phase]
    _batch_group = group
    print(f"Output destination: data/dipoleb/{group}/")

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
    print(f"  Energies (eV): {[f'{e:.0e}' for e in sorted(set(r['energy_eV'] for r in runs))]}")
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
