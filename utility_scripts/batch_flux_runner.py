#!/usr/bin/env python3
"""
Batch Flux Map Runner
=====================
Orchestrates a parameter sweep of dipoleB_adp.py across (energy, L-shell, pitch angle)
to generate the trajectory data needed for an AP-8-comparable meridian-plane flux map.

Supports parallel execution on multi-core machines (e.g. M4 Max with 16 cores).
Each worker gets its own JSON config file to avoid race conditions.

This script does NOT modify dipoleB_adp.py or any core library files. It writes a
JSON config per run, then calls `python dipoleB_adp.py batch:/path/to/config.json`.
The "batch" run mode in dipoleB_testparticles.py reads from that config.

Usage:
    python utility_scripts/batch_flux_runner.py --phase 1 --workers 10              # fixed-step (default)
    python utility_scripts/batch_flux_runner.py --phase 3 --workers 8 --driver dipoleB_adp.py  # adaptive
    python utility_scripts/batch_flux_runner.py --phase 1 --workers 10 --dry-run
    python utility_scripts/batch_flux_runner.py --phase 1 --workers 10 --resume

    # Michel phase portrait sweep at L=1.5, 10 MeV
    python utility_scripts/batch_flux_runner.py --michel --L 1.5 --energy 10e6 --workers 10
    python utility_scripts/batch_flux_runner.py --michel --L 1.5 --energy 10e6 --phi-steps 12 --workers 10

Phases:
    1  — 10, 30 MeV  × 16 L-shells × 10 pitch angles  (belt core, fast)
    2  — 100 MeV     × 16 L-shells × 10 pitch angles  (energy falloff)
    3  — 300 MeV     × 10 L-shells × 10 pitch angles  (transition regime)
    4  — 1, 10 GeV   × 16 L-shells × 10 pitch angles  (Dragt regime, slow)

Michel mode:
    Sweeps pitch angle × gyrophase φ at a single (L, energy) to produce
    the data for a Michel (1971) α vs φ phase portrait.  Each initial φ
    fills out the island/chain structure; multiple pitch angles reveal
    Regions Ia, Ib, and II on one plot.

Author: Batch runner for H. Jiles thesis work
"""

import os
import sys
import json
import argparse
import subprocess
import time
import tempfile
import shutil
import numpy as np
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

# ─── Paths ───────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
PROGRESS_PATH = os.path.join(SCRIPT_DIR, "batch_flux_progress.json")
BATCH_TMP_DIR = os.path.join(SCRIPT_DIR, "batch_tmp")

# ─── Parameter Grids ─────────────────────────────────────────────────────────
# Energies in eV — matching AP-8 range (0.1 MeV to 400 MeV)
#
# Quick reference:  0.1 MeV = 100 keV = 1e5 eV
#                     1 MeV = 1e6 eV
#                    10 MeV = 1e7 eV
#                   100 MeV = 1e8 eV
#                   400 MeV = 4e8 eV
#                     1 GeV = 1e9 eV
#                    10 GeV = 1e10 eV
#
# AP-8 covers 0.1–400 MeV. We split into phases by physics regime:
#   Phase 1: 0.1–30 MeV   (deeply adiabatic, dipoleB.py)
#   Phase 2: 50–400 MeV   (transition regime, dipoleB.py or dipoleB_adp.py)
#   Phase 3: 1–10 GeV     (Dragt/Störmer, dipoleB_adp.py)
#
ENERGIES_EV = {
    1: [0.1e6, 0.3e6, 1e6, 3e6, 10e6, 30e6],   # 0.1–30 MeV  (AP-8 core)
    2: [50e6, 100e6, 200e6, 400e6],              # 50–400 MeV  (AP-8 upper end)
    3: [1e9, 10e9],                               # 1, 10 GeV   (beyond AP-8, Dragt regime)
}

# L-shells (equatorial launch distance in R_E)
# AP-8 flux extends to roughly L ≈ 3.5, faint edge near ρ ≈ 4
L_FINE   = list(np.arange(1.2, 3.6, 0.1))   # 1.2, 1.3, ..., 3.5  (24 values)
L_INNER  = list(np.arange(1.2, 3.1, 0.1))   # 1.2, ..., 3.0       (19 values, for GeV)

L_SHELLS = {
    1: L_FINE,
    2: L_FINE,
    3: L_INNER,     # GeV — only inner L (outer L lost to atmosphere)
}

# Pitch angles (degrees) — finer grid for better omnidirectional flux resolution
PITCH_ANGLES = [10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0,
                50.0, 55.0, 60.0, 65.0, 70.0, 75.0, 80.0, 85.0, 89.0]

# Simulation duration: number of characteristic gyroperiods
GYROPERIODS = {
    1: 5e4,   # 0.1-30 MeV: adiabatic, fast convergence
    2: 2e4,   # 50-400 MeV: fast bouncers, 2e4 gives hundreds of bounces
    3: 5e5,   # 1-10 GeV: chaotic, but many terminate early
}

# ─── Michel Phase Portrait Defaults ─────────────────────────────────────────
# Pitch angles for Michel sweep — denser near 90° where island structure lives
MICHEL_PITCHES = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 65.0, 70.0, 75.0,
                  80.0, 82.0, 84.0, 85.0, 86.0, 87.0, 88.0, 89.0]
MICHEL_GYROPERIODS = 1e4  # 100k gyroperiods — enough equatorial crossings for good portrait


def build_run_list(phase):
    """Generate list of run dicts for a given phase."""
    runs = []
    for energy in ENERGIES_EV[phase]:
        for L in L_SHELLS[phase]:
            for pitch in PITCH_ANGLES:
                runs.append({
                    "energy_eV": energy,
                    "L_shell": round(L, 2),
                    "pitch_deg": pitch,
                    "gyroperiods": GYROPERIODS[phase],
                    "phase": phase,
                })
    return runs


def build_michel_sweep(energy_eV, L_shell, phi_steps=12, pitches=None,
                       gyroperiods=None):
    """
    Generate run list for a Michel phase portrait at a single (L, energy).

    Sweeps pitch angles × gyrophase φ. The φ sweep fills out the island
    structure in the α-φ portrait — each initial φ produces a different
    trajectory that samples a different part of the phase space.

    Args:
        energy_eV:  particle energy in eV
        L_shell:    equatorial launch distance in R_E
        phi_steps:  number of initial gyrophase values (evenly spaced 0–360°)
        pitches:    list of pitch angles (degrees); defaults to MICHEL_PITCHES
        gyroperiods: simulation length; defaults to MICHEL_GYROPERIODS

    Returns:
        list of run dicts
    """
    if pitches is None:
        pitches = MICHEL_PITCHES
    if gyroperiods is None:
        gyroperiods = MICHEL_GYROPERIODS

    phi_values = np.linspace(0, 360, phi_steps, endpoint=False).tolist()

    runs = []
    for pitch in pitches:
        for phi in phi_values:
            runs.append({
                "energy_eV": energy_eV,
                "L_shell": round(L_shell, 2),
                "pitch_deg": pitch,
                "phi_deg": round(phi, 2),
                "gyroperiods": gyroperiods,
                "phase": "michel",
            })
    return runs


def run_key(run):
    """Unique string key for a run, used for progress tracking."""
    if run.get("phase") == "michel":
        return f"E{run['energy_eV']:.0e}_L{run['L_shell']:.2f}_P{run['pitch_deg']:.1f}_phi{run['phi_deg']:.1f}"
    return f"E{run['energy_eV']:.0e}_L{run['L_shell']:.2f}_P{run['pitch_deg']:.1f}"


def load_progress():
    if os.path.exists(PROGRESS_PATH):
        with open(PROGRESS_PATH, "r") as f:
            return json.load(f)
    return {"completed": [], "failed": [], "started": None}


def save_progress(progress):
    with open(PROGRESS_PATH, "w") as f:
        json.dump(progress, f, indent=2)


def estimate_time(run):
    """Rough estimate of run time in minutes."""
    E_MeV = run["energy_eV"] / 1e6
    pitch = run["pitch_deg"]
    L = run["L_shell"]

    if E_MeV <= 30:
        base_min = 3
    elif E_MeV <= 100:
        base_min = 15
    elif E_MeV <= 300:
        base_min = 60
    elif E_MeV <= 1000:
        base_min = 240
    else:
        base_min = 600

    if pitch >= 85:
        base_min *= 1.5
    elif pitch <= 30:
        base_min *= 0.7

    if L >= 4.0 and E_MeV >= 100:
        base_min *= 0.5

    return base_min


def write_config(run, config_path):
    """Write per-worker JSON config file."""
    is_michel = (run.get("phase") == "michel")
    config = {
        "energy_eV":   run["energy_eV"],
        "L_shell":     run["L_shell"],
        "pitch_deg":   run["pitch_deg"],
        "phi_deg":     run.get("phi_deg", 0.0),
        "gyroperiods": run["gyroperiods"],
        "output_folder": "outputs/michel" if is_michel else "outputs/flux_map",
        "run_storage":   "outputs/outputs_rawdata",
        "particle":      "Proton",
    }
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)


def execute_single_run(run):
    """
    Execute one dipoleB batch run.
    Called by worker processes — must be a top-level function for pickling.
    Returns (run_key, status, elapsed_min, error_msg).

    The run dict must include a "driver" key ("dipoleB.py" or "dipoleB_adp.py").
    """
    key = run_key(run)
    driver = run.get("driver", "dipoleB.py")

    # Each worker gets its own config file (PID-based to avoid collisions)
    config_path = os.path.join(BATCH_TMP_DIR, f"batch_config_{os.getpid()}_{key}.json")
    write_config(run, config_path)

    start = time.time()
    try:
        result = subprocess.run(
            [sys.executable, driver, f"batch:{config_path}"],
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
            est = estimate_time(run)
            print(f"  [{i+1:>4d}/{total}] {key}  "
                  f"E={E_MeV:.0f} MeV  L={run['L_shell']:.2f}  "
                  f"α={run['pitch_deg']:.0f}°  (~{est:.0f} min)")
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
                E_MeV = run["energy_eV"] / 1e6

                if status == "completed":
                    print(f"  [{completed_total:>4d}/{total}] ✓ {key}  "
                          f"{elapsed:.1f} min")
                    progress["completed"].append(key)
                else:
                    print(f"  [{completed_total:>4d}/{total}] ✗ {key}  "
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
            est = estimate_time(run)
            print(f"[{i+1:>4d}/{total}] {key}  "
                  f"E={E_MeV:.0f} MeV  L={run['L_shell']:.2f}  "
                  f"α={run['pitch_deg']:.0f}°  (~{est:.0f} min)")

            if dry_run:
                continue

            key, status, elapsed, err = execute_single_run(run)

            if status == "completed":
                print(f"         ✓ completed in {elapsed:.1f} min")
                progress["completed"].append(key)
                n_done += 1
            else:
                print(f"         ✗ {status} after {elapsed:.1f} min")
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
    parser = argparse.ArgumentParser(
        description="Batch flux map runner for dipoleB_adp.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --phase 1 --dry-run                        Preview Phase 1
  %(prog)s --phase 1 --workers 10                     Phase 1, fixed-step, 10 cores
  %(prog)s --phase 3 --workers 8 --driver dipoleB_adp.py Phase 3, adaptive-step
  %(prog)s --phase 1 --workers 10 --resume            Resume interrupted Phase 1
  %(prog)s --single E1e+07_L2.00_P89.0                Run one specific case

  # Michel phase portrait sweep
  %(prog)s --michel --L 1.5 --energy 10e6 --workers 10
  %(prog)s --michel --L 1.5 --energy 10e6 --phi-steps 18 --workers 10
  %(prog)s --michel --L 1.5 --energy 10e6 --dry-run
        """)
    parser.add_argument("--phase", type=int, default=1, choices=[1, 2, 3],
                        help="Which phase to run (default: 1)")
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of parallel workers (default: 1 = sequential). "
                             "Recommended: 8-10 for M4 Max, 4-6 for M1/M2.")
    parser.add_argument("--driver", type=str, default="dipoleB.py",
                        choices=["dipoleB.py", "dipoleB_adp.py"],
                        help="Which driver script to use (default: dipoleB.py). "
                             "Use dipoleB.py for fixed-step (fast, adiabatic regime). "
                             "Use dipoleB_adp.py for adaptive-step (high energy, chaotic).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print run plan without executing")
    parser.add_argument("--resume", action="store_true",
                        help="Skip runs already completed in previous session")
    parser.add_argument("--single", type=str, default=None,
                        help="Run a single case by key, e.g. 'E1e+07_L2.00_P89.0'")

    # Michel mode arguments
    michel_group = parser.add_argument_group("Michel phase portrait sweep")
    michel_group.add_argument("--michel", action="store_true",
                              help="Run a Michel phase portrait sweep instead of flux map phases")
    michel_group.add_argument("--L", type=float, default=1.5,
                              help="L-shell for Michel sweep (default: 1.5)")
    michel_group.add_argument("--energy", type=float, default=10e6,
                              help="Energy in eV for Michel sweep (default: 10e6 = 10 MeV)")
    michel_group.add_argument("--phi-steps", type=int, default=12,
                              help="Number of initial gyrophase values 0–360° (default: 12 = every 30°)")
    michel_group.add_argument("--michel-gyroperiods", type=float, default=None,
                              help="Override simulation length for Michel runs (default: 1e5)")
    args = parser.parse_args()

    # ── Build run list ──────────────────────────────────────────────
    if args.michel:
        runs = build_michel_sweep(
            energy_eV=args.energy,
            L_shell=args.L,
            phi_steps=args.phi_steps,
            gyroperiods=args.michel_gyroperiods,
        )
        mode_label = "MICHEL PHASE PORTRAIT"
    else:
        runs = build_run_list(args.phase)
        mode_label = f"BATCH FLUX MAP — Phase {args.phase}"

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
        runs = [r for r in runs if run_key(r) == args.single]
        if not runs:
            # Try building the full list to find it
            all_runs = build_run_list(args.phase) if not args.michel else runs
            matches = [r for r in all_runs if run_key(r) == args.single]
            if matches:
                runs = matches
            else:
                print(f"No match for '{args.single}'")
                sys.exit(1)

    if not runs:
        print("All runs already completed! Nothing to do.")
        sys.exit(0)

    # Summary
    total_est = sum(estimate_time(r) for r in runs)
    parallel_est = total_est / max(args.workers, 1)

    print(f"{'='*65}")
    print(f"  {mode_label}  [{args.driver}]")
    print(f"  Runs: {len(runs)}   Workers: {args.workers}")
    print(f"  Estimated time: {total_est/60:.1f} hrs sequential, "
          f"~{parallel_est/60:.1f} hrs with {args.workers} workers")
    print(f"  Energies: {sorted(set(r['energy_eV']/1e6 for r in runs))} MeV")
    print(f"  L-shells: {sorted(set(r['L_shell'] for r in runs))}")
    print(f"  Pitch angles: {sorted(set(r['pitch_deg'] for r in runs))}")
    if args.michel:
        print(f"  Gyrophase φ: {sorted(set(r['phi_deg'] for r in runs))}°")
        print(f"  Total: {len(MICHEL_PITCHES)} pitches × {args.phi_steps} φ values")
    # Show driver split info
    n_adaptive = sum(1 for r in runs if r.get("pitch_deg", 90) < 30.0)
    if n_adaptive > 0 and not args.michel:
        print(f"  Note: {n_adaptive} runs with pitch < 30° will use dipoleB_adp.py (adaptive)")
    print(f"{'='*65}")
    if args.dry_run:
        print("  *** DRY RUN — no simulations will execute ***\n")
    print()

    # Create directories
    out_dir = "outputs/michel" if args.michel else "outputs/flux_map"
    os.makedirs(os.path.join(PROJECT_ROOT, out_dir), exist_ok=True)
    os.makedirs(BATCH_TMP_DIR, exist_ok=True)

    progress["started"] = datetime.now().isoformat()

    # Inject driver into each run dict so workers know which script to call
    # Phase 2 & 3: always use adaptive integrator (dipoleB_adp.py) — non-adiabatic
    # Phase 1: use adaptive for low pitch angles (< 30°), user-chosen otherwise
    ADAPTIVE_PITCH_THRESHOLD = 30.0
    for r in runs:
        if args.phase in (2, 3):
            r["driver"] = "dipoleB_adp.py"
        elif r["pitch_deg"] < ADAPTIVE_PITCH_THRESHOLD:
            r["driver"] = "dipoleB_adp.py"
        else:
            r["driver"] = args.driver

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

    # Deduplicate CSV after parallel runs (race condition protection)
    if args.workers > 1 and not args.dry_run:
        print("\nDeduplicating master_simulation_log.csv...")
        merge_csv_logs()

    # Final summary
    label = "MICHEL SWEEP" if args.michel else f"Phase {args.phase}"
    print(f"\n{'='*65}")
    print(f"  BATCH COMPLETE — {label}")
    print(f"  Completed: {n_done}  Failed: {n_fail}")
    if n_fail > 0:
        print(f"  Failed runs logged in {PROGRESS_PATH}")
    if args.michel:
        E_MeV = args.energy / 1e6
        print(f"\n  Next step: build the portrait")
        print(f"    python michel_phase_portrait.py outputs/outputs_rawdata/run_*L{args.L}*.h5")
    print(f"{'='*65}")


def merge_csv_logs():
    """
    After parallel runs, the master_simulation_log.csv may have race-condition
    duplicates or missing entries. This rebuilds it by deduplicating on the
    standard key columns.
    """
    import pandas as pd
    csv_path = os.path.join(PROJECT_ROOT, "outputs", "flux_map", "master_simulation_log.csv")
    if not os.path.exists(csv_path):
        print("  No master_simulation_log.csv found — nothing to merge.")
        return

    df = pd.read_csv(csv_path)
    before = len(df)
    dup_keys = ["energy_keV", "L_eff", "phi_deg", "pitch_deg", "particle", "method", "steps"]
    existing_cols = [k for k in dup_keys if k in df.columns]
    df = df.drop_duplicates(subset=existing_cols, keep="last")
    after = len(df)
    df.to_csv(csv_path, index=False)

    if before != after:
        print(f"  CSV dedup: {before} → {after} rows ({before - after} duplicates removed)")
    else:
        print(f"  CSV clean: {after} rows, no duplicates.")


if __name__ == "__main__":
    main()
