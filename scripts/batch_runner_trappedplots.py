#!/usr/bin/env python3
"""
Trapped-Particle Sweep Runner (phase-based refinement)
======================================================
Sweeps an L-shell grid at each energy, producing the h5/CSV runs that
scripts/plots/trappedbands.py consumes.

The sweep is organized into PHASES, mirroring batch_runner_eandp. A phase is a
refinement tier: phase 1 is the coarse first pass (large dL over every energy);
later phases refine the (energies, L-window, dL) you pick after examining the
plots. You add or tweak a phase by editing one dict — never by copying presets.
Shared constants (pitch, gyroperiods, coarse L-range) live in DEFAULTS once;
each energy carries its own coarse L_stop; a phase states only what differs.

All phases feed ONE group (folder) → one master CSV → one trappedbands plot.

Typical loop
------------
    python scripts/batch_runner_trappedplots.py --phase 1              # coarse pass
    # ...examine the trappedbands plot, decide who needs finer dL...
    #    append a phase to PHASES below, then:
    python scripts/batch_runner_trappedplots.py --phase 2 --resume     # only new points run
    # ...repeat...
    python scripts/batch_runner_trappedplots.py --phase all --resume   # assemble everything
    python scripts/plots/trappedbands.py data/dipoleb/trapped/master_simulation_log.csv

`--phase all` runs the de-duplicated union of every phase: a fine L that
coincides with a coarse L is never run twice, and --resume skips anything
already done, so you get exactly the data the plot needs and no more.
"""

import os
import sys
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

# Default coarse L_stop for an ad-hoc --energy scan (energies in the table below
# carry their own; a custom scan has no table entry, so it falls back to this).
CUSTOM_L_STOP = 6.0


# ═══════════════════════════════════════════════════════════════════════════════
#  Study definition  (the only thing you edit day-to-day)
# ═══════════════════════════════════════════════════════════════════════════════

# All phases write here → one master CSV → one trappedbands plot.
GROUP = "trapped_60b"

# Shared defaults. A phase inherits these and overrides only what it names.
DEFAULTS = {
    "pitch_deg":   60.0,
    "gyroperiods": 1e5,
    "L_start":     1.1,
    "L_step":      0.1,    # coarse first-pass dL; refinement phases override this
}

# The varying axis: name → (energy_eV, coarse L_stop).
# Add more energies by appending a line. L_stop shrinks with energy.
ENERGIES = {
    "1e1":   (1e1, 9.0),
    "1e2":  (1e2, 9.0),
    "1e3":   (1e3, 9.0),
    "1e4":  (1e4, 9.0),
    "1e5": (1e5, 9.0),
    "1e6":   (1e6, 9.0),
    "1e7":  (1e7, 9.0),
    "1e8": (1e8, 6.0),
    "1e9":   (1e9, 3.0),
}

# Refinement phases.  APPEND a new phase or TWEAK an existing one anytime.
#   "energies" : the string "all", or a list of ENERGIES keys.
#   L_start / L_step / pitch_deg / gyroperiods : override DEFAULTS if named.
#   L_stop     : override the per-energy coarse L_stop if named (else per-energy).
#
# ─── Worked example ──────────────────────────────────────────────────────────
# Round 1 — run the coarse pass and look at the plot:
#     python scripts/batch_runner_trappedplots.py --phase 1
#     python scripts/plots/trappedbands.py data/dipoleb/trapped/master_simulation_log.csv
#
#   Say the plot shows the trapped→open transition for 1 MeV and 10 MeV falls
#   somewhere between the coarse points L=2.1 and L=3.1 — you can't tell exactly
#   where. Add a phase that walks that window at dL=0.1 for just those energies:
#     2: {"energies": ["1mev", "10mev"], "L_start": 2.1, "L_stop": 3.1, "L_step": 0.1},
#
#   Then run ONLY the new points (coarse already done, --resume skips them):
#     python scripts/batch_runner_trappedplots.py --phase 2 --resume
#
# Round 2 — the boundary now looks like it sits near L=2.6 for 1 MeV; zoom in:
#     3: {"energies": ["1mev"], "L_start": 2.5, "L_stop": 2.7, "L_step": 0.02},
#     python scripts/batch_runner_trappedplots.py --phase 3 --resume
#
# Done — assemble everything (deduped, nothing re-run) for the final figure:
#     python scripts/batch_runner_trappedplots.py --phase all --resume
#
# The commented 2/3 below are that example, ready to uncomment. `--list-phases`
# shows the run counts before you launch; overlapping L values (e.g. L=2.1/3.1
# shared by coarse + phase 2, or L=2.6 shared by phase 2 + phase 3) run once.
# ─────────────────────────────────────────────────────────────────────────────
PHASES = {
    1: {"energies": "all"},   # coarse first pass (dL = DEFAULTS["L_step"], per-energy L_stop)
    2: {"energies": ["1e9"], "L_start": 1.5, "L_stop": 2.5, "L_step": 0.01},
    3: {"energies": ["1e8"], "L_start": 3.5, "L_stop": 4.5, "L_step": 0.01},
    4: {"energies": ["1e7"], "L_start": 6.5, "L_stop": 8.5, "L_step": 0.01},
    # 3: {"energies": ["1mev"],          "L_start": 2.5, "L_stop": 2.7, "L_step": 0.02},
}


# ═══════════════════════════════════════════════════════════════════════════════
#  Run-list construction
# ═══════════════════════════════════════════════════════════════════════════════

def _L_grid(L_start, L_stop, L_step):
    """Build inclusive L-shell array, rounded to avoid float noise."""
    vals = np.arange(L_start, L_stop + L_step / 2, L_step)
    return [round(v, 4) for v in vals]


def validate_phases():
    """Fail fast if a phase names an energy key that isn't in ENERGIES."""
    stale = {}
    for ph, spec in PHASES.items():
        ens = spec.get("energies", "all")
        if ens == "all":
            continue
        bad = [e for e in ens if e not in ENERGIES]
        if bad:
            stale[ph] = bad
    if stale:
        lines = "\n".join(f"    phase {ph}: unknown energies {bad}"
                          for ph, bad in stale.items())
        raise SystemExit(
            "ERROR: PHASES reference energy keys not in ENERGIES:\n"
            f"{lines}\n  Valid keys: {sorted(ENERGIES)}"
        )


def _phase_spec(phase):
    """DEFAULTS merged with this phase's overrides."""
    spec = dict(DEFAULTS)
    spec.update(PHASES[phase])
    return spec


def build_phase_runs(phase):
    """Run dicts for one phase: each named energy swept over its own L-grid.

    L_stop for an energy is the phase's override if it names one, else that
    energy's coarse L_stop from ENERGIES.
    """
    spec = _phase_spec(phase)
    names = list(ENERGIES) if spec["energies"] == "all" else list(spec["energies"])
    runs = []
    for name in names:
        energy_eV, energy_L_stop = ENERGIES[name]
        L_stop = spec.get("L_stop", energy_L_stop)   # phase override wins
        Ls = _L_grid(spec["L_start"], L_stop, spec["L_step"])
        for L in Ls:
            runs.append({
                "energy_eV":   float(energy_eV),
                "x_initial":   L,
                "pitch_deg":   float(spec["pitch_deg"]),
                "gyroperiods": float(spec["gyroperiods"]),
                "group":       GROUP,
                "phase":       phase,
            })
    return runs


def build_all_phase_runs():
    """Union of every phase, de-duplicated by run_key so overlapping coarse/fine
    L-values never run twice.  The lowest phase number to define a given
    (energy, L, pitch) point wins."""
    seen = {}
    for phase in sorted(PHASES):
        for run in build_phase_runs(phase):
            seen.setdefault(run_key(run), run)
    return list(seen.values())


def build_runs_custom(energy_eV, L_start, L_stop, L_step, pitch_deg, gyroperiods, group):
    """Ad-hoc single-energy L-sweep from CLI flags (bypasses PHASES)."""
    Ls = _L_grid(L_start, L_stop, L_step)
    return [{
        "energy_eV":   float(energy_eV),
        "x_initial":   L,
        "pitch_deg":   float(pitch_deg),
        "gyroperiods": float(gyroperiods),
        "group":       group,
        "phase":       "custom",
    } for L in Ls]


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
            "energy_eV":        float(row["energy_eV"]),
            "x_initial":        round(L_val, 4),
            "pitch_deg":        float(row["pitch_deg"]),
            "gyroperiods":      float(row["gyroperiods"]),
            "group":            group,
            "use_adaptive":     True,
            "clean_before_run": True,   # wipe old fixed-step data
        })

    print(f"  Found {len(runs)} chaotic orbits to rerun with adaptive stepping.")
    return runs


def _print_phase_plan():
    """Show each phase's plan and the de-duplicated total."""
    print("\nPhases (each inherits DEFAULTS; overrides shown):")
    for ph in sorted(PHASES):
        spec = _phase_spec(ph)
        names = list(ENERGIES) if spec["energies"] == "all" else list(spec["energies"])
        n_runs = 0
        for name in names:
            _, e_lstop = ENERGIES[name]
            L_stop = spec.get("L_stop", e_lstop)
            n_runs += len(_L_grid(spec["L_start"], L_stop, spec["L_step"]))
        ens_lbl   = "all" if spec["energies"] == "all" else ", ".join(names)
        lstop_lbl = spec["L_stop"] if "L_stop" in spec else "per-energy"
        print(f"  Phase {ph}: {len(names)} energies, dL={spec['L_step']}, "
              f"L {spec['L_start']}–{lstop_lbl}, pitch={spec['pitch_deg']}°, "
              f"gyros={spec['gyroperiods']:.0e}  = {n_runs} runs")
        print(f"           energies: {ens_lbl}")
    total = len(build_all_phase_runs())
    print(f"\n  --phase all → {total} unique runs after de-duplication")
    print(f"  Group: {GROUP}   →  data/dipoleb/{GROUP}/\n")


def _clean_group(group):
    """Wipe the group's data folder and reset the progress file (clean rebuild)."""
    import shutil
    group_dir = os.path.join(PROJECT_ROOT, "data", "dipoleb", group)
    if os.path.isdir(group_dir):
        shutil.rmtree(group_dir)
        print(f"  Cleaned data/dipoleb/{group}/")
    if os.path.exists(PROGRESS_PATH):
        os.remove(PROGRESS_PATH)
        print(f"  Reset progress ({os.path.basename(PROGRESS_PATH)})")


# ═══════════════════════════════════════════════════════════════════════════════
#  Config writing + execution  (unchanged machinery)
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
        f.write("# Auto-generated trapped-sweep batch config — merged with base.yml\n")
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
            # A clean exit is NOT proof of success: some runs (e.g. a particle
            # lost almost immediately) exit 0 without writing their
            # master_simulation_log.csv row. Verify the output exists, else the
            # run would be marked "completed" and silently skipped on --resume.
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
                  f"E={E_MeV:.3g} MeV  L={run['x_initial']:.2f}  "
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
                    if key in progress["failed"]:
                        progress["failed"].remove(key)
                else:
                    n_fail += 1
                    if err:
                        for line in err.split("\n")[-3:]:
                            print(f"          {line}")
                    if key not in progress["failed"]:
                        progress["failed"].append(key)
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
    in one cell doesn't truncate older cells' data). Warns about schema drift.
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
    schemas = {}
    for f in csv_files:
        try:
            df_one = pd.read_csv(f, dtype={"run_id": str})
            schemas[frozenset(df_one.columns)] = schemas.get(frozenset(df_one.columns), 0) + 1
            frames.append(df_one)
        except Exception as e:
            print(f"  Warning: could not read {f}: {e}")

    if not frames:
        return

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
        description="Trapped-particle L-sweep runner with phase-based refinement.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Workflow:
  %(prog)s --phase 1                  Run the coarse first pass (phase 1)
  %(prog)s --list-phases              Show every phase's plan + deduped total
  %(prog)s --phase 2 --resume         Run a refinement phase, skipping done cells
  %(prog)s --phase all --resume       Assemble the whole deduped dataset
  %(prog)s --dry-run --phase all      Preview the full plan
  %(prog)s --consolidate-only         Rebuild the master CSV only
  %(prog)s --clean --phase all        Wipe the group + progress, then rebuild

After runs finish, plot:
  python scripts/plots/trappedbands.py data/dipoleb/{GROUP}/master_simulation_log.csv
""")

    parser.add_argument("--phase", default="all",
                        help="Phase to run: a number (e.g. 1) or 'all' (default) "
                             "for the de-duplicated union of every phase.")
    parser.add_argument("--list-phases", action="store_true", dest="list_phases",
                        help="Show each phase's plan and the deduped total, then exit.")
    parser.add_argument("--group", type=str, default=None,
                        help=f"Output group override (default: {GROUP}).")

    # Ad-hoc custom scan (bypasses PHASES)
    parser.add_argument("--energy", type=float, default=None,
                        help="Ad-hoc scan at this energy (eV), bypassing PHASES.")
    parser.add_argument("--L-start", type=float, default=None, dest="L_start")
    parser.add_argument("--L-stop", type=float, default=None, dest="L_stop")
    parser.add_argument("--L-step", type=float, default=None, dest="L_step")
    parser.add_argument("--pitch", type=float, default=None,
                        help=f"Pitch (deg) for a custom scan (default: {DEFAULTS['pitch_deg']}).")
    parser.add_argument("--gyroperiods", type=float, default=None,
                        help=f"Gyroperiods for a custom scan (default: {DEFAULTS['gyroperiods']:.0e}).")

    parser.add_argument("--adaptive", action="store_true",
                        help="Use adaptive PS stepping (default: fixed-step).")
    parser.add_argument("--workers", type=int, default=_default_workers,
                        help=f"Parallel workers (default: {_default_workers}).")
    parser.add_argument("--dry-run", action="store_true", dest="dry_run",
                        help="Print the run plan without executing.")
    parser.add_argument("--resume", action="store_true",
                        help="Skip runs already completed in the progress file.")
    parser.add_argument("--consolidate-only", action="store_true", dest="consolidate_only",
                        help="Rebuild the group's master CSV without running.")
    parser.add_argument("--rerun-chaotic", action="store_true", dest="rerun_chaotic",
                        help="Re-run CHAOTIC orbits from the master CSV with adaptive stepping.")
    parser.add_argument("--clean", action="store_true",
                        help="Wipe the group's data folder AND the progress file "
                             "before running (clean rebuild).")

    args = parser.parse_args()
    validate_phases()

    group = args.group or GROUP

    # ── List phases ──────────────────────────────────────────────────
    if args.list_phases:
        _print_phase_plan()
        return

    # ── Consolidate-only ─────────────────────────────────────────────
    if args.consolidate_only:
        print(f"Consolidating data/dipoleb/{group}/ ...")
        consolidate_csv_logs(group)
        return

    # ── Clean rebuild ────────────────────────────────────────────────
    if args.clean:
        _clean_group(group)

    # ── Build the run list ───────────────────────────────────────────
    if args.rerun_chaotic:
        print(f"Scanning data/dipoleb/{group}/master_simulation_log.csv for chaotic orbits...")
        all_runs = build_chaotic_rerun_list(group)
        mode = "RERUN CHAOTIC"
    elif args.energy is not None:
        all_runs = build_runs_custom(
            args.energy,
            DEFAULTS["L_start"]     if args.L_start     is None else args.L_start,
            CUSTOM_L_STOP           if args.L_stop      is None else args.L_stop,
            DEFAULTS["L_step"]      if args.L_step      is None else args.L_step,
            DEFAULTS["pitch_deg"]   if args.pitch       is None else args.pitch,
            DEFAULTS["gyroperiods"] if args.gyroperiods is None else args.gyroperiods,
            group)
        mode = "CUSTOM SCAN"
    elif str(args.phase).lower() == "all":
        all_runs = build_all_phase_runs()
        mode = "ALL PHASES"
    else:
        try:
            ph = int(args.phase)
        except ValueError:
            raise SystemExit(f"--phase must be an integer or 'all' (got {args.phase!r}).")
        if ph not in PHASES:
            raise SystemExit(f"No phase {ph}. Defined phases: {sorted(PHASES)}.")
        all_runs = build_phase_runs(ph)
        mode = f"PHASE {ph}"

    if not all_runs:
        print("No runs to build (empty phase / no chaotic orbits).")
        return

    # Resolve the output group + adaptive flag on every run.
    for r in all_runs:
        r["group"] = group
        r["use_adaptive"] = True if args.rerun_chaotic else args.adaptive

    # ── Resume filter ────────────────────────────────────────────────
    runs = all_runs
    if args.resume:
        progress = load_progress()
        before = len(runs)
        runs = [r for r in runs if run_key(r) not in progress["completed"]]
        skipped = before - len(runs)
        if skipped:
            print(f"Resuming: skipping {skipped} already-completed runs.\n")

    if not runs:
        print("Nothing to do — all requested runs already completed.")
        return

    # ── Summary ──────────────────────────────────────────────────────
    energies = sorted(set(r["energy_eV"] for r in runs))
    L_vals   = sorted(set(r["x_initial"] for r in runs))
    print(f"{'='*65}")
    print(f"  TRAPPED SWEEP — {mode}   group: {group}")
    print(f"  Output: data/dipoleb/{group}/")
    print(f"  Runs: {len(runs)}   Workers: {args.workers}   "
          f"Stepping: {'adaptive' if (args.adaptive or args.rerun_chaotic) else 'fixed-step'}")
    print(f"  Energies (eV): {[f'{e:.0e}' for e in energies]}")
    print(f"  L-values: {len(L_vals)}  ({L_vals[0]:.2f} – {L_vals[-1]:.2f})")
    print(f"{'='*65}")
    if args.dry_run:
        print("  *** DRY RUN — no simulations will execute ***")

    os.makedirs(BATCH_TMP_DIR, exist_ok=True)

    # ── Execute ──────────────────────────────────────────────────────
    n_done = n_fail = 0
    try:
        n_done, n_fail = run_parallel(runs, args.workers, dry_run=args.dry_run)
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
    print(f"  TRAPPED SWEEP COMPLETE — {mode}   group: {group}")
    print(f"  Completed: {n_done}  Failed: {n_fail}")
    print(f"  Master CSV: data/dipoleb/{group}/master_simulation_log.csv")
    print(f"\n  Plot:")
    print(f"    python scripts/plots/trappedbands.py data/dipoleb/{group}/master_simulation_log.csv")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
