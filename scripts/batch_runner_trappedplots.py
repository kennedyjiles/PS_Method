#!/usr/bin/env python3
"""
Trapped-particle L-sweep runner with phase-based refinement.

Runs dipoleb.py over an L grid at each energy in ENERGIES. PHASES are
refinement tiers: phase 1 is the coarse pass, later phases are finer windows
added after looking at the plot. All phases write to one group and
`--phase all` runs their de-duplicated union. Output feeds
scripts/plots/trappedbands.py.

Usage:
    python scripts/batch_runner_trappedplots.py --phase 1
    python scripts/batch_runner_trappedplots.py --phase 2 --resume
    python scripts/batch_runner_trappedplots.py --phase all --resume
    python scripts/batch_runner_trappedplots.py --list-phases
    python scripts/batch_runner_trappedplots.py data/dipoleb/<group>/master_simulation_log.csv   # rebuild that CSV only
    python scripts/batch_runner_trappedplots.py --energy 1e8 --L-start 2 --L-stop 4 --L-step 0.05 --phi 0 90 180 270
    python scripts/batch_runner_trappedplots.py --rerun-chaotic              # 1e6 T_g, adaptive, segmented
    python scripts/batch_runner_trappedplots.py --clean --phase all          # wipe group + ledger first

Output: <output_root>/data/dipoleb/<group>/fm_<run_key>/ (output_root from
base.yml or --output-root). --resume trusts the disk: a cell is done iff its
per-run master_simulation_log.csv exists, and cells with committed segments
continue from the last one rather than being wiped. Consolidation reindexes
the merged CSV to CANONICAL_COLUMNS, which must track writers.py:master_csv.
"""

import os
import sys
import glob
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
PROGRESS_PATH = os.path.join(SCRIPT_DIR, "batch_progress_trapped.json")
BATCH_TMP_DIR = os.path.join(SCRIPT_DIR, "batch_tmp")
BASE_YML      = os.path.join(PROJECT_ROOT, "configs", "dipoleb", "base.yml")


def _data_root():
    """Where dipoleb.py actually writes: honours base.yml's `output_root`.

    config_loader._resolve_output_paths() builds
        <output_root>/data/dipoleb/<group>/<config-name>/
    when output_root is set, else data/dipoleb/... relative to the cwd
    (PROJECT_ROOT, since execute_single_run launches with cwd=PROJECT_ROOT).
    The driver must look in the same place, or every run is reported as
    "no_output" even though it succeeded.
    """
    root = None
    try:
        with open(BASE_YML) as f:
            root = (yaml.safe_load(f) or {}).get("output_root")
    except OSError:
        pass
    if root:
        root = os.path.expanduser(str(root))
        if not os.path.isabs(root):
            root = os.path.join(PROJECT_ROOT, root)
        return os.path.join(root, "data", "dipoleb")
    return os.path.join(PROJECT_ROOT, "data", "dipoleb")


DATA_ROOT = _data_root()

# Default coarse L_stop for an ad-hoc --energy scan (energies in the table below
# carry their own; a custom scan has no table entry, so it falls back to this).
CUSTOM_L_STOP = 6.0

# Canonical master-CSV column order, mirroring the record dict in
# ps_method/writers.py:master_csv(). The consolidator reindexes to
# this so the group CSV keeps writers.py's ordering no matter which per-cell
# file happens to be read first. Columns added to writers.py later (the newest
# is n_eq_crossings, the PS equatorial-crossing count) must be added here too,
# in the same position; any column present in the data but missing from this
# list is still kept — it is appended after the canonical block.
CANONICAL_COLUMNS = [
    "run_id", "particle", "energy_eV", "gyroperiods",
    "x", "y", "z", "pitch_deg", "phi_deg",
    "L_eff", "W0_sq", "boundary", "mu_sq", "orbit_character",
    "steps", "dt", "method",
    "energy_mean_err", "energy_max_err", "mu_mean_err", "mu_max_err", "mu_max_run",
    "eps_initial", "eps_mean", "eps_max",
    "n_eq_crossings",
    "hit_atmosphere", "hit_atm_r", "n_mirror_crossings",
    "bounce_period_s", "bounce_freq_hz", "drift_period_s", "drift_direction",
]

# Columns introduced after the first trapped sweeps ran. Cells written before
# these existed legitimately lack them, so their absence is reported as
# coverage, not as a schema-drift warning.
LATE_ADDED_COLUMNS = {"n_eq_crossings"}


# ═══════════════════════════════════════════════════════════════════════════════
#  Study definition  (the only thing you edit day-to-day)
# ═══════════════════════════════════════════════════════════════════════════════

# All phases write here → one master CSV → one trappedbands plot.
GROUP = "trapped_60_temp"

# Shared defaults. A phase inherits these and overrides only what it names.
DEFAULTS = {
    "pitch_deg":   60.0,
    # launch gyrophase(s), deg: a number or a list. dipoleb sets
    # v_perp = v sin(alpha) (cos phi, sin phi, 0) at (x0, 0, 0), so 0 = radially
    # outward, 90 = +phi-hat (azimuthal), 180 = inward, 270 = -phi-hat.
    "phi_deg":     0.0,
    "gyroperiods": 1e2,
    "L_start":     1.5,
    "L_step":      0.1,    # coarse first-pass dL; refinement phases override this
}

# The varying axis: name → (energy_eV, coarse L_stop).
# Add more energies by appending a line. L_stop shrinks with energy.
ENERGIES = {
    "1e5": (1e5, 9.0),
    "1e6":   (1e6, 9.0),
    "1e7":  (1e7, 9.0),
    "1e8": (1e8, 6.0),
    "1e9":   (1e9, 3.0),
    "4e8":   (4e8, 2.0),
}
#### for 85 pitch #####
# ENERGIES = {         
#     "1e5": (1e5, 9.0),
#     "1e6":   (1e6, 9.0),
#     "1e7":  (1e7, 9.0),
#     "1e8": (1e8, 6.0),
#     "1e9":   (1e9, 3.0),
# }

PHASES = {
    1: {"energies": "all"},   # coarse first pass (dL = DEFAULTS["L_step"], per-energy L_stop)
    2: {"energies": ["1e9"], "L_start": 1.5, "L_stop": 3.0, "L_step": 0.1},
    3: {"energies": ["4e8"], "L_start": 2.0, "L_stop": 2.0, "L_step": 0.1, "gyroperiods": 1e6},
    4: {"energies": ["1e9"], "L_start": 2.31, "L_stop": 2.32, "L_step": 0.001, "gyroperiods": 1e6},
    5: {"energies": ["1e9"], "L_start": 1.5, "L_stop": 1.9, "L_step": 0.1, "gyroperiods": 1e3},
    # gyrophase sweep example: every point below runs at each listed phi
    # 6: {"energies": ["1e8"], "L_start": 3.0, "L_stop": 5.0, "L_step": 0.1,
    #     "phi_deg": [0, 90, 180, 270]},
    # 3: {"energies": ["1mev"],          "L_start": 2.5, "L_stop": 2.7, "L_step": 0.02},
}


# ═══════════════════════════════════════════════════════════════════════════════
#  Run-list construction
# ═══════════════════════════════════════════════════════════════════════════════

def _L_grid(L_start, L_stop, L_step):
    """Build inclusive L-shell array, rounded to avoid float noise."""
    vals = np.arange(L_start, L_stop + L_step / 2, L_step)
    return [round(v, 4) for v in vals]


def _phi_list(phi):
    """A phase's phi_deg as a list of floats (accepts a number or a list)."""
    vals = phi if isinstance(phi, (list, tuple)) else [phi]
    return [float(v) % 360.0 for v in vals]


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


def build_phase_runs(phase, phi_override=None):
    """Run dicts for one phase: each named energy swept over its own L-grid,
    at each of the phase's gyrophases.

    L_stop for an energy is the phase's override if it names one, else that
    energy's coarse L_stop from ENERGIES. phi_override (the --phi flag)
    replaces the phase's phi_deg list.
    """
    spec = _phase_spec(phase)
    names = list(ENERGIES) if spec["energies"] == "all" else list(spec["energies"])
    phis = _phi_list(spec["phi_deg"] if phi_override is None else phi_override)
    runs = []
    for name in names:
        energy_eV, energy_L_stop = ENERGIES[name]
        L_stop = spec.get("L_stop", energy_L_stop)   # phase override wins
        Ls = _L_grid(spec["L_start"], L_stop, spec["L_step"])
        for L in Ls:
            for phi in phis:
                runs.append({
                    "energy_eV":   float(energy_eV),
                    "x_initial":   L,
                    "pitch_deg":   float(spec["pitch_deg"]),
                    "phi_deg":     phi,
                    "gyroperiods": float(spec["gyroperiods"]),
                    "group":       GROUP,
                    "phase":       phase,
                })
    return runs


def build_all_phase_runs(phi_override=None):
    """Union of every phase, de-duplicated by run_key so overlapping coarse/fine
    L-values never run twice.  The lowest phase number to define a given
    (energy, L, pitch, phi) point wins."""
    seen = {}
    for phase in sorted(PHASES):
        for run in build_phase_runs(phase, phi_override):
            seen.setdefault(run_key(run), run)
    return list(seen.values())


def build_runs_custom(energy_eV, L_start, L_stop, L_step, pitch_deg, gyroperiods,
                      group, phi_deg=0.0):
    """Ad-hoc single-energy L-sweep from CLI flags (bypasses PHASES)."""
    Ls = _L_grid(L_start, L_stop, L_step)
    return [{
        "energy_eV":   float(energy_eV),
        "x_initial":   L,
        "pitch_deg":   float(pitch_deg),
        "phi_deg":     phi,
        "gyroperiods": float(gyroperiods),
        "group":       group,
        "phase":       "custom",
    } for L in Ls for phi in _phi_list(phi_deg)]


def build_chaotic_rerun_list(group):
    """Read the group's master CSV and return run dicts for CHAOTIC orbits."""
    import pandas as pd

    csv_path = os.path.join(DATA_ROOT, group, "master_simulation_log.csv")
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
    param_cols = ["energy_eV", "L_eff", "pitch_deg", "phi_deg"]
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
            # blank phi (older rows) reads as NaN -> the phi = 0 launch it was
            "phi_deg":          (float(row["phi_deg"]) if "phi_deg" in row
                                 and pd.notna(row["phi_deg"]) else 0.0),
            # Chaotic tier: 1e6 T_g regardless of the original run length,
            # so loss-cone first-passage / sticky-escape physics has room to
            # happen. Adaptive stepping is mandatory here — chaotic mu
            # wandering produces deep mirror excursions (and possible
            # sub-r_atmosphere dives) that fixed steps cannot resolve.
            "gyroperiods":      1e6,
            "group":            group,
            "use_adaptive":     True,
            "clean_before_run": True,   # wipe old fixed-step data
            # checkpoint every 5e4 T_g (~20 segments per run) so crashes
            # and precipitation blow-ups lose one segment, not the run
            "segment_gyroperiods": 5e4,
        })

    print(f"  Found {len(runs)} chaotic orbits to rerun with adaptive stepping.")
    return runs


def _print_phase_plan():
    """Show each phase's plan and the de-duplicated total."""
    print("\nPhases (each inherits DEFAULTS; overrides shown):")
    for ph in sorted(PHASES):
        spec = _phase_spec(ph)
        names = list(ENERGIES) if spec["energies"] == "all" else list(spec["energies"])
        phis = _phi_list(spec["phi_deg"])
        n_runs = 0
        for name in names:
            _, e_lstop = ENERGIES[name]
            L_stop = spec.get("L_stop", e_lstop)
            n_runs += len(_L_grid(spec["L_start"], L_stop, spec["L_step"])) * len(phis)
        ens_lbl   = "all" if spec["energies"] == "all" else ", ".join(names)
        lstop_lbl = spec["L_stop"] if "L_stop" in spec else "per-energy"
        print(f"  Phase {ph}: {len(names)} energies, dL={spec['L_step']}, "
              f"L {spec['L_start']}–{lstop_lbl}, pitch={spec['pitch_deg']}°, "
              f"phi={', '.join(f'{p:g}' for p in phis)}°, "
              f"gyros={spec['gyroperiods']:.0e}  = {n_runs} runs")
        print(f"           energies: {ens_lbl}")
    total = len(build_all_phase_runs())
    print(f"\n  --phase all → {total} unique runs after de-duplication")
    print(f"  Group: {GROUP}   →  data/dipoleb/{GROUP}/\n")


def _clean_group(group):
    """Wipe the group's data folder and reset the progress file (clean rebuild)."""
    import shutil
    group_dir = os.path.join(DATA_ROOT, group)
    if os.path.isdir(group_dir):
        shutil.rmtree(group_dir)
        print(f"  Cleaned data/dipoleb/{group}/")
    if os.path.exists(PROGRESS_PATH):
        os.remove(PROGRESS_PATH)
        print(f"  Reset progress ({os.path.basename(PROGRESS_PATH)})")


# ═══════════════════════════════════════════════════════════════════════════════
#  Config writing + execution
# ═══════════════════════════════════════════════════════════════════════════════

def run_key(run):
    # phi = 0 keeps the pre-gyrophase key, so existing run folders (and their
    # --resume detection) are unchanged; any other phi gets a _G<phi> suffix.
    key = f"E{run['energy_eV']:.1e}_L{run['x_initial']:.2f}_P{run['pitch_deg']:.1f}"
    phi = float(run.get("phi_deg", 0.0)) % 360.0
    return key if phi == 0.0 else f"{key}_G{phi:.1f}"


def write_config(run, config_path):
    """Write a minimal per-worker YAML that overrides base.yml."""
    cfg = {
        "base_config":  os.path.abspath(BASE_YML),
        "batch_group":  run["group"],
        "energy_eV":    float(run["energy_eV"]),
        "x_initial":    float(run["x_initial"]),
        "pitch_deg":    float(run["pitch_deg"]),
        # explicit even at 0: base.yml's phi_deg is 0.0, so the merged config
        # (and physics_hash) of existing phi = 0 runs is unchanged
        "phi_deg":      float(run.get("phi_deg", 0.0)),
        "gyroperiods":  float(run["gyroperiods"]),
        "particle":     "proton",
        "read_data":    False,
        "solvers":      {"adaptive": bool(run.get("use_adaptive", False))},
    }
    # Segmented checkpointing for long (chaotic-tier) runs: a mid-run crash
    # (e.g. precipitation dive stressing the integrator) costs one segment,
    # not the whole 1e6-T_g run. Hash-neutral for cache identity
    # (ps_segment_gyroperiods is in _NON_HASH_KEYS).
    if run.get("segment_gyroperiods"):
        cfg["ps_segment_gyroperiods"] = float(run["segment_gyroperiods"])
    # Per-run override beats base.yml's output_root (_deep_merge puts the run
    # config on top), so batches writing to different destinations can be
    # launched back to back without editing a shared file mid-flight.
    if run.get("output_root"):
        cfg["output_root"] = str(run["output_root"])
    with open(config_path, "w") as f:
        f.write("# Auto-generated trapped-sweep batch config — merged with base.yml\n")
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)


def data_root_for(run):
    """Where this run's output lives.

    A per-run output_root wins over the module-level DATA_ROOT (which is read
    from base.yml at import). It has to travel on the run dict: macOS spawns
    the pool workers, so rebinding the global in main() would never reach them.
    Mirrors config_loader._resolve_output_paths: <output_root>/data/dipoleb/.
    """
    root = run.get("output_root")
    if root:
        return os.path.join(os.path.expanduser(str(root)), "data", "dipoleb")
    return DATA_ROOT


def run_folder_for(run):
    return os.path.join(data_root_for(run), run["group"], f"fm_{run_key(run)}")


def segment_files(folder):
    """Committed <hash>_segNNN.h5 files for a run folder, both layouts.

    Canonical storage is <run_folder>/_rawdata/, but some runs carry their
    segments at the top level of the run folder. Miss one and the caller
    concludes there is nothing to resume -- and may then delete it.
    """
    return (glob.glob(os.path.join(folder, "*_seg*.h5"))
            + glob.glob(os.path.join(folder, "_rawdata", "*_seg*.h5"))
            + glob.glob(os.path.join(folder, "_rawdata", "*", "*_seg*.h5")))


def committed_segments(run):
    """Highest committed segment index on disk for this run, or None.

    dipoleb's _run_ps_segments resumes from the last committed segment by
    itself; this is so the batch runner can report it and, above all, refuse to
    delete it. (.building files are ignored -- they are the segment that was
    mid-write when the run died, and clear_building_segments removes them.)
    """
    idx = []
    for f in segment_files(run_folder_for(run)):
        stem = os.path.basename(f).rsplit("_seg", 1)[-1].split(".")[0]
        if stem.isdigit():
            idx.append(int(stem))
    return max(idx) if idx else None


def run_is_complete(run):
    """A finished run has written its per-run master_simulation_log.csv."""
    return os.path.exists(os.path.join(run_folder_for(run),
                                       "master_simulation_log.csv"))


def execute_single_run(run):
    """Run one dipoleb.py subprocess. Returns (key, status, elapsed_min, error)."""
    key = run_key(run)

    run_folder = run_folder_for(run)

    # Clean old run folder before rerun (e.g. chaotic reruns with adaptive).
    # Two guards, because this is an rmtree of a folder that can hold >100 GB
    # of committed checkpoints:
    #   keep_segments  -- set by the --resume filter when segments were found
    #   segment_files  -- a live re-check, so a run that is resumable is never
    #                     wiped implicitly even if the filter missed it
    # Only an explicit --clean is allowed to remove committed segments.
    if run.get("clean_before_run", False) and not run.get("keep_segments", False):
        import shutil
        if os.path.isdir(run_folder):
            if segment_files(run_folder):
                print(f"  {key}: refusing to wipe — committed segments present; "
                      f"resuming instead (use --clean to discard them)")
            else:
                shutil.rmtree(run_folder)

    config_path = os.path.join(BATCH_TMP_DIR, f"fm_{key}.yml")
    write_config(run, config_path)

    start = time.time()
    try:
        result = subprocess.run(
            [sys.executable, "dipoleb.py", config_path],
            cwd=PROJECT_ROOT,
            # None = no timeout (the default). A long chaotic-tier run can
            # legitimately exceed any fixed wall-clock budget, and killing it
            # mid-segment throws away that segment's work.
            timeout=run.get("timeout_s"),
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
        _h = (run.get("timeout_s") or 0) / 3600.0
        return (key, "timeout", elapsed, f"{_h:g}h timeout exceeded "
                                         f"(committed segments are kept; "
                                         f"re-run with --resume to continue)")
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
                  f"phi={run.get('phi_deg', 0.0):g}°  "
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

def _order_columns(cols):
    """Canonical order first, then any unrecognized columns (alphabetically)."""
    known = [c for c in CANONICAL_COLUMNS if c in cols]
    extra = sorted(c for c in cols if c not in CANONICAL_COLUMNS)
    return known + extra


def consolidate_csv_logs(group):
    """
    Merge per-cell master_simulation_log.csv into one at the group level.

    Takes the UNION of columns across all per-cell files, then reindexes to
    CANONICAL_COLUMNS so the group CSV always has writers.py's column order —
    a cell written before a column existed (e.g. n_eq_crossings) contributes a
    blank there rather than shifting the schema. Columns in LATE_ADDED_COLUMNS
    are reported as coverage; anything else missing is a real drift warning.
    """
    import glob
    import pandas as pd

    group_dir = os.path.join(DATA_ROOT, group)
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
        drifted = {cols: n for cols, n in schemas.items()
                   if (union - set(cols)) - LATE_ADDED_COLUMNS}
        if drifted:
            print(f"  WARNING: {len(schemas)} different per-cell schemas detected:")
            for cols, count in sorted(schemas.items(), key=lambda kv: -kv[1]):
                missing = sorted((union - set(cols)) - LATE_ADDED_COLUMNS)
                print(f"    [{count:4d} files]  missing columns: {missing if missing else 'none'}")
            print(f"  Taking column union; rows missing a column will have NaN there.")
        # Columns added to writers.py mid-study: expected to be absent from
        # older cells, so report how far they reach instead of warning.
        for col in sorted(LATE_ADDED_COLUMNS & union):
            have = sum(n for cols, n in schemas.items() if col in cols)
            print(f"  Note: '{col}' present in {have}/{len(csv_files)} cells "
                  f"(added after the earlier cells ran — they stay blank).")

    df = pd.concat(frames, ignore_index=True)
    dup_keys = ["energy_eV", "L_eff", "phi_deg", "pitch_deg", "particle", "method", "steps"]
    existing_cols = [k for k in dup_keys if k in df.columns]
    df = df.drop_duplicates(subset=existing_cols, keep="last")
    df = df.reindex(columns=_order_columns(df.columns))

    unknown = [c for c in df.columns if c not in CANONICAL_COLUMNS]
    if unknown:
        print(f"  Note: columns not in CANONICAL_COLUMNS kept at the end: {unknown}")
        print(f"        (add them to CANONICAL_COLUMNS to fix their position)")

    master_path = os.path.join(group_dir, "master_simulation_log.csv")
    df.to_csv(master_path, index=False)
    print(f"  Consolidated {len(csv_files)} per-cell CSVs → {master_path}  "
          f"({len(df)} rows × {len(df.columns)} cols)")

    # Per-column fill report for the late-added diagnostics, so a plot that
    # needs one knows up front how much of the sweep actually carries it.
    for col in sorted(LATE_ADDED_COLUMNS):
        if col in df.columns:
            n_have = int(df[col].notna().sum())
            print(f"  {col}: {n_have}/{len(df)} rows populated")


# ═══════════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════════

def group_from_target(target):
    """Resolve a positional target to a group (folder) name.

    Accepts what you would naturally paste from the plotting command line:
      data/dipoleb/trapped_85/master_simulation_log.csv  → "trapped_85"
      data/dipoleb/trapped_85                            → "trapped_85"
      data/dipoleb/trapped_85/                           → "trapped_85"
      trapped_85                                         → "trapped_85"
    """
    t = target.rstrip(os.sep)
    if t.lower().endswith(".csv"):
        t = os.path.dirname(t)
        if not t:
            raise SystemExit(
                f"Cannot tell which group {target!r} belongs to — pass the CSV "
                f"inside its group folder, e.g. data/dipoleb/{GROUP}/master_simulation_log.csv"
            )
    group = os.path.basename(t.rstrip(os.sep))
    if not group:
        raise SystemExit(f"Could not resolve a group name from {target!r}.")

    group_dir = os.path.join(DATA_ROOT, group)
    if not os.path.isdir(group_dir):
        print(f"  Note: data/dipoleb/{group}/ does not exist yet — it will be "
              f"created when runs execute.")
    return group


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

Group targeting (positional — the same path trappedbands.py takes):
  %(prog)s data/dipoleb/{GROUP}/master_simulation_log.csv
                                      Rebuild THAT group's master CSV
  %(prog)s data/dipoleb/{GROUP} --phase 2 --resume
                                      Run phase 2 into that group

After runs finish, plot:
  python scripts/plots/trappedbands.py data/dipoleb/{GROUP}/master_simulation_log.csv
""")

    # Positional target: the same path you hand to trappedbands.py. It only
    # selects the group; on its own (no phase/energy/rerun flag) it rebuilds
    # that group's master CSV rather than silently launching a full sweep.
    parser.add_argument("target", nargs="?", default=None,
                        help="Group to act on: a master CSV path, a group folder, "
                             "or a bare group name (e.g. "
                             f"data/dipoleb/{GROUP}/master_simulation_log.csv). "
                             "Alone, it rebuilds that group's master CSV.")

    parser.add_argument("--phase", default=None,
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
    parser.add_argument("--phi", type=float, nargs="+", default=None,
                        help=f"Launch gyrophase(s) in deg, e.g. --phi 0 90 180 270. "
                             f"Sets a custom scan's phi and replaces every phase's "
                             f"phi_deg list (default: DEFAULTS/phase value, "
                             f"{DEFAULTS['phi_deg']}). Chaotic reruns keep each "
                             f"row's own phi.")
    parser.add_argument("--gyroperiods", type=float, default=None,
                        help=f"Override gyroperiods for ANY run list — phase, custom, "
                             f"or chaotic rerun (default: {DEFAULTS['gyroperiods']:.0e} "
                             f"for phases, 1e6 for --rerun-chaotic).")
    parser.add_argument("--segment-gyroperiods", type=float, default=None,
                        dest="segment_gyroperiods",
                        help="Checkpoint segment length in gyroperiods (hash-neutral; "
                             "chaotic reruns default to 5e4).")

    parser.add_argument("--adaptive", action="store_true",
                        help="Use adaptive PS stepping (default: fixed-step).")
    parser.add_argument("--workers", type=int, default=_default_workers,
                        help=f"Parallel workers (default: {_default_workers}).")
    parser.add_argument("--dry-run", action="store_true", dest="dry_run",
                        help="Print the run plan without executing.")
    parser.add_argument("--output-root", default=None, dest="output_root",
                        help="Write this batch under <PATH>/data/dipoleb/ "
                             "instead of base.yml's output_root. Per-invocation, "
                             "so external and local batches can run back to back "
                             "without editing base.yml.")
    parser.add_argument("--progress-file", default=None, dest="progress_file",
                        help="Progress ledger for this batch (default: "
                             "scripts/batch_progress_trapped.json). Give a "
                             "separate file when two batches run at once — each "
                             "runner rewrites its ledger wholesale.")
    parser.add_argument("--timeout-hours", type=float, default=None,
                        dest="timeout_hours",
                        help="Wall-clock limit per cell in hours. Default: no "
                             "timeout. A killed run keeps its committed "
                             "segments and continues under --resume.")
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

    # Per-invocation overrides, applied before anything reads these. Both are
    # parent-side only (consolidation, --clean, the progress ledger); the value
    # that the spawned workers need rides on each run dict instead.
    global DATA_ROOT, PROGRESS_PATH
    if args.output_root:
        DATA_ROOT = os.path.join(os.path.expanduser(args.output_root),
                                 "data", "dipoleb")
        print(f"Output root override: {DATA_ROOT}")
    if args.progress_file:
        PROGRESS_PATH = os.path.abspath(os.path.expanduser(args.progress_file))
        print(f"Progress ledger:      {PROGRESS_PATH}")

    validate_phases()

    if args.target is not None:
        target_group = group_from_target(args.target)
        if args.group and args.group != target_group:
            raise SystemExit(
                f"Conflicting groups: positional {args.target!r} → {target_group!r}, "
                f"but --group says {args.group!r}. Pass only one."
            )
        group = target_group
    else:
        group = args.group or GROUP

    # A bare `prog <master CSV>` (no action flag) means "work on this group's
    # CSV" — rebuild it, don't kick off the full de-duplicated sweep.
    if (args.target is not None and args.phase is None and args.energy is None
            and not (args.rerun_chaotic or args.list_phases or args.clean)):
        args.consolidate_only = True
        print(f"No phase/energy/rerun flag given with {args.target!r} — "
              f"rebuilding that group's master CSV.")
        print(f"(To run simulations for this group, add e.g. "
              f"--phase all --resume.)\n")

    if args.phase is None:
        args.phase = "all"

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
            group,
            DEFAULTS["phi_deg"]     if args.phi         is None else args.phi)
        mode = "CUSTOM SCAN"
    elif str(args.phase).lower() == "all":
        all_runs = build_all_phase_runs(args.phi)
        mode = "ALL PHASES"
    else:
        try:
            ph = int(args.phase)
        except ValueError:
            raise SystemExit(f"--phase must be an integer or 'all' (got {args.phase!r}).")
        if ph not in PHASES:
            raise SystemExit(f"No phase {ph}. Defined phases: {sorted(PHASES)}.")
        all_runs = build_phase_runs(ph, args.phi)
        mode = f"PHASE {ph}"

    if not all_runs:
        print("No runs to build (empty phase / no chaotic orbits).")
        return

    # Resolve the output group + adaptive flag on every run.
    # --gyroperiods / --segment-gyroperiods override ANY run list
    # (phase, custom, or chaotic rerun), so a failure mop-up can be given
    # the full chaotic-tier profile, e.g.:
    #   --phase 1 --resume --adaptive --gyroperiods 1e6 --segment-gyroperiods 5e4
    for r in all_runs:
        r["group"] = group
        r["use_adaptive"] = True if args.rerun_chaotic else args.adaptive
        if args.gyroperiods is not None:
            r["gyroperiods"] = float(args.gyroperiods)
        if args.segment_gyroperiods is not None:
            r["segment_gyroperiods"] = float(args.segment_gyroperiods)

    # Carried on each run dict, not a module global: ProcessPoolExecutor
    # spawns workers on macOS, so globals set here would not reach them.
    _timeout_s = args.timeout_hours * 3600.0 if args.timeout_hours else None
    for r in all_runs:
        r["timeout_s"] = _timeout_s
        if args.output_root:
            r["output_root"] = args.output_root

    # ── Resume filter ────────────────────────────────────────────────
    runs = all_runs
    if args.resume:
        progress = load_progress()
        done = set(progress["completed"])
        before = len(runs)
        # The disk is the authority in BOTH directions: a ledger entry is a
        # claim, the run's own master_simulation_log.csv is the evidence.
        # Trusting the ledger alone would skip runs whose output was moved,
        # cleaned, or never landed, and would miss runs the ledger never saw.
        keep, from_disk, stale = [], 0, []
        for r in runs:
            in_ledger = run_key(r) in done
            if run_is_complete(r):
                if not in_ledger:
                    from_disk += 1
                continue
            if in_ledger:
                stale.append(run_key(r))
            keep.append(r)
        runs = keep
        skipped = before - len(runs)
        if skipped:
            print(f"Resuming: skipping {skipped} already-completed runs"
                  + (f" ({from_disk} detected on disk, not in progress.json)"
                     if from_disk else "") + ".")
        if stale:
            print(f"  {len(stale)} run(s) marked completed in "
                  f"{os.path.basename(PROGRESS_PATH)} have no "
                  f"master_simulation_log.csv on disk — re-running them:")
            for k in stale:
                print(f"     {k}")
        # Anything left with committed segments continues from the last one
        # rather than restarting: keep_segments blocks the pre-run wipe, and
        # dipoleb's _run_ps_segments resumes from the last committed segment.
        partial = 0
        for r in runs:
            last = committed_segments(r)
            if last is not None:
                r["keep_segments"] = True
                partial += 1
        if partial:
            print(f"  {partial} run(s) have committed segments and will "
                  f"continue from where they stopped.")
            for r in runs:
                last = committed_segments(r)
                if last is not None:
                    print(f"     {run_key(r)}: resuming after segment {last}")
        print()

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
    print(f"  Gyrophases (deg): {sorted(set(r.get('phi_deg', 0.0) for r in runs))}")
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
