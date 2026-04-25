"""
Output writers extracted from dipoleB.py — summary text file and master CSV log.

Each function receives only the data it needs (no globals). Called from
dipoleB.py after all computation and plotting is complete.
"""

import os
import gc
import numpy as np
import pandas as pd
import h5py
from functions.functions_library_dipole import (
    build_figure_filename, write_dict, summarize_error, summarize,
    compute_mu_ps, compute_mu_rk, vector_potential_dipole, expand_h5_to_full,
)


# ===================================================================
# ============== Tail mask helper (shared by summary writer) ========
# ===================================================================
def _make_tail_mask(n_points, step_size, tail_start, max_tail_steps):
    """Build a boolean mask for the last fraction of a time series."""
    j0 = int(tail_start / step_size)
    j0 = max(0, min(j0, n_points - 1))

    if n_points - j0 > max_tail_steps:
        j0 = n_points - max_tail_steps

    mask = np.zeros(n_points, dtype=bool)
    mask[j0:] = True

    if not np.any(mask):
        NMIN = min(1000, n_points)
        mask[-NMIN:] = True
        j0 = n_points - NMIN

    return mask, j0


# ===================================================================
# ============== Summary Text File Writer ===========================
# ===================================================================
def write_summary_txt(
    summary, run_folder, stem, dragt_log, bounce_results, drift_results,
    gyroperiods, norm_time, mass, cache_path,
    # Solver flags
    USE_PS, USE_RK4, USE_RK45, USE_RKG, PS_decimate,
    # Step sizes
    ps_step, rk4_step=None, rkg_step=None,
    # Energy drift arrays (already computed)
    rel_drift_ps=None, rel_drift_rk4=None, rel_drift_rk45=None, rel_drift_rkg=None,
    # Mu reference values
    mu0_ps=None, mu0_rk4=None, mu0_rk45=None, mu0_rkg=None,
    # Solver solutions (for mu tail computation)
    solution_rk4=None, solution_rkg=None, y_rk45_common=None,
    # PS storage info
    ps_store_stride=1,
    # npfloat type
    npfloat=np.float64,
):
    """
    Write the simulation summary text file, including tail-averaged energy
    and mu errors, Dragt diagnostics, and bounce/drift statistics.
    """

    # --- Tail fraction setup ---
    if gyroperiods < 1e6:
        TAIL_FRAC = 0.01
    else:
        TAIL_FRAC = 0.0001

    tail_start = (1.0 - TAIL_FRAC) * npfloat(norm_time)
    MAX_TAIL_STEPS = 500000

    # --- Build tail masks ---
    j0_ps = j0_rk4 = j0_rk45 = j0_rkg = 0

    if USE_PS:
        step_ps = ps_store_stride * ps_step
        _, j0_ps = _make_tail_mask(rel_drift_ps.size, step_ps, tail_start, MAX_TAIL_STEPS)

    if USE_RK45:
        _, j0_rk45 = _make_tail_mask(len(rel_drift_rk45), ps_step, tail_start, MAX_TAIL_STEPS)

    if USE_RK4:
        _, j0_rk4 = _make_tail_mask(len(rel_drift_rk4), rk4_step, tail_start, MAX_TAIL_STEPS)

    if USE_RKG:
        _, j0_rkg = _make_tail_mask(len(rel_drift_rkg), rkg_step, tail_start, MAX_TAIL_STEPS)

    # --- Write file ---
    output_filename = build_figure_filename(summary, run_folder, stem,
                                            figure_tag="simulation_summary", ext="txt")

    with open(output_filename, "w") as f:
        f.write("=== Simulation Summary ===\n")
        write_dict(f, summary)
        f.write("\n")

        # --- Energy tail errors ---
        f.write("\n=== |delta E|/E0 (tail average) ===\n")
        if USE_RK45:
            summarize_error("RK45", rel_drift_rk45[j0_rk45:], f)
        if USE_RK4:
            summarize_error("RK4", rel_drift_rk4[j0_rk4:], f)
        if USE_RKG:
            summarize_error("RKG", rel_drift_rkg[j0_rkg:], f)
        if USE_PS:
            summarize_error("PS", rel_drift_ps[j0_ps:], f)

        # --- Mu tail errors ---
        f.write("\n=== |delta mu|/mu0 (tail average) ===\n")

        if USE_RK45:
            y_tail = y_rk45_common[:, j0_rk45:]
            mu_tail = compute_mu_rk(y_tail.T, mass)
            summarize_error("RK45", np.abs(mu_tail - mu0_rk45) / mu0_rk45, f)
            del y_tail, mu_tail
            gc.collect()

        if USE_RK4:
            y_tail = solution_rk4[:, j0_rk4:]
            mu_tail = compute_mu_rk(y_tail.T, mass)
            summarize_error("RK4", np.abs(mu_tail - mu0_rk4) / mu0_rk4, f)
            del y_tail, mu_tail
            gc.collect()

        if USE_RKG:
            r_tail = solution_rkg[j0_rkg:, 0:3]
            p_tail = solution_rkg[j0_rkg:, 3:6]

            A_tail = np.empty_like(r_tail)
            for i in range(len(r_tail)):
                A_tail[i] = vector_potential_dipole(r_tail[i])

            v_tail = p_tail - A_tail
            state_tail = np.hstack((r_tail, v_tail))

            mu_tail = compute_mu_rk(state_tail, mass)
            summarize_error("RKG", np.abs(mu_tail - mu0_rkg) / mu0_rkg, f)
            del r_tail, p_tail, A_tail, v_tail, state_tail, mu_tail
            gc.collect()

        if USE_PS:
            step_ps_store = ps_store_stride * ps_step

            with h5py.File(cache_path, "r") as ps_h5:
                ps_y = ps_h5["ps"]["y"]
                n_store = ps_y.shape[1]

                j0 = int(tail_start / step_ps_store)
                j0 = max(0, min(j0, n_store - 1))

                if n_store - j0 > MAX_TAIL_STEPS:
                    j0 = n_store - MAX_TAIL_STEPS

                y_tail = expand_h5_to_full(ps_y[:, j0:])

            mu_tail = compute_mu_ps(y_tail, mass)
            summarize_error("PS", np.abs(mu_tail - mu0_ps) / mu0_ps, f)
            del y_tail, mu_tail
            gc.collect()

        # --- Dragt diagnostics ---
        if dragt_log["L_eff"] is not None:
            f.write("\n=== Dragt Diagnostics ===\n")
            f.write(f"Dragt L-shell           : {dragt_log['L_eff']:.4f} R_E\n")
            f.write(f"W0^2                    : {dragt_log['W0_sq']:.8f}\n")
            f.write(f"Boundary status         : {dragt_log['boundary']}\n")
            f.write(f"mu^2 (sin^2 alpha_eq)   : {dragt_log['mu_sq']:.6f}\n")
            f.write(f"Orbit character          : {dragt_log['orbit_character']}\n")
            f.write(f"Adiabaticity (initial)  : {dragt_log['eps_initial']:.4f}\n")
            f.write(f"Adiabaticity (mean)     : {dragt_log['eps_mean']:.4f}\n")
            f.write(f"Adiabaticity (max)      : {dragt_log['eps_max']:.4f}\n")
            if dragt_log["hit_atmosphere"]:
                f.write(f"Atmosphere flag         : HIT (r_min = {dragt_log['hit_atm_r']:.4f} R_E)\n")
            else:
                f.write(f"Atmosphere flag         : CLEAR\n")

        # --- Bounce & drift ---
        if USE_PS:
            f.write("\n=== Bounce and Drift Motion ===\n")

            if bounce_results is None or bounce_results.get("full_mean_s") is None:
                f.write("Bounce: not detected / insufficient mirror crossings\n")
            else:
                f.write(f"Mirror crossings        : {bounce_results['n_crossings']}\n")
                f.write(f"Bounce period (s)       : {bounce_results['full_mean_s']:.6g}\n")
                f.write(f"Bounce frequency (Hz)   : {bounce_results['frequency_hz']:.6g}\n")

            if drift_results is None or drift_results.get("period_s") is None:
                f.write("Drift: not enough azimuthal phase to estimate\n")
            else:
                direction = drift_results.get("direction", 0)
                dir_str = "eastward" if direction > 0 else "westward"

                f.write(f"Drift period (s)        : {drift_results['period_s']:.6g}\n")
                f.write(f"Drift direction         : {dir_str}\n")

            f.write("\n")


# ===================================================================
# ============== Master CSV Log Writer ==============================
# ===================================================================
def write_master_csv(
    output_folder, stem, particle_type,
    KE_particle, x_initial, y_initial, z_initial, pitch_deg, phi_deg,
    dragt_log,
    # Per-method data: list of (method_name, steps, dt, e_drift_arr, mu_drift_arr)
    method_records,
):
    """
    Build records and append to master_simulation_log.csv with duplicate detection.
    """
    records = []
    for method, steps, dt, e_drift, mu_drift in method_records:
        e = summarize(e_drift)
        mu = summarize(mu_drift)

        records.append({
            "run_id": stem,
            "particle": particle_type,
            "energy_keV": KE_particle,
            "x": x_initial,
            "y": y_initial,
            "z": z_initial,
            "pitch_deg": pitch_deg,
            "phi_deg": phi_deg,
            "L_eff": dragt_log["L_eff"],
            "eps_initial": dragt_log["eps_initial"],
            "eps_mean": dragt_log["eps_mean"],
            "eps_max": dragt_log["eps_max"],
            "W0_sq": dragt_log["W0_sq"],
            "boundary": dragt_log["boundary"],
            "mu_sq": dragt_log["mu_sq"],
            "orbit_character": dragt_log["orbit_character"],
            "hit_atmosphere": dragt_log["hit_atmosphere"],
            "hit_atm_r": dragt_log["hit_atm_r"],
            "steps": steps,
            "dt": dt,
            "method": method,
            "energy_mean_err": e["mean"],
            "energy_max_err": e["max"],
            "energy_rms_err": e["rms"],
            "mu_mean_err": mu["mean"],
            "mu_max_err": mu["max"],
            "mu_rms_err": mu["rms"],
        })

    df_new = pd.DataFrame(records)
    csv_path = f"{output_folder}/master_simulation_log.csv"
    dup_keys = ["energy_keV", "L_eff", "phi_deg", "pitch_deg", "particle", "method", "steps"]

    if os.path.exists(csv_path):
        df_existing = pd.read_csv(csv_path)
        for _, row in df_new.iterrows():
            mask = True
            for k in dup_keys:
                if row[k] is not None and k in df_existing.columns:
                    mask = mask & (df_existing[k] == row[k])
            df_existing = df_existing[~mask]
        df_out = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_out = df_new

    df_out.to_csv(csv_path, index=False)
