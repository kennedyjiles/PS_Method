"""
Kinetic energy, P_phi, and mu conservation diagnostics for dipole trajectories.

    compute_ke_errors            — KE errors for all enabled solvers
    compute_pphi_errors          — P_phi errors for all enabled solvers
    compute_mu_errors            — full-run mu errors for all enabled solvers

Internal helpers:
    _vector_potential_batch      — vectorized dipole A for an (N,3) position array
    log_spaced_indices           — log-spaced sample indices (plot-only decimation)
    _compute_energy_ps_chunked   — KE error from chunked PS h5 (uniform + log sampling)
    _pphi_from_xyz_v             — canonical P_phi from position/velocity
    _pphi_drift                  — relative (or absolute) P_phi drift
    _mu_from_xyz_v               — normalized mu from position/velocity (B from position)
    _mu_drift                    — relative (or absolute) mu drift
"""

import numpy as np
import h5py
from . import utils as ul


def _vector_potential_batch(R):
    """Vectorized dipole vector potential for an (N, 3) array of positions.

    A = (y/r³, -x/r³, 0). Equivalent to calling
    dipoleb_physics.vector_potential row-by-row, but in one numpy pass — the
    per-point Python loop was a major stall once a decimated RKG array reaches
    millions of points.
    """
    R = np.asarray(R)
    x = R[:, 0]; y = R[:, 1]; z = R[:, 2]
    r2 = x * x + y * y + z * z
    r3 = r2 * np.sqrt(r2)
    A = np.zeros_like(R, dtype=ul.npfloat)
    nz = r3 != 0
    A[nz, 0] =  y[nz] / r3[nz]
    A[nz, 1] = -x[nz] / r3[nz]
    # A[:, 2] stays 0
    return A


def log_spaced_indices(n, max_points):
    """Sorted global indices sampled ~uniformly in log(index).

    Used for PLOTTING ONLY. Uniform-stride decimation collapses the early
    decades on a log time axis (the first kept point after 0 lands at
    τ/T ≈ stride·dt, so the curve appears to "start" partway across). Sampling
    uniformly in log(index) keeps a roughly constant number of points per
    decade, so the beginning of the run is resolved as well as the end.

    Always includes index 0 and n-1; returns ``np.arange(n)`` when the data
    already fits under ``max_points``.
    """
    n = int(n)
    if n <= max_points:
        return np.arange(n, dtype=np.int64)
    lg = np.logspace(0.0, np.log10(n - 1), int(max_points))
    idx = np.unique(np.round(lg).astype(np.int64))
    idx = np.clip(idx, 1, n - 1)
    return np.unique(np.concatenate(([0], idx)))


def _compute_energy_ps_chunked(
    ps_y_h5,
    e0_ps,
    dt_ps_store,
    chunk_cols=200000,
    stride=1,
    log_indices=None,
    dtype=None,
):
    """
    Compute |E - E_0|/E_0 from a chunked h5 PS dataset, decimated by `stride`.

    Memory-efficient: reads h5 in chunks of `chunk_cols` columns at a time,
    computes the relative KE drift for that chunk, and writes the
    stride-aligned points into pre-allocated output arrays.

    The uniform `stride` decimation is the canonical output used by the
    summary tail statistics (so those numbers are unchanged). If `log_indices`
    is given (a sorted index array from ``log_spaced_indices``), a SECOND
    log-spaced sampling is collected in the SAME pass — used only for the plot
    so the early decades are visible — and returned alongside.

    Returns
    -------
    t_plot, drift_plot                      (when log_indices is None)
    t_plot, drift_plot, t_log, drift_log    (when log_indices is given)
        Uniform-strided and (optionally) log-spaced time / relative-drift.
    """
    if dtype is None:
        dtype = ul.npfloat
    n_store = ps_y_h5.shape[1]

    # Pre-allocate uniform (canonical) output (upper bound on size)
    n_points = (n_store + stride - 1) // stride
    t_plot = np.empty(n_points, dtype=ul.npfloat)
    drift_plot = np.empty(n_points, dtype=ul.npfloat)
    k = 0

    want_log = log_indices is not None
    if want_log:
        log_indices = np.asarray(log_indices, dtype=np.int64)
        t_log = np.empty(len(log_indices), dtype=ul.npfloat)
        drift_log = np.empty(len(log_indices), dtype=ul.npfloat)
        kl = 0

    for j0 in range(0, n_store, chunk_cols):
        j1 = min(j0 + chunk_cols, n_store)
        v = ps_y_h5[3:6, j0:j1].astype(dtype, copy=False)

        E = 0.5 * np.sum(v * v, axis=0)
        rel = np.abs(E - e0_ps) / e0_ps

        # Vectorized stride decimation: pick global indices in [j0, j1) that are
        # multiples of stride. Equivalent to the per-step `j_global % stride == 0`
        # check but done in one numpy slice instead of a Python loop.
        first_aligned = ((j0 + stride - 1) // stride) * stride
        if first_aligned < j1:
            aligned_global = np.arange(first_aligned, j1, stride)
            n_pts = len(aligned_global)
            t_plot[k:k+n_pts]    = aligned_global * dt_ps_store
            drift_plot[k:k+n_pts] = rel[aligned_global - j0]
            k += n_pts

        # Log-spaced sampling for the plot (same chunk, no extra h5 read).
        if want_log:
            lo = np.searchsorted(log_indices, j0, side="left")
            hi = np.searchsorted(log_indices, j1, side="left")
            sel = log_indices[lo:hi]
            n_l = len(sel)
            if n_l:
                t_log[kl:kl+n_l]     = sel * dt_ps_store
                drift_log[kl:kl+n_l] = rel[sel - j0]
                kl += n_l

    if want_log:
        return t_plot[:k], drift_plot[:k], t_log[:kl], drift_log[:kl]
    return t_plot[:k], drift_plot[:k]



# ------------------------------------------------------------------
#  compute_ps_invariants_fused — ONE h5 pass for KE + P_phi + mu
# ------------------------------------------------------------------
def compute_ps_invariants_fused(
    cache_path, e0_ps, ps_y_initial, ps_step, ps_decimate,
    max_plot_points, energy_stride, charge_sign, dtype=None,
):
    """Stream the PS trajectory ONCE and derive all three invariant errors.

    compute_ke_errors, compute_pphi_errors and compute_mu_errors each used to
    open `cache_path` and walk the whole VDS themselves — three full passes
    over the same bytes (measured: 7.7 + 7.9 + 8.5 s per cec255 segment, i.e.
    ~18 h across 2715 segments, vs ~8 h fused). They all chunk identically and
    differ only in the per-chunk kernel, so one read feeds all three.

    Decimation semantics are preserved EXACTLY as each function had them, so
    every downstream number is unchanged:
      - KE   uniform output is aligned on the GLOBAL index (as
        _compute_energy_ps_chunked did).
      - P_phi / mu uniform outputs restart the stride at each chunk boundary
        (as their own loops did). That is inconsistent with KE and only
        matches it when chunk_cols % energy_stride == 0, but it feeds the
        summary tail statistics, so it is left alone here rather than
        silently changing published numbers.
    Plots use the log-spaced series, which is global-indexed for all three.

    Returns a dict of per-quantity dicts, or None if there is nothing to read.
    """
    if dtype is None:
        dtype = ul.npfloat

    y0 = ps_y_initial
    rho0 = np.sqrt(y0[0]**2 + y0[1]**2)
    r0   = np.sqrt(y0[0]**2 + y0[1]**2 + y0[2]**2)
    vphi0 = (y0[0]*y0[4] - y0[1]*y0[3]) / rho0
    P_phi_initial = float((rho0 * vphi0) - charge_sign * (rho0**2 / r0**3))
    mu_initial = float(_mu_from_xyz_v(
        np.asarray([y0[0]]), np.asarray([y0[1]]), np.asarray([y0[2]]),
        np.asarray([y0[3]]), np.asarray([y0[4]]), np.asarray([y0[5]]))[0])

    dt_ps_store = ps_step * (ps_decimate if ps_decimate > 1 else 1)
    stride = energy_stride
    chunk_cols = max_plot_points

    with h5py.File(cache_path, "r") as h5:
        ds = h5["ps"]["y"]
        N = ds.shape[1]
        log_idx = log_spaced_indices(N, max_plot_points)

        # KE uniform: pre-allocated, global-index aligned (KE reader's scheme).
        n_ke = (N + stride - 1) // stride
        ke_unif = np.empty(n_ke, dtype=ul.npfloat)
        k_ke = 0
        # P_phi / mu uniform: per-chunk stride restart (their original scheme).
        pp_dec, mu_dec = [], []
        # Log-spaced (plot) series, shared index bookkeeping.
        ke_log, pp_log, mu_log, t_log = [], [], [], []

        for i0 in range(0, N, chunk_cols):
            i1 = min(i0 + chunk_cols, N)
            ch = ds[:6, i0:i1]

            v = ch[3:6].astype(dtype, copy=False)
            E = 0.5 * np.sum(v * v, axis=0)
            err_ke = np.abs(E - e0_ps) / e0_ps

            pp = _pphi_from_xyz_v(ch[0], ch[1], ch[2], ch[3], ch[4], ch[5], charge_sign)
            err_pp = _pphi_drift(P_phi_initial, pp)

            mu = _mu_from_xyz_v(ch[0], ch[1], ch[2], ch[3], ch[4], ch[5])
            err_mu = _mu_drift(mu_initial, mu)

            # --- uniform decimation ---
            first_aligned = ((i0 + stride - 1) // stride) * stride
            if first_aligned < i1:
                aligned = np.arange(first_aligned, i1, stride)
                n_a = len(aligned)
                ke_unif[k_ke:k_ke + n_a] = err_ke[aligned - i0]
                k_ke += n_a
            # .copy(): a strided view would pin the whole chunk via .base.
            pp_dec.append(err_pp[::stride].copy())
            mu_dec.append(err_mu[::stride].copy())

            # --- log-spaced sampling for the plots ---
            lo = np.searchsorted(log_idx, i0, side="left")
            hi = np.searchsorted(log_idx, i1, side="left")
            sel = log_idx[lo:hi]
            if len(sel):
                loc = sel - i0
                ke_log.append(err_ke[loc])
                pp_log.append(err_pp[loc])
                mu_log.append(err_mu[loc])
                t_log.append(sel)

            del ch, v, E, err_ke, pp, err_pp, mu, err_mu

    t_log_arr = (np.concatenate(t_log).astype(ul.npfloat) * dt_ps_store
                 if t_log else np.empty(0, dtype=ul.npfloat))

    def _cat(parts):
        return np.concatenate(parts) if parts else np.empty(0, dtype=ul.npfloat)

    return {
        "ke":   {"uniform": ke_unif[:k_ke],
                 "t_log": t_log_arr, "drift_log": _cat(ke_log)},
        "pphi": {"uniform": _cat(pp_dec), "initial": P_phi_initial,
                 "t_log": t_log_arr, "drift_log": _cat(pp_log),
                 "ylabel": (r"$|\Delta P_\phi|$" if P_phi_initial == 0
                            else r"$|\Delta P_\phi|/|P_{\phi,0}|$")},
        "mu":   {"uniform": _cat(mu_dec), "initial": mu_initial,
                 "t_log": t_log_arr, "drift_log": _cat(mu_log),
                 "ylabel": (r"$|\Delta \mu_n|$" if mu_initial == 0
                            else r"$|\Delta \mu_n|/\mu_0$")},
    }


# ------------------------------------------------------------------
#  compute_ke_errors  — kinetic-energy relative error for all solvers
# ------------------------------------------------------------------
def compute_ke_errors(
    T_gyro, n_ps=None,
    max_plot_points=1_000_000,
    # PS
    USE_PS=False, cache_path=None, ps_step=None, ps_decimate=1, e0_ps=None,
    # RK4
    USE_RK4=False, solution_rk4=None, rk4_step=None, rk4_y_initial=None,
    # RKG
    USE_RKG=False, solution_rkg=None, rkg_step=None, rkg_y_initial=None,
    # RK45
    USE_RK45=False, y_rk45_common=None, rk45_y_initial=None,
    # External h5 overlay files
    USE_EXTERNAL_H5_ps=False,  external_h5_ps=None,
    USE_EXTERNAL_H5_rk4=False, external_h5_rk4=None,
    USE_EXTERNAL_H5_rk45=False, external_h5_rk45=None,
    USE_EXTERNAL_H5_rkg=False,  external_h5_rkg=None,
    # Needed for RKG Hamiltonian energy (canonical p → v requires charge sign:
    # v = p - charge_sign * A, matching hamiltonian_rhs)
    vector_potential_func=None,
    charge_sign=1,
    # For loading external rk4/rk45 files
    load_results_h5_func=None,
    ps_fused=None,
):
    """Compute KE relative error arrays for every enabled solver.

    Note: rel_drift_ps is decimated by `energy_stride` (the chunked PS reader
    decimates inline to save memory). rel_drift_rk4/rk45/rkg are full-length.

    Returns a dict with keys:
        time_factor, energy_stride,
        rel_drift_ps, rel_drift_rk4, rel_drift_rk45, rel_drift_rkg,
        ke_ps, ke_rk4, ke_rk45, ke_rkg,           (plot-ready tuples or None)
        ke_ext_ps, ke_ext_rk4, ke_ext_rk45, ke_ext_rkg  (external overlay tuples or None)
    """
    time_factor = 1.0 / T_gyro

    energy_stride = 1
    if USE_PS and n_ps is not None:
        energy_stride = max(1, n_ps // max_plot_points)

    # --- External H5 overlays ---
    ke_ext_ps = ke_ext_rk4 = ke_ext_rk45 = ke_ext_rkg = None

    if USE_EXTERNAL_H5_ps:
        with h5py.File(external_h5_ps, 'r') as external:
            ext_ps = external["ps"]
            y_ext = ext_ps["y"]
            n_store = y_ext.shape[1]
            ps_step_ext = ext_ps.attrs["dt"]
            ps_decimate_ext = ext_ps.attrs.get("decimate", 1)
            dt_store_ext = ps_step_ext * ps_decimate_ext
            energy_stride_ext = max(1, n_store // max_plot_points)
            idx = np.arange(0, n_store, energy_stride_ext)
            t_eval_ps_ext = idx * dt_store_ext
            vxe = y_ext[3, ::energy_stride_ext].astype(ul.npfloat)
            vye = y_ext[4, ::energy_stride_ext].astype(ul.npfloat)
            vze = y_ext[5, ::energy_stride_ext].astype(ul.npfloat)
            E_ext = 0.5 * (vxe*vxe + vye*vye + vze*vze)
            rel_drift_ps_ext = np.abs(E_ext - E_ext[0]) / E_ext[0]
            ps_order_ext = ext_ps.attrs.get("max_ps", None)
        ke_ext_ps = (t_eval_ps_ext, rel_drift_ps_ext, ps_order_ext)

    if USE_EXTERNAL_H5_rk4:
        external_rk4 = load_results_h5_func(external_h5_rk4, groups=("rk4",))
        ext_rk4 = external_rk4["rk4"]
        y_rk4_ext = ext_rk4["y"]
        if "t" in ext_rk4 and ext_rk4["t"] is not None:
            t_eval_rk4_ext = ext_rk4["t"]
        elif "dt" in ext_rk4 and "steps" in ext_rk4:
            t_eval_rk4_ext = ext_rk4["dt"] * np.arange(ext_rk4["steps"] + 1, dtype=ul.npfloat)
        else:
            raise ValueError(
                "External RK4 H5 file has no time information "
                "(no 't', no 'dt/steps').")
        if y_rk4_ext.shape[0] != 6:
            y_rk4_ext = y_rk4_ext.T
        v_rk4_ext = y_rk4_ext[3:6]
        E_rk4_ext = 0.5 * np.sum(v_rk4_ext**2, axis=0)
        rel_drift_rk4_ext = np.abs(E_rk4_ext - E_rk4_ext[0]) / E_rk4_ext[0]
        ke_ext_rk4 = (t_eval_rk4_ext, rel_drift_rk4_ext)

    if USE_EXTERNAL_H5_rk45:
        externalb = load_results_h5_func(external_h5_rk45, groups=("rk45",))
        ext_rk45 = externalb["rk45"]
        y_rk45_ext = ext_rk45["y"]
        if y_rk45_ext.shape[0] != 6:
            y_rk45_ext = y_rk45_ext.T
        n_store = y_rk45_ext.shape[1]
        if "t" in ext_rk45 and ext_rk45["t"] is not None:
            t_ext = np.asarray(ext_rk45["t"])
        else:
            ps_step_ext = ext_rk45.get("dt", ps_step)
            ps_decimate_ext = ext_rk45.get("decimate", 1)
            dt_store_ext = ps_step_ext * ps_decimate_ext
            t_ext = dt_store_ext * np.arange(n_store, dtype=ul.npfloat)
        # Log-spaced (not uniform stride) so the early τ/T decades show on the
        # log-x plot. y_rk45_ext is already in memory, so this is a cheap index.
        idx = log_spaced_indices(n_store, max_plot_points)
        t_eval_rk45_ext = t_ext[idx]
        v = y_rk45_ext[3:6, idx].astype(ul.npfloat)
        E = 0.5 * np.sum(v*v, axis=0)
        rel_drift_rk45_ext = np.abs(E - E[0]) / E[0]
        ke_ext_rk45 = (t_eval_rk45_ext, rel_drift_rk45_ext)

    if USE_EXTERNAL_H5_rkg:
        # Large external RKG files (10s–100s of GB) can't be subsampled by
        # scattered/strided access without h5py thrashing chunks. Instead make
        # ONE sequential pass: read each block once, compute energy (vectorized),
        # and keep the log-spaced indices that fall in it — skipping blocks that
        # contain none. Log spacing keeps the early τ/T decades on the plot.
        with h5py.File(external_h5_rkg, 'r') as external_file:
            ext_rkg = external_file["rkg"]
            y_dataset = ext_rkg["y"]
            is_transposed = (y_dataset.shape[0] == 6)
            n_steps = y_dataset.shape[1] if is_transposed else y_dataset.shape[0]

            t_source = ext_rkg["t"] if "t" in ext_rkg else None
            dt_rkg = None
            if t_source is None:
                import json as _json
                dt_rkg = ext_rkg.attrs.get("dt", ext_rkg.get("dt", None))
                if dt_rkg is None and "params_json" in external_file.attrs:
                    params = _json.loads(external_file.attrs["params_json"])
                    dt_rkg = params.get("rkg_step")
                if dt_rkg is None:
                    raise ValueError("External RKG H5 file has no time info.")
                if hasattr(dt_rkg, 'value'):
                    dt_rkg = dt_rkg[()]

            keep = log_spaced_indices(n_steps, max_plot_points)
            block = max(max_plot_points, 1)

            E_parts, t_parts = [], []
            E0 = None
            for j0 in range(0, n_steps, block):
                j1 = min(j0 + block, n_steps)
                lo = np.searchsorted(keep, j0, side="left")
                hi = np.searchsorted(keep, j1, side="left")
                # First block always read (sets E0); later empty blocks skipped.
                if hi == lo and E0 is not None:
                    continue
                blk = (y_dataset[:, j0:j1].T if is_transposed
                       else y_dataset[j0:j1, :])
                A = _vector_potential_batch(blk[:, 0:3])
                v = blk[:, 3:6] - charge_sign * A
                E = ul.npfloat(0.5) * np.sum(v * v, axis=1, dtype=ul.npfloat)
                if E0 is None:
                    E0 = E[0]
                sel = keep[lo:hi]
                if len(sel):
                    E_parts.append(E[sel - j0])
                    if t_source is not None:
                        t_parts.append(t_source[j0:j1][sel - j0])
                    else:
                        t_parts.append(sel.astype(ul.npfloat) * dt_rkg)
                del blk, A, v, E

        E_sel = np.concatenate(E_parts)
        rel_drift_ext_rkg = np.abs(E_sel - E0) / E0
        t_ext_rkg = np.concatenate(t_parts)
        ke_ext_rkg = (t_ext_rkg, rel_drift_ext_rkg)

    # --- Current-run PS energy (chunked from h5) ---
    # rel_drift_ps = UNIFORM decimation -> feeds the summary tail (unchanged).
    # (t_ps_log, drift_ps_log) = LOG-spaced -> plot only, so the early decades
    # of τ/T are visible. Both are collected in one h5 pass.
    rel_drift_ps = None
    t_ps_log = drift_ps_log = None
    if USE_PS and ps_fused is not None:
        # PS series already produced by compute_ps_invariants_fused — skip this
        # function's own full pass over the VDS.
        rel_drift_ps = ps_fused["ke"]["uniform"]
        t_ps_log     = ps_fused["ke"]["t_log"]
        drift_ps_log = ps_fused["ke"]["drift_log"]
    elif USE_PS:
        with h5py.File(cache_path, "r") as ps_h5:
            ps_y_h5 = ps_h5["ps"]["y"]
            log_idx = log_spaced_indices(ps_y_h5.shape[1], max_plot_points)
            _, rel_drift_ps, t_ps_log, drift_ps_log = _compute_energy_ps_chunked(
                ps_y_h5=ps_y_h5,
                e0_ps=e0_ps,
                dt_ps_store=ps_step * (ps_decimate if ps_decimate > 1 else 1),
                chunk_cols=max_plot_points,
                stride=energy_stride,
                log_indices=log_idx,
            )

    # --- Current-run RKG (Hamiltonian) ---
    # For trimmed files, *_y_initial preserves the source IC so E0 reflects
    # the true initial energy rather than the energy at trim-start.
    rel_drift_rkg = None
    if USE_RKG:
        r_rkg = solution_rkg[:, 0:3]
        p_rkg = solution_rkg[:, 3:6]
        A_rkg = _vector_potential_batch(r_rkg)
        # v = p - charge_sign * A (matches hamiltonian_rhs)
        v_rkg = p_rkg - charge_sign * A_rkg
        E_rkg = ul.npfloat(0.5) * np.sum(v_rkg**2, axis=1, dtype=ul.npfloat)
        if rkg_y_initial is not None:
            v0 = rkg_y_initial[3:6] - charge_sign * vector_potential_func(rkg_y_initial[0:3])
            E_rkg_0 = ul.npfloat(0.5) * ul.npfloat(np.sum(v0 * v0))
        else:
            E_rkg_0 = E_rkg[0]
        rel_drift_rkg = np.abs(E_rkg - E_rkg_0) / E_rkg_0

    # --- Current-run RK45 ---
    rel_drift_rk45 = None
    if USE_RK45:
        v_rk45 = y_rk45_common[3:6]
        E_rk45 = 0.5 * np.sum(v_rk45**2, axis=0)
        if rk45_y_initial is not None:
            v0 = rk45_y_initial[3:6]
            E_rk45_0 = 0.5 * float(np.sum(v0 * v0))
        else:
            E_rk45_0 = E_rk45[0]
        rel_drift_rk45 = np.abs(E_rk45 - E_rk45_0) / E_rk45_0

    # --- Current-run RK4 ---
    rel_drift_rk4 = None
    if USE_RK4:
        v_rk4 = solution_rk4[3:6]
        E_rk4 = ul.npfloat(0.5) * np.sum(v_rk4**2, axis=0, dtype=ul.npfloat)
        if rk4_y_initial is not None:
            v0 = rk4_y_initial[3:6]
            E_rk4_0 = ul.npfloat(0.5) * ul.npfloat(np.sum(v0 * v0))
        else:
            E_rk4_0 = E_rk4[0]
        rel_drift_rk4 = np.abs(E_rk4 - E_rk4_0) / E_rk4_0

    # --- Build time arrays ---
    t_rk4 = rk4_step * np.arange(len(rel_drift_rk4), dtype=ul.npfloat) if USE_RK4 else None
    t_rkg = rkg_step * np.arange(len(rel_drift_rkg), dtype=ul.npfloat) if USE_RKG else None
    t_rk45 = ps_step * np.arange(len(rel_drift_rk45), dtype=ul.npfloat) if USE_RK45 else None

    # --- Assemble plot-ready tuples ---
    # Plot uses the LOG-spaced arrays; the summary uses uniform rel_drift_ps.
    ke_ps   = (t_ps_log, drift_ps_log)     if USE_PS   else None
    ke_rk4  = (t_rk4, rel_drift_rk4)      if USE_RK4  else None
    ke_rk45 = (t_rk45, rel_drift_rk45)    if USE_RK45 else None
    ke_rkg  = (t_rkg, rel_drift_rkg)       if USE_RKG  else None

    return {
        "time_factor":    time_factor,
        "energy_stride":  energy_stride,
        # raw arrays (needed by summary writer + CSV log)
        "rel_drift_ps":   rel_drift_ps,
        "rel_drift_rk4":  rel_drift_rk4,
        "rel_drift_rk45": rel_drift_rk45,
        "rel_drift_rkg":  rel_drift_rkg,
        # plot-ready tuples
        "ke_ps":   ke_ps,
        "ke_rk4":  ke_rk4,
        "ke_rk45": ke_rk45,
        "ke_rkg":  ke_rkg,
        # external overlays
        "ke_ext_ps":   ke_ext_ps,
        "ke_ext_rk4":  ke_ext_rk4,
        "ke_ext_rk45": ke_ext_rk45,
        "ke_ext_rkg":  ke_ext_rkg,
    }


# ------------------------------------------------------------------
#  compute_pphi_errors  — canonical angular momentum error for all solvers
# ------------------------------------------------------------------
def _pphi_from_xyz_v(x, y, z, vx, vy, vz, charge_sign):
    """Canonical P_phi = rho*v_phi - cs * rho^2/r^3, given positions and velocities."""
    rho = np.sqrt(x*x + y*y)
    r   = np.sqrt(x*x + y*y + z*z)
    # v_phi = (x*vy - y*vx)/rho  (azimuthal component)
    v_phi = (x*vy - y*vx) / rho
    return (rho * v_phi) - charge_sign * (rho * rho) / (r * r * r)


def _pphi_drift(P_phi_initial, P_phi_array):
    """Relative drift (or absolute, when P_phi_initial is zero)."""
    if P_phi_initial == 0:
        return np.abs(P_phi_array)
    return np.abs((P_phi_array - P_phi_initial) / P_phi_initial)


def compute_pphi_errors(
    T_gyro, n_ps=None,
    max_plot_points=1_000_000,
    # PS
    USE_PS=False, cache_path=None, ps_step=None, ps_decimate=1, ps_y_initial=None,
    # RK4
    USE_RK4=False, solution_rk4=None, rk4_step=None, rk4_y_initial=None,
    # RKG
    USE_RKG=False, solution_rkg=None, rkg_step=None, rkg_y_initial=None,
    # RK45
    USE_RK45=False, y_rk45_common=None, rk45_y_initial=None,
    # External h5 overlay files (mirror compute_ke_errors)
    USE_EXTERNAL_H5_ps=False,  external_h5_ps=None,
    USE_EXTERNAL_H5_rk4=False, external_h5_rk4=None,
    USE_EXTERNAL_H5_rk45=False, external_h5_rk45=None,
    USE_EXTERNAL_H5_rkg=False,  external_h5_rkg=None,
    # For RKG: convert momentum → velocity via v = p - charge_sign * A
    vector_potential_func=None,
    charge_sign=1,
    # For loading external rk4/rk45 files
    load_results_h5_func=None,
    ps_fused=None,
):
    """Compute |ΔP_φ|/|P_{φ,0}| arrays for every enabled solver.

    Same pattern as compute_ke_errors — returns a dict with plot-ready
    tuples (t_array, drift_array) for each solver, plus the time_factor
    and the initial P_phi values.

    For RK4 / RK45 / PS: state is (x,y,z, vx,vy,vz) — P_phi computed directly.
    For RKG: state is (x,y,z, px,py,pz) — convert to velocity first with
    v = p - charge_sign * A(r), matching the Hamiltonian convention used
    in compute_ke_errors.

    Returns dict with keys:
        time_factor, energy_stride,
        rel_pphi_ps, rel_pphi_rk4, rel_pphi_rk45, rel_pphi_rkg,
        pphi_ps, pphi_rk4, pphi_rk45, pphi_rkg,   (plot-ready tuples or None)
        P_phi_initial_ps, P_phi_initial_rk4, P_phi_initial_rk45, P_phi_initial_rkg
        ylabel_ps, ylabel_rk4, ylabel_rk45, ylabel_rkg
    """
    time_factor = 1.0 / T_gyro

    energy_stride = 1
    if USE_PS and n_ps is not None:
        energy_stride = max(1, n_ps // max_plot_points)

    # --- External H5 overlays (mirror compute_ke_errors) ---
    # Each external file is read the SAME way as in compute_ke_errors:
    #   PS   -> strided read straight off the h5 dataset
    #   RK4  -> full load of the rk4 group only (see load_results_h5_dipoleb)
    #   RK45 -> full load, then log-spaced index (early τ/T decades visible)
    #   RKG  -> ONE sequential block pass (10s–100s GB safe), momenta→velocity
    # The external initial P_φ is taken from the file's own first sample, so an
    # external-only run needs no local IC.
    pphi_ext_ps = pphi_ext_rk4 = pphi_ext_rk45 = pphi_ext_rkg = None

    if USE_EXTERNAL_H5_ps:
        with h5py.File(external_h5_ps, 'r') as external:
            ext_ps = external["ps"]
            y_ext = ext_ps["y"]
            n_store = y_ext.shape[1]
            ps_step_ext = ext_ps.attrs["dt"]
            ps_decimate_ext = ext_ps.attrs.get("decimate", 1)
            dt_store_ext = ps_step_ext * ps_decimate_ext
            stride_ext = max(1, n_store // max_plot_points)
            idx = np.arange(0, n_store, stride_ext)
            t_eval_ps_ext = idx * dt_store_ext
            xe  = y_ext[0, ::stride_ext].astype(ul.npfloat)
            yye = y_ext[1, ::stride_ext].astype(ul.npfloat)
            ze  = y_ext[2, ::stride_ext].astype(ul.npfloat)
            vxe = y_ext[3, ::stride_ext].astype(ul.npfloat)
            vye = y_ext[4, ::stride_ext].astype(ul.npfloat)
            vze = y_ext[5, ::stride_ext].astype(ul.npfloat)
            pp_ext = _pphi_from_xyz_v(xe, yye, ze, vxe, vye, vze, charge_sign)
            rel_pphi_ps_ext = _pphi_drift(float(pp_ext[0]), pp_ext)
            ps_order_ext = ext_ps.attrs.get("max_ps", None)
        pphi_ext_ps = (t_eval_ps_ext, rel_pphi_ps_ext, ps_order_ext)

    if USE_EXTERNAL_H5_rk4:
        external_rk4 = load_results_h5_func(external_h5_rk4, groups=("rk4",))
        ext_rk4 = external_rk4["rk4"]
        y_rk4_ext = np.asarray(ext_rk4["y"])
        if "t" in ext_rk4 and ext_rk4["t"] is not None:
            t_eval_rk4_ext = np.asarray(ext_rk4["t"])
        elif "dt" in ext_rk4 and "steps" in ext_rk4:
            t_eval_rk4_ext = ext_rk4["dt"] * np.arange(ext_rk4["steps"] + 1, dtype=ul.npfloat)
        else:
            raise ValueError(
                "External RK4 H5 file has no time information "
                "(no 't', no 'dt/steps').")
        if y_rk4_ext.shape[0] != 6:
            y_rk4_ext = y_rk4_ext.T
        pp_ext = _pphi_from_xyz_v(y_rk4_ext[0], y_rk4_ext[1], y_rk4_ext[2],
                                  y_rk4_ext[3], y_rk4_ext[4], y_rk4_ext[5], charge_sign)
        rel_pphi_rk4_ext = _pphi_drift(float(pp_ext[0]), pp_ext)
        pphi_ext_rk4 = (t_eval_rk4_ext, rel_pphi_rk4_ext)

    if USE_EXTERNAL_H5_rk45:
        externalb = load_results_h5_func(external_h5_rk45, groups=("rk45",))
        ext_rk45 = externalb["rk45"]
        y_rk45_ext = np.asarray(ext_rk45["y"])
        if y_rk45_ext.shape[0] != 6:
            y_rk45_ext = y_rk45_ext.T
        n_store = y_rk45_ext.shape[1]
        if "t" in ext_rk45 and ext_rk45["t"] is not None:
            t_ext = np.asarray(ext_rk45["t"])
        else:
            ps_step_ext = ext_rk45.get("dt", ps_step)
            ps_decimate_ext = ext_rk45.get("decimate", 1)
            dt_store_ext = ps_step_ext * ps_decimate_ext
            t_ext = dt_store_ext * np.arange(n_store, dtype=ul.npfloat)
        # Log-spaced (not uniform stride) so the early τ/T decades show.
        idx = log_spaced_indices(n_store, max_plot_points)
        t_eval_rk45_ext = t_ext[idx]
        s = y_rk45_ext[:, idx].astype(ul.npfloat)
        pp_ext = _pphi_from_xyz_v(s[0], s[1], s[2], s[3], s[4], s[5], charge_sign)
        rel_pphi_rk45_ext = _pphi_drift(float(pp_ext[0]), pp_ext)
        pphi_ext_rk45 = (t_eval_rk45_ext, rel_pphi_rk45_ext)

    if USE_EXTERNAL_H5_rkg:
        # ONE sequential pass: read each block once, convert momenta→velocity
        # (v = p - charge_sign*A), compute P_φ, and keep only the log-spaced
        # indices that fall in the block — skipping blocks that contain none.
        with h5py.File(external_h5_rkg, 'r') as external_file:
            ext_rkg = external_file["rkg"]
            y_dataset = ext_rkg["y"]
            is_transposed = (y_dataset.shape[0] == 6)
            n_steps = y_dataset.shape[1] if is_transposed else y_dataset.shape[0]

            t_source = ext_rkg["t"] if "t" in ext_rkg else None
            dt_rkg = None
            if t_source is None:
                import json as _json
                dt_rkg = ext_rkg.attrs.get("dt", ext_rkg.get("dt", None))
                if dt_rkg is None and "params_json" in external_file.attrs:
                    params = _json.loads(external_file.attrs["params_json"])
                    dt_rkg = params.get("rkg_step")
                if dt_rkg is None:
                    raise ValueError("External RKG H5 file has no time info.")
                if hasattr(dt_rkg, 'value'):
                    dt_rkg = dt_rkg[()]

            keep = log_spaced_indices(n_steps, max_plot_points)
            block = max(max_plot_points, 1)

            pp_parts, t_parts = [], []
            pp0 = None
            for j0 in range(0, n_steps, block):
                j1 = min(j0 + block, n_steps)
                lo = np.searchsorted(keep, j0, side="left")
                hi = np.searchsorted(keep, j1, side="left")
                # First block always read (sets pp0); later empty blocks skipped.
                if hi == lo and pp0 is not None:
                    continue
                blk = (y_dataset[:, j0:j1].T if is_transposed
                       else y_dataset[j0:j1, :])
                A = _vector_potential_batch(blk[:, 0:3])
                v = blk[:, 3:6] - charge_sign * A
                pp = _pphi_from_xyz_v(blk[:, 0], blk[:, 1], blk[:, 2],
                                      v[:, 0], v[:, 1], v[:, 2], charge_sign)
                if pp0 is None:
                    pp0 = float(pp[0])
                sel = keep[lo:hi]
                if len(sel):
                    pp_parts.append(pp[sel - j0])
                    if t_source is not None:
                        t_parts.append(t_source[j0:j1][sel - j0])
                    else:
                        t_parts.append(sel.astype(ul.npfloat) * dt_rkg)
                del blk, A, v, pp

        pp_sel = np.concatenate(pp_parts)
        rel_pphi_rkg_ext = _pphi_drift(pp0, pp_sel)
        t_ext_rkg = np.concatenate(t_parts)
        pphi_ext_rkg = (t_ext_rkg, rel_pphi_rkg_ext)

    # --- PS: chunked from h5 to keep memory bounded ---
    # rel_pphi_ps = UNIFORM decimation -> summary tail (unchanged). The
    # (t_pphi_log, drift_pphi_log) pair is LOG-spaced -> plot only, collected
    # in the same h5 pass so the early decades of τ/T are visible.
    rel_pphi_ps = None
    t_pphi_log = drift_pphi_log = None
    P_phi_initial_ps = None
    ylabel_ps = None
    if USE_PS and ps_fused is not None:
        _f = ps_fused["pphi"]
        rel_pphi_ps      = _f["uniform"]
        t_pphi_log       = _f["t_log"]
        drift_pphi_log   = _f["drift_log"]
        P_phi_initial_ps = _f["initial"]
        ylabel_ps        = _f["ylabel"]
    elif USE_PS:
        y0 = ps_y_initial
        rho0  = np.sqrt(y0[0]**2 + y0[1]**2)
        r0    = np.sqrt(y0[0]**2 + y0[1]**2 + y0[2]**2)
        vphi0 = (y0[0]*y0[4] - y0[1]*y0[3]) / rho0
        P_phi_initial_ps = float((rho0 * vphi0) - charge_sign * (rho0**2 / r0**3))

        chunk_cols = max_plot_points
        dt_ps_store = ps_step * (ps_decimate if ps_decimate > 1 else 1)
        with h5py.File(cache_path, "r") as h5:
            ds = h5["ps"]["y"]
            N = ds.shape[1]
            log_idx = log_spaced_indices(N, max_plot_points)
            err_dec = []
            err_log = []
            t_log = []
            for i0 in range(0, N, chunk_cols):
                i1 = min(i0 + chunk_cols, N)
                ch = ds[:6, i0:i1]
                pp = _pphi_from_xyz_v(ch[0], ch[1], ch[2], ch[3], ch[4], ch[5], charge_sign)
                err = _pphi_drift(P_phi_initial_ps, pp)
                # .copy() is REQUIRED: err[::stride] is a view that would pin the
                # whole chunk's `err` array alive via .base, so del err frees
                # nothing and memory grows unbounded over a large VDS (the KE path
                # avoids this by writing into a pre-allocated buffer). Copy the
                # tiny decimated slice so each chunk is released each iteration.
                err_dec.append(err[::energy_stride].copy())    # uniform (summary)
                lo = np.searchsorted(log_idx, i0, side="left")  # log (plot)
                hi = np.searchsorted(log_idx, i1, side="left")
                sel = log_idx[lo:hi]
                if len(sel):
                    err_log.append(err[sel - i0])
                    t_log.append(sel)
                del ch, pp, err
        rel_pphi_ps = np.concatenate(err_dec)
        drift_pphi_log = np.concatenate(err_log)
        t_pphi_log = np.concatenate(t_log).astype(ul.npfloat) * dt_ps_store
        ylabel_ps = (r"$|\Delta P_\phi|$" if P_phi_initial_ps == 0
                     else r"$|\Delta P_\phi|/|P_{\phi,0}|$")

    # --- RK4 ---
    rel_pphi_rk4 = None; P_phi_initial_rk4 = None; ylabel_rk4 = None
    if USE_RK4:
        rk4 = solution_rk4
        pp_rk4 = _pphi_from_xyz_v(rk4[0], rk4[1], rk4[2],
                                   rk4[3], rk4[4], rk4[5], charge_sign)
        if rk4_y_initial is not None:
            y0 = rk4_y_initial
            rho0 = np.sqrt(y0[0]**2 + y0[1]**2)
            r0   = np.sqrt(y0[0]**2 + y0[1]**2 + y0[2]**2)
            vphi0 = (y0[0]*y0[4] - y0[1]*y0[3]) / rho0
            P_phi_initial_rk4 = float((rho0*vphi0) - charge_sign*(rho0**2/r0**3))
        else:
            P_phi_initial_rk4 = float(pp_rk4[0])
        rel_pphi_rk4 = _pphi_drift(P_phi_initial_rk4, pp_rk4)
        ylabel_rk4 = (r"$|\Delta P_\phi|$" if P_phi_initial_rk4 == 0
                      else r"$|\Delta P_\phi|/|P_{\phi,0}|$")

    # --- RK45 (state in y_rk45_common, shape (6, N)) ---
    rel_pphi_rk45 = None; P_phi_initial_rk45 = None; ylabel_rk45 = None
    if USE_RK45:
        rk = y_rk45_common
        pp_rk = _pphi_from_xyz_v(rk[0], rk[1], rk[2],
                                  rk[3], rk[4], rk[5], charge_sign)
        if rk45_y_initial is not None:
            y0 = rk45_y_initial
            rho0 = np.sqrt(y0[0]**2 + y0[1]**2)
            r0   = np.sqrt(y0[0]**2 + y0[1]**2 + y0[2]**2)
            vphi0 = (y0[0]*y0[4] - y0[1]*y0[3]) / rho0
            P_phi_initial_rk45 = float((rho0*vphi0) - charge_sign*(rho0**2/r0**3))
        else:
            P_phi_initial_rk45 = float(pp_rk[0])
        rel_pphi_rk45 = _pphi_drift(P_phi_initial_rk45, pp_rk)
        ylabel_rk45 = (r"$|\Delta P_\phi|$" if P_phi_initial_rk45 == 0
                       else r"$|\Delta P_\phi|/|P_{\phi,0}|$")

    # --- RKG (state in solution_rkg, shape (N, 6); col 3:6 are MOMENTA) ---
    rel_pphi_rkg = None; P_phi_initial_rkg = None; ylabel_rkg = None
    if USE_RKG:
        r_rkg = solution_rkg[:, 0:3]
        p_rkg = solution_rkg[:, 3:6]
        # v = p - charge_sign * A(r)  (matches energy convention)
        A_rkg = _vector_potential_batch(r_rkg)
        v_rkg = p_rkg - charge_sign * A_rkg
        pp_rkg = _pphi_from_xyz_v(r_rkg[:, 0], r_rkg[:, 1], r_rkg[:, 2],
                                   v_rkg[:, 0], v_rkg[:, 1], v_rkg[:, 2], charge_sign)
        if rkg_y_initial is not None:
            r0v = rkg_y_initial[0:3]
            p0  = rkg_y_initial[3:6]
            A0  = vector_potential_func(r0v)
            v0  = p0 - charge_sign * A0
            rho0 = np.sqrt(r0v[0]**2 + r0v[1]**2)
            r0   = np.sqrt(r0v[0]**2 + r0v[1]**2 + r0v[2]**2)
            vphi0 = (r0v[0]*v0[1] - r0v[1]*v0[0]) / rho0
            P_phi_initial_rkg = float((rho0*vphi0) - charge_sign*(rho0**2/r0**3))
        else:
            P_phi_initial_rkg = float(pp_rkg[0])
        rel_pphi_rkg = _pphi_drift(P_phi_initial_rkg, pp_rkg)
        ylabel_rkg = (r"$|\Delta P_\phi|$" if P_phi_initial_rkg == 0
                      else r"$|\Delta P_\phi|/|P_{\phi,0}|$")

    # --- Time arrays for non-PS solvers ---
    t_rk4  = (rk4_step  * np.arange(len(rel_pphi_rk4),  dtype=ul.npfloat)) if USE_RK4  else None
    t_rkg  = (rkg_step  * np.arange(len(rel_pphi_rkg),  dtype=ul.npfloat)) if USE_RKG  else None
    # RK45 was stored at PS-step cadence in the existing code path
    t_rk45 = (ps_step   * np.arange(len(rel_pphi_rk45), dtype=ul.npfloat)) if USE_RK45 else None

    # --- Plot-ready tuples ---
    # Plot uses the LOG-spaced arrays; the summary uses uniform rel_pphi_ps.
    pphi_ps   = (t_pphi_log, drift_pphi_log) if USE_PS   else None
    pphi_rk4  = (t_rk4,     rel_pphi_rk4)  if USE_RK4  else None
    pphi_rk45 = (t_rk45,    rel_pphi_rk45) if USE_RK45 else None
    pphi_rkg  = (t_rkg,     rel_pphi_rkg)  if USE_RKG  else None

    # Use the first solver's ylabel as the shared one
    ylabel = next((y for y in (ylabel_ps, ylabel_rk4, ylabel_rk45, ylabel_rkg)
                   if y is not None), r"$|\Delta P_\phi|/|P_{\phi,0}|$")

    return {
        "time_factor":   time_factor,
        "energy_stride": energy_stride,
        "rel_pphi_ps":   rel_pphi_ps,
        "rel_pphi_rk4":  rel_pphi_rk4,
        "rel_pphi_rk45": rel_pphi_rk45,
        "rel_pphi_rkg":  rel_pphi_rkg,
        "pphi_ps":   pphi_ps,
        "pphi_rk4":  pphi_rk4,
        "pphi_rk45": pphi_rk45,
        "pphi_rkg":  pphi_rkg,
        # external overlays
        "pphi_ext_ps":   pphi_ext_ps,
        "pphi_ext_rk4":  pphi_ext_rk4,
        "pphi_ext_rk45": pphi_ext_rk45,
        "pphi_ext_rkg":  pphi_ext_rkg,
        "P_phi_initial_ps":   P_phi_initial_ps,
        "P_phi_initial_rk4":  P_phi_initial_rk4,
        "P_phi_initial_rk45": P_phi_initial_rk45,
        "P_phi_initial_rkg":  P_phi_initial_rkg,
        "ylabel":     ylabel,
    }


# ------------------------------------------------------------------
#  compute_mu_errors  — magnetic moment error for all solvers
# ------------------------------------------------------------------
def _mu_from_xyz_v(x, y, z, vx, vy, vz):
    """Normalized magnetic moment mu = v_perp^2 / (2|B|) from state arrays.

    B is recomputed from position (same formulas as lorentz_force), so the
    reader stays h5-layout-agnostic — works whether the cache stores the
    17-row full layout or the 9-row compact one (rows 0-5 are always state).
    Uses the identity |v_perp|^2 = |v|^2 - (v.B)^2/B^2 (vectorized version of
    dipoleb_moment_analysis.compute_mu_rk). mu is in NORMALIZED units — every
    consumer uses it as a baseline-relative drift, so constants cancel.
    """
    r2 = x*x + y*y + z*z
    with np.errstate(divide="ignore", invalid="ignore"):
        r5inv = np.where(r2 > 0, r2**(-2.5), 0.0)
    Bx = -3.0 * x * z * r5inv
    By = -3.0 * y * z * r5inv
    Bz = -(3.0 * z*z - r2) * r5inv

    B2      = Bx*Bx + By*By + Bz*Bz
    v_dot_B = vx*Bx + vy*By + vz*Bz
    v2      = vx*vx + vy*vy + vz*vz
    with np.errstate(divide="ignore", invalid="ignore"):
        v_perp2 = v2 - v_dot_B*v_dot_B / B2
        mu = np.where(B2 > 0, v_perp2 / (2.0 * np.sqrt(B2)), 0.0)
    return mu


def _mu_drift(mu_initial, mu_array):
    """Relative drift (or absolute, when mu_initial is zero)."""
    if mu_initial == 0:
        return np.abs(mu_array)
    return np.abs((mu_array - mu_initial) / mu_initial)


def compute_mu_errors(
    T_gyro, n_ps=None,
    max_plot_points=1_000_000,
    # PS
    USE_PS=False, cache_path=None, ps_step=None, ps_decimate=1, ps_y_initial=None,
    # RK4
    USE_RK4=False, solution_rk4=None, rk4_step=None, rk4_y_initial=None,
    # RKG
    USE_RKG=False, solution_rkg=None, rkg_step=None, rkg_y_initial=None,
    # RK45
    USE_RK45=False, y_rk45_common=None, rk45_y_initial=None,
    # External h5 overlay files (mirror compute_ke_errors)
    USE_EXTERNAL_H5_ps=False,  external_h5_ps=None,
    USE_EXTERNAL_H5_rk4=False, external_h5_rk4=None,
    USE_EXTERNAL_H5_rk45=False, external_h5_rk45=None,
    USE_EXTERNAL_H5_rkg=False,  external_h5_rkg=None,
    # For RKG: convert momentum → velocity via v = p - charge_sign * A
    vector_potential_func=None,
    charge_sign=1,
    # For loading external rk4/rk45 files
    load_results_h5_func=None,
    ps_fused=None,
):
    """Compute |Δμ_n|/μ_0 arrays for every enabled solver over the FULL run.

    Same pattern as compute_ke_errors / compute_pphi_errors — chunked h5
    reads keep memory bounded on large files, and each solver returns a
    plot-ready (t_array, drift_array) tuple. Unlike the windowed
    mu_deviation diagnostic (dipoleb_moment_analysis), this covers the
    whole run for the log-log conservation plot.

    For RK4 / RK45 / PS: state is (x,y,z, vx,vy,vz) — mu computed directly.
    For RKG: state is (x,y,z, px,py,pz) — convert to velocity first with
    v = p - charge_sign * A(r), matching the Hamiltonian convention used
    in compute_ke_errors.

    Returns dict with keys:
        time_factor, energy_stride,
        rel_mu_ps, rel_mu_rk4, rel_mu_rk45, rel_mu_rkg,
        mu_ps, mu_rk4, mu_rk45, mu_rkg,           (plot-ready tuples or None)
        mu_ext_ps, mu_ext_rk4, mu_ext_rk45, mu_ext_rkg,
        mu_initial_ps, mu_initial_rk4, mu_initial_rk45, mu_initial_rkg,
        ylabel
    """
    time_factor = 1.0 / T_gyro

    energy_stride = 1
    if USE_PS and n_ps is not None:
        energy_stride = max(1, n_ps // max_plot_points)

    # --- External H5 overlays (mirror compute_ke_errors / compute_pphi_errors) ---
    #   PS   -> strided read straight off the h5 dataset
    #   RK4  -> full load of the rk4 group only (see load_results_h5_dipoleb)
    #   RK45 -> full load, then log-spaced index (early τ/T decades visible)
    #   RKG  -> ONE sequential block pass (10s–100s GB safe), momenta→velocity
    # The external initial mu is taken from the file's own first sample, so an
    # external-only run needs no local IC.
    mu_ext_ps = mu_ext_rk4 = mu_ext_rk45 = mu_ext_rkg = None

    if USE_EXTERNAL_H5_ps:
        with h5py.File(external_h5_ps, 'r') as external:
            ext_ps = external["ps"]
            y_ext = ext_ps["y"]
            n_store = y_ext.shape[1]
            ps_step_ext = ext_ps.attrs["dt"]
            ps_decimate_ext = ext_ps.attrs.get("decimate", 1)
            dt_store_ext = ps_step_ext * ps_decimate_ext
            stride_ext = max(1, n_store // max_plot_points)
            idx = np.arange(0, n_store, stride_ext)
            t_eval_ps_ext = idx * dt_store_ext
            xe  = y_ext[0, ::stride_ext].astype(ul.npfloat)
            yye = y_ext[1, ::stride_ext].astype(ul.npfloat)
            ze  = y_ext[2, ::stride_ext].astype(ul.npfloat)
            vxe = y_ext[3, ::stride_ext].astype(ul.npfloat)
            vye = y_ext[4, ::stride_ext].astype(ul.npfloat)
            vze = y_ext[5, ::stride_ext].astype(ul.npfloat)
            mu_ext = _mu_from_xyz_v(xe, yye, ze, vxe, vye, vze)
            rel_mu_ps_ext = _mu_drift(float(mu_ext[0]), mu_ext)
            ps_order_ext = ext_ps.attrs.get("max_ps", None)
        mu_ext_ps = (t_eval_ps_ext, rel_mu_ps_ext, ps_order_ext)

    if USE_EXTERNAL_H5_rk4:
        external_rk4 = load_results_h5_func(external_h5_rk4, groups=("rk4",))
        ext_rk4 = external_rk4["rk4"]
        y_rk4_ext = np.asarray(ext_rk4["y"])
        if "t" in ext_rk4 and ext_rk4["t"] is not None:
            t_eval_rk4_ext = np.asarray(ext_rk4["t"])
        elif "dt" in ext_rk4 and "steps" in ext_rk4:
            t_eval_rk4_ext = ext_rk4["dt"] * np.arange(ext_rk4["steps"] + 1, dtype=ul.npfloat)
        else:
            raise ValueError(
                "External RK4 H5 file has no time information "
                "(no 't', no 'dt/steps').")
        if y_rk4_ext.shape[0] != 6:
            y_rk4_ext = y_rk4_ext.T
        mu_ext = _mu_from_xyz_v(y_rk4_ext[0], y_rk4_ext[1], y_rk4_ext[2],
                                y_rk4_ext[3], y_rk4_ext[4], y_rk4_ext[5])
        rel_mu_rk4_ext = _mu_drift(float(mu_ext[0]), mu_ext)
        mu_ext_rk4 = (t_eval_rk4_ext, rel_mu_rk4_ext)

    if USE_EXTERNAL_H5_rk45:
        externalb = load_results_h5_func(external_h5_rk45, groups=("rk45",))
        ext_rk45 = externalb["rk45"]
        y_rk45_ext = np.asarray(ext_rk45["y"])
        if y_rk45_ext.shape[0] != 6:
            y_rk45_ext = y_rk45_ext.T
        n_store = y_rk45_ext.shape[1]
        if "t" in ext_rk45 and ext_rk45["t"] is not None:
            t_ext = np.asarray(ext_rk45["t"])
        else:
            ps_step_ext = ext_rk45.get("dt", ps_step)
            ps_decimate_ext = ext_rk45.get("decimate", 1)
            dt_store_ext = ps_step_ext * ps_decimate_ext
            t_ext = dt_store_ext * np.arange(n_store, dtype=ul.npfloat)
        # Log-spaced (not uniform stride) so the early τ/T decades show.
        idx = log_spaced_indices(n_store, max_plot_points)
        t_eval_rk45_ext = t_ext[idx]
        s = y_rk45_ext[:, idx].astype(ul.npfloat)
        mu_ext = _mu_from_xyz_v(s[0], s[1], s[2], s[3], s[4], s[5])
        rel_mu_rk45_ext = _mu_drift(float(mu_ext[0]), mu_ext)
        mu_ext_rk45 = (t_eval_rk45_ext, rel_mu_rk45_ext)

    if USE_EXTERNAL_H5_rkg:
        # ONE sequential pass: read each block once, convert momenta→velocity
        # (v = p - charge_sign*A), compute mu, and keep only the log-spaced
        # indices that fall in the block — skipping blocks that contain none.
        with h5py.File(external_h5_rkg, 'r') as external_file:
            ext_rkg = external_file["rkg"]
            y_dataset = ext_rkg["y"]
            is_transposed = (y_dataset.shape[0] == 6)
            n_steps = y_dataset.shape[1] if is_transposed else y_dataset.shape[0]

            t_source = ext_rkg["t"] if "t" in ext_rkg else None
            dt_rkg = None
            if t_source is None:
                import json as _json
                dt_rkg = ext_rkg.attrs.get("dt", ext_rkg.get("dt", None))
                if dt_rkg is None and "params_json" in external_file.attrs:
                    params = _json.loads(external_file.attrs["params_json"])
                    dt_rkg = params.get("rkg_step")
                if dt_rkg is None:
                    raise ValueError("External RKG H5 file has no time info.")
                if hasattr(dt_rkg, 'value'):
                    dt_rkg = dt_rkg[()]

            keep = log_spaced_indices(n_steps, max_plot_points)
            block = max(max_plot_points, 1)

            mu_parts, t_parts = [], []
            mu0 = None
            for j0 in range(0, n_steps, block):
                j1 = min(j0 + block, n_steps)
                lo = np.searchsorted(keep, j0, side="left")
                hi = np.searchsorted(keep, j1, side="left")
                # First block always read (sets mu0); later empty blocks skipped.
                if hi == lo and mu0 is not None:
                    continue
                blk = (y_dataset[:, j0:j1].T if is_transposed
                       else y_dataset[j0:j1, :])
                A = _vector_potential_batch(blk[:, 0:3])
                v = blk[:, 3:6] - charge_sign * A
                mu = _mu_from_xyz_v(blk[:, 0], blk[:, 1], blk[:, 2],
                                    v[:, 0], v[:, 1], v[:, 2])
                if mu0 is None:
                    mu0 = float(mu[0])
                sel = keep[lo:hi]
                if len(sel):
                    mu_parts.append(mu[sel - j0])
                    if t_source is not None:
                        t_parts.append(t_source[j0:j1][sel - j0])
                    else:
                        t_parts.append(sel.astype(ul.npfloat) * dt_rkg)
                del blk, A, v, mu

        mu_sel = np.concatenate(mu_parts)
        rel_mu_rkg_ext = _mu_drift(mu0, mu_sel)
        t_ext_rkg = np.concatenate(t_parts)
        mu_ext_rkg = (t_ext_rkg, rel_mu_rkg_ext)

    # --- PS: chunked from h5 to keep memory bounded ---
    # rel_mu_ps = UNIFORM decimation -> summary-style tail stats. The
    # (t_mu_log, drift_mu_log) pair is LOG-spaced -> plot only, collected
    # in the same h5 pass so the early decades of τ/T are visible.
    rel_mu_ps = None
    t_mu_log = drift_mu_log = None
    mu_initial_ps = None
    ylabel_ps = None
    if USE_PS and ps_fused is not None:
        _f = ps_fused["mu"]
        rel_mu_ps     = _f["uniform"]
        t_mu_log      = _f["t_log"]
        drift_mu_log  = _f["drift_log"]
        mu_initial_ps = _f["initial"]
        ylabel_ps     = _f["ylabel"]
    elif USE_PS:
        y0 = ps_y_initial
        mu_initial_ps = float(_mu_from_xyz_v(
            np.asarray([y0[0]]), np.asarray([y0[1]]), np.asarray([y0[2]]),
            np.asarray([y0[3]]), np.asarray([y0[4]]), np.asarray([y0[5]]))[0])

        chunk_cols = max_plot_points
        dt_ps_store = ps_step * (ps_decimate if ps_decimate > 1 else 1)
        with h5py.File(cache_path, "r") as h5:
            ds = h5["ps"]["y"]
            N = ds.shape[1]
            log_idx = log_spaced_indices(N, max_plot_points)
            err_dec = []
            err_log = []
            t_log = []
            for i0 in range(0, N, chunk_cols):
                i1 = min(i0 + chunk_cols, N)
                ch = ds[:6, i0:i1]
                mu = _mu_from_xyz_v(ch[0], ch[1], ch[2], ch[3], ch[4], ch[5])
                err = _mu_drift(mu_initial_ps, mu)
                # .copy() is REQUIRED: err[::stride] is a view that would pin the
                # whole chunk's `err` array alive via .base, so del err frees
                # nothing and memory grows unbounded over a large VDS. Copy the
                # tiny decimated slice so each chunk is released each iteration.
                err_dec.append(err[::energy_stride].copy())    # uniform (summary)
                lo = np.searchsorted(log_idx, i0, side="left")  # log (plot)
                hi = np.searchsorted(log_idx, i1, side="left")
                sel = log_idx[lo:hi]
                if len(sel):
                    err_log.append(err[sel - i0])
                    t_log.append(sel)
                del ch, mu, err
        rel_mu_ps = np.concatenate(err_dec)
        drift_mu_log = np.concatenate(err_log)
        t_mu_log = np.concatenate(t_log).astype(ul.npfloat) * dt_ps_store
        ylabel_ps = (r"$|\Delta \mu_n|$" if mu_initial_ps == 0
                     else r"$|\Delta \mu_n|/\mu_0$")

    # --- RK4 ---
    rel_mu_rk4 = None; mu_initial_rk4 = None; ylabel_rk4 = None
    if USE_RK4:
        rk4 = solution_rk4
        mu_rk4_arr = _mu_from_xyz_v(rk4[0], rk4[1], rk4[2],
                                    rk4[3], rk4[4], rk4[5])
        if rk4_y_initial is not None:
            y0 = rk4_y_initial
            mu_initial_rk4 = float(_mu_from_xyz_v(
                np.asarray([y0[0]]), np.asarray([y0[1]]), np.asarray([y0[2]]),
                np.asarray([y0[3]]), np.asarray([y0[4]]), np.asarray([y0[5]]))[0])
        else:
            mu_initial_rk4 = float(mu_rk4_arr[0])
        rel_mu_rk4 = _mu_drift(mu_initial_rk4, mu_rk4_arr)
        ylabel_rk4 = (r"$|\Delta \mu_n|$" if mu_initial_rk4 == 0
                      else r"$|\Delta \mu_n|/\mu_0$")

    # --- RK45 (state in y_rk45_common, shape (6, N)) ---
    rel_mu_rk45 = None; mu_initial_rk45 = None; ylabel_rk45 = None
    if USE_RK45:
        rk = y_rk45_common
        mu_rk45_arr = _mu_from_xyz_v(rk[0], rk[1], rk[2],
                                     rk[3], rk[4], rk[5])
        if rk45_y_initial is not None:
            y0 = rk45_y_initial
            mu_initial_rk45 = float(_mu_from_xyz_v(
                np.asarray([y0[0]]), np.asarray([y0[1]]), np.asarray([y0[2]]),
                np.asarray([y0[3]]), np.asarray([y0[4]]), np.asarray([y0[5]]))[0])
        else:
            mu_initial_rk45 = float(mu_rk45_arr[0])
        rel_mu_rk45 = _mu_drift(mu_initial_rk45, mu_rk45_arr)
        ylabel_rk45 = (r"$|\Delta \mu_n|$" if mu_initial_rk45 == 0
                       else r"$|\Delta \mu_n|/\mu_0$")

    # --- RKG (state in solution_rkg, shape (N, 6); col 3:6 are MOMENTA) ---
    rel_mu_rkg = None; mu_initial_rkg = None; ylabel_rkg = None
    if USE_RKG:
        r_rkg = solution_rkg[:, 0:3]
        p_rkg = solution_rkg[:, 3:6]
        # v = p - charge_sign * A(r)  (matches energy convention)
        A_rkg = _vector_potential_batch(r_rkg)
        v_rkg = p_rkg - charge_sign * A_rkg
        mu_rkg_arr = _mu_from_xyz_v(r_rkg[:, 0], r_rkg[:, 1], r_rkg[:, 2],
                                    v_rkg[:, 0], v_rkg[:, 1], v_rkg[:, 2])
        if rkg_y_initial is not None:
            r0v = rkg_y_initial[0:3]
            p0  = rkg_y_initial[3:6]
            A0  = vector_potential_func(r0v)
            v0  = p0 - charge_sign * A0
            mu_initial_rkg = float(_mu_from_xyz_v(
                np.asarray([r0v[0]]), np.asarray([r0v[1]]), np.asarray([r0v[2]]),
                np.asarray([v0[0]]), np.asarray([v0[1]]), np.asarray([v0[2]]))[0])
        else:
            mu_initial_rkg = float(mu_rkg_arr[0])
        rel_mu_rkg = _mu_drift(mu_initial_rkg, mu_rkg_arr)
        ylabel_rkg = (r"$|\Delta \mu_n|$" if mu_initial_rkg == 0
                      else r"$|\Delta \mu_n|/\mu_0$")

    # --- Time arrays for non-PS solvers ---
    t_rk4  = (rk4_step  * np.arange(len(rel_mu_rk4),  dtype=ul.npfloat)) if USE_RK4  else None
    t_rkg  = (rkg_step  * np.arange(len(rel_mu_rkg),  dtype=ul.npfloat)) if USE_RKG  else None
    # RK45 was stored at PS-step cadence in the existing code path
    t_rk45 = (ps_step   * np.arange(len(rel_mu_rk45), dtype=ul.npfloat)) if USE_RK45 else None

    # --- Plot-ready tuples ---
    # Plot uses the LOG-spaced arrays; rel_mu_ps stays uniform (tail stats).
    mu_ps   = (t_mu_log, drift_mu_log) if USE_PS   else None
    mu_rk4  = (t_rk4,    rel_mu_rk4)   if USE_RK4  else None
    mu_rk45 = (t_rk45,   rel_mu_rk45)  if USE_RK45 else None
    mu_rkg  = (t_rkg,    rel_mu_rkg)   if USE_RKG  else None

    # Use the first solver's ylabel as the shared one
    ylabel = next((y for y in (ylabel_ps, ylabel_rk4, ylabel_rk45, ylabel_rkg)
                   if y is not None), r"$|\Delta \mu_n|/\mu_0$")

    return {
        "time_factor":   time_factor,
        "energy_stride": energy_stride,
        "rel_mu_ps":   rel_mu_ps,
        "rel_mu_rk4":  rel_mu_rk4,
        "rel_mu_rk45": rel_mu_rk45,
        "rel_mu_rkg":  rel_mu_rkg,
        "mu_ps":   mu_ps,
        "mu_rk4":  mu_rk4,
        "mu_rk45": mu_rk45,
        "mu_rkg":  mu_rkg,
        # external overlays
        "mu_ext_ps":   mu_ext_ps,
        "mu_ext_rk4":  mu_ext_rk4,
        "mu_ext_rk45": mu_ext_rk45,
        "mu_ext_rkg":  mu_ext_rkg,
        "mu_initial_ps":   mu_initial_ps,
        "mu_initial_rk4":  mu_initial_rk4,
        "mu_initial_rk45": mu_initial_rk45,
        "mu_initial_rkg":  mu_initial_rkg,
        "ylabel":     ylabel,
    }
