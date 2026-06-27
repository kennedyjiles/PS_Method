"""
Kinetic energy and P_phi conservation diagnostics for dipole trajectories.

    compute_ke_errors            — KE errors for all enabled solvers
    compute_pphi_errors          — P_phi errors for all enabled solvers

Internal helpers:
    _vector_potential_batch      — vectorized dipole A for an (N,3) position array
    log_spaced_indices           — log-spaced sample indices (plot-only decimation)
    _compute_energy_ps_chunked   — KE error from chunked PS h5 (uniform + log sampling)
    _pphi_from_xyz_v             — canonical P_phi from position/velocity
    _pphi_drift                  — relative (or absolute) P_phi drift
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
        external_rk4 = load_results_h5_func(external_h5_rk4)
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
        externalb = load_results_h5_func(external_h5_rk45)
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
    if USE_PS:
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
    # For RKG: convert momentum → velocity via v = p - charge_sign * A
    vector_potential_func=None,
    charge_sign=1,
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

    # --- PS: chunked from h5 to keep memory bounded ---
    # rel_pphi_ps = UNIFORM decimation -> summary tail (unchanged). The
    # (t_pphi_log, drift_pphi_log) pair is LOG-spaced -> plot only, collected
    # in the same h5 pass so the early decades of τ/T are visible.
    rel_pphi_ps = None
    t_pphi_log = drift_pphi_log = None
    P_phi_initial_ps = None
    ylabel_ps = None
    if USE_PS:
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
                err_dec.append(err[::energy_stride])           # uniform (summary)
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
        "P_phi_initial_ps":   P_phi_initial_ps,
        "P_phi_initial_rk4":  P_phi_initial_rk4,
        "P_phi_initial_rk45": P_phi_initial_rk45,
        "P_phi_initial_rkg":  P_phi_initial_rkg,
        "ylabel":     ylabel,
    }
