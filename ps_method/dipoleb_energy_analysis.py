"""
Kinetic energy and P_phi conservation diagnostics for dipole trajectories.

    compute_energy_ps_chunked    — KE error from h5 in chunks
    compute_ke_errors            — KE errors for all enabled solvers
    compute_pphi_error_chunked   — P_phi error from h5 in chunks
"""

import numpy as np
import h5py
from . import utils as ul


def compute_energy_ps_chunked(
    ps_y_h5,
    E0_ps,
    dt_ps_store,
    chunk_cols=200000,
    stride=1,
    dtype=None,
    return_plot_data=True,
):
    """
    Computes relative kinetic energy drift in a memory-efficient, chunked manner.
    Optionally returns decimated (stride-sampled) plot arrays only.
    """
    if dtype is None:
        dtype = ul.npfloat
    n_store = ps_y_h5.shape[1]

    if return_plot_data:
        # Estimate length of final array with stride
        n_points = (n_store + stride - 1) // stride
        t_plot = np.empty(n_points, dtype=ul.npfloat)
        drift_plot = np.empty(n_points, dtype=ul.npfloat)
        k = 0

    j_global = 0
    for j0 in range(0, n_store, chunk_cols):
        j1 = min(j0 + chunk_cols, n_store)
        v = ps_y_h5[3:6, j0:j1].astype(dtype, copy=False)

        E = 0.5 * np.sum(v * v, axis=0)
        rel = np.abs(E - E0_ps) / E0_ps

        if return_plot_data:
            for j_local in range(j1 - j0):
                if j_global % stride == 0:
                    t_plot[k] = j_global * dt_ps_store
                    drift_plot[k] = rel[j_local]
                    k += 1
                j_global += 1
        else:
            # If keeping full rel array
            raise NotImplementedError("Full array return not yet implemented in memory-saving mode")

    if return_plot_data:
        return t_plot[:k], drift_plot[:k]


# ===================================================================
# Compute Canonical Angular Momentum (in normalized units, not Dragt)
# ===================================================================

def compute_pphi_error_chunked(
    cache_path, y0, charge_sign, ps_step, time_factor,
    chunk_cols=1_000_000, max_plot_points=500_000,
):
    """
    Compute the relative (or absolute) error of canonical angular momentum
    P_phi from chunked PS h5 data.

    Parameters
    ----------
    cache_path : str
        Path to the PS h5 file.
    y0 : array_like
        Initial 17-element state vector (compact h5 rows expanded).
        Needs positions (0-2) and velocities (3-5).
    charge_sign : float
        +1 for proton, -1 for electron.
    ps_step : float
        PS step size (normalized time units).
    time_factor : float
        Conversion factor from normalized time to gyroperiods (1/T_gyro).
    chunk_cols : int
        Number of columns to read per h5 chunk.
    max_plot_points : int
        Target max points for the decimated output arrays.

    Returns
    -------
    dict with keys:
        "t_gyro"        : 1D array, time in gyroperiods (decimated)
        "rel_error_log" : 1D array, error with zeros replaced by 1e-16
        "max_err"       : float, global max error
        "P_phi_initial" : float, initial canonical angular momentum
        "ylabel"        : str, appropriate axis label
    """
    # --- Initial P_phi ---
    rho0  = np.sqrt(y0[0]**2 + y0[1]**2)
    r0    = np.sqrt(y0[0]**2 + y0[1]**2 + y0[2]**2)
    vphi0 = (y0[0]*y0[4] - y0[1]*y0[3]) / rho0
    P_phi_initial = (rho0 * vphi0) - charge_sign * (rho0**2 / r0**3)

    # --- Chunked read ---
    with h5py.File(cache_path, "r") as h5:
        ds = h5["ps"]["y"]
        N = ds.shape[1]
        dec = max(1, N // max_plot_points)
        err_dec = []
        max_err = 0.0

        for i0 in range(0, N, chunk_cols):
            i1 = min(i0 + chunk_cols, N)
            ch = ds[:6, i0:i1]
            rho = np.sqrt(ch[0]**2 + ch[1]**2)
            r   = np.sqrt(ch[0]**2 + ch[1]**2 + ch[2]**2)
            vp  = (ch[0]*ch[4] - ch[1]*ch[3]) / rho
            pp  = (rho * vp) - charge_sign * (rho**2 / r**3)

            if P_phi_initial == 0:
                err = np.abs(pp)
            else:
                err = np.abs((pp - P_phi_initial) / P_phi_initial)

            cm = float(np.max(err))
            if cm > max_err:
                max_err = cm

            err_dec.append(err[::dec])
            del ch, rho, r, vp, pp, err

    rel_error = np.concatenate(err_dec)
    rel_error_log = np.where(rel_error == 0, 1e-16, rel_error)
    t_gyro = ps_step * np.arange(len(rel_error_log), dtype=ul.npfloat) * dec * time_factor

    ylabel = (r"Absolute Error $|\Delta P_\phi|$" if P_phi_initial == 0
              else r"Relative Error $|(P_\phi - P_{\phi,0}) / P_{\phi,0}|$")

    return {
        "t_gyro":        t_gyro,
        "rel_error_log": rel_error_log,
        "max_err":       max_err,
        "P_phi_initial": P_phi_initial,
        "ylabel":        ylabel,
    }


# ------------------------------------------------------------------
#  compute_ke_errors  — kinetic-energy relative error for all solvers
# ------------------------------------------------------------------
def compute_ke_errors(
    T_gyro, n_ps=None,
    MAX_PLOT_POINTS=1_000_000,
    # PS
    USE_PS=False, cache_path=None, ps_step=None, PS_decimate=1, E0_ps=None,
    # RK4
    USE_RK4=False, solution_rk4=None, rk4_step=None,
    # RKG
    USE_RKG=False, solution_rkg=None, rkg_step=None,
    # RK45
    USE_RK45=False, y_rk45_common=None,
    # External h5 overlay files
    USE_EXTERNAL_H5_ps=False,  external_h5_ps=None,
    USE_EXTERNAL_H5_rk4=False, external_h5_rk4=None,
    USE_EXTERNAL_H5_rk45=False, external_h5_rk45=None,
    USE_EXTERNAL_H5_rkg=False,  external_h5_rkg=None,
    # Needed for RKG Hamiltonian energy
    vector_potential_func=None,
    # For loading external rk4/rk45 files
    load_results_h5_func=None,
):
    """Compute KE relative error arrays for every enabled solver.

    Returns a dict with keys:
        time_factor, energy_stride,
        rel_drift_ps, rel_drift_rk4, rel_drift_rk45, rel_drift_rkg,
        ke_ps, ke_rk4, ke_rk45, ke_rkg,           (plot-ready tuples or None)
        ke_ext_ps, ke_ext_rk4, ke_ext_rk45, ke_ext_rkg  (external overlay tuples or None)
    """
    time_factor = 1.0 / T_gyro

    energy_stride = 1
    if USE_PS and n_ps is not None:
        energy_stride = max(1, n_ps // MAX_PLOT_POINTS)

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
            energy_stride_ext = max(1, n_store // MAX_PLOT_POINTS)
            idx = np.arange(0, n_store, energy_stride_ext)
            t_eval_ps_ext = idx * dt_store_ext
            vxe = y_ext[3, ::energy_stride_ext].astype(np.float64)
            vye = y_ext[4, ::energy_stride_ext].astype(np.float64)
            vze = y_ext[5, ::energy_stride_ext].astype(np.float64)
            E_ext = 0.5 * (vxe*vxe + vye*vye + vze*vze)
            rel_drift_ps_ext = (E_ext - E_ext[0]) / E_ext[0]
            PS_order_ext = ext_ps.attrs.get("max_ps", None)
        ke_ext_ps = (t_eval_ps_ext, rel_drift_ps_ext, PS_order_ext)

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
        energy_stride_ext = max(1, n_store // MAX_PLOT_POINTS)
        idx = np.arange(0, n_store, energy_stride_ext)
        t_eval_rk45_ext = t_ext[idx]
        v = y_rk45_ext[3:6, idx].astype(np.float64)
        E = 0.5 * np.sum(v*v, axis=0)
        rel_drift_rk45_ext = (E - E[0]) / E[0]
        ke_ext_rk45 = (t_eval_rk45_ext, rel_drift_rk45_ext)

    if USE_EXTERNAL_H5_rkg:
        with h5py.File(external_h5_rkg, 'r') as external_file:
            ext_rkg = external_file["rkg"]
            y_dataset = ext_rkg["y"]
            is_transposed = (y_dataset.shape[0] == 6)
            n_steps = y_dataset.shape[1] if is_transposed else y_dataset.shape[0]
            rkg_stride = max(1, n_steps // MAX_PLOT_POINTS)
            if "t" in ext_rkg:
                t_ext_rkg = ext_rkg["t"][::rkg_stride]
            else:
                import json as _json
                dt_rkg = ext_rkg.attrs.get("dt", ext_rkg.get("dt", None))
                if dt_rkg is None and "params_json" in external_file.attrs:
                    params = _json.loads(external_file.attrs["params_json"])
                    dt_rkg = params.get("rkg_step")
                if dt_rkg is not None:
                    if hasattr(dt_rkg, 'value'): dt_rkg = dt_rkg[()]
                    idx = np.arange(0, n_steps, rkg_stride)
                    t_ext_rkg = dt_rkg * idx.astype(ul.npfloat)
                else:
                    raise ValueError("External RKG H5 file has no time info.")
            if is_transposed:
                y_ext_rkg = y_dataset[:, ::rkg_stride].T
            else:
                y_ext_rkg = y_dataset[::rkg_stride, :]

        r_rkg_ext = y_ext_rkg[:, 0:3]
        p_rkg_ext = y_ext_rkg[:, 3:6]
        A_rkg_ext = np.zeros_like(r_rkg_ext)
        for i in range(len(r_rkg_ext)):
            A_rkg_ext[i] = vector_potential_func(r_rkg_ext[i])
        v_rkg_ext = p_rkg_ext - A_rkg_ext
        E_rkg_ext = ul.npfloat(0.5) * np.sum(v_rkg_ext**2, axis=1, dtype=ul.npfloat)
        rel_drift_ext_rkg = np.abs(E_rkg_ext - E_rkg_ext[0]) / E_rkg_ext[0]
        ke_ext_rkg = (t_ext_rkg, rel_drift_ext_rkg)

    # --- Current-run PS energy (chunked from h5) ---
    rel_drift_ps = t_ps_plot = None
    if USE_PS:
        with h5py.File(cache_path, "r") as ps_h5:
            ps_y_h5 = ps_h5["ps"]["y"]
            t_ps_plot, rel_drift_ps = compute_energy_ps_chunked(
                ps_y_h5=ps_y_h5,
                E0_ps=E0_ps,
                dt_ps_store=ps_step * (PS_decimate if PS_decimate > 1 else 1),
                chunk_cols=MAX_PLOT_POINTS,
                stride=energy_stride,
                return_plot_data=True,
            )

    # --- Current-run RKG (Hamiltonian) ---
    rel_drift_rkg = None
    if USE_RKG:
        r_rkg = solution_rkg[:, 0:3]
        p_rkg = solution_rkg[:, 3:6]
        A_rkg = np.zeros_like(r_rkg)
        for i in range(len(r_rkg)):
            A_rkg[i] = vector_potential_func(r_rkg[i])
        v_rkg = p_rkg - A_rkg
        E_rkg = ul.npfloat(0.5) * np.sum(v_rkg**2, axis=1, dtype=ul.npfloat)
        E_rkg_0 = E_rkg[0]
        rel_drift_rkg = np.abs(E_rkg - E_rkg_0) / E_rkg_0

    # --- Current-run RK45 ---
    rel_drift_rk45 = None
    if USE_RK45:
        v_rk45 = y_rk45_common[3:6]
        E_rk45 = 0.5 * np.sum(v_rk45**2, axis=0)
        E_rk45_0 = E_rk45[0]
        rel_drift_rk45 = np.abs(E_rk45 - E_rk45_0) / E_rk45_0

    # --- Current-run RK4 ---
    rel_drift_rk4 = None
    if USE_RK4:
        v_rk4 = solution_rk4[3:6]
        E_rk4 = ul.npfloat(0.5) * np.sum(v_rk4**2, axis=0, dtype=ul.npfloat)
        E_rk4_0 = E_rk4[0]
        rel_drift_rk4 = np.abs(E_rk4 - E_rk4_0) / E_rk4_0

    # --- Build time arrays ---
    t_rk4 = rk4_step * np.arange(len(rel_drift_rk4), dtype=ul.npfloat) if USE_RK4 else None
    t_rkg = rkg_step * np.arange(len(rel_drift_rkg), dtype=ul.npfloat) if USE_RKG else None
    t_rk45 = ps_step * np.arange(len(rel_drift_rk45), dtype=ul.npfloat) if USE_RK45 else None

    # --- Assemble plot-ready tuples ---
    ke_ps   = (t_ps_plot, rel_drift_ps)    if USE_PS   else None
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
