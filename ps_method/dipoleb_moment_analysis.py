"""
dipoleb_moment_analysis.py — Magnetic moment diagnostics for dipole trajectories.

    compute_mu_ps            — mu from PS solution array
    compute_mu_rk            — mu from RK solution array
    compute_mu_deviation_rk  — mu deviation over time (RK solvers)
    compute_mu_deviation_ps  — mu deviation over time (PS, chunked from h5)

Internal helpers:
    _gyro_window_indices     — index range for a gyro-window slice
"""

import numpy as np
import h5py
from ps_method.utils import npfloat, maybe_njit
from ps_method.writers import expand_h5_to_full
from ps_method.dipoleb_physics import vector_potential


@maybe_njit
def compute_mu_ps(solution_ps, mass):
    x, y, z = solution_ps[0], solution_ps[1], solution_ps[2]
    vx, vy, vz = solution_ps[3], solution_ps[4], solution_ps[5]
    Bx, By, Bz = solution_ps[14], solution_ps[15], solution_ps[16]

    mu = np.zeros_like(x)
    for i in range(len(x)):
        B = np.array([Bx[i], By[i], Bz[i]])
        B2 = np.dot(B, B)
        if B2 == 0:
            mu[i] = 0.0
            continue
        v = np.array([vx[i], vy[i], vz[i]])
        v_par = (np.dot(v, B) / B2) * B
        v_perp = v - v_par
        mu[i] = mass * np.dot(v_perp, v_perp) / (2 * np.sqrt(B2))
    return mu

@maybe_njit
def compute_mu_rk(solution_rk, mass):
    mu = np.zeros(len(solution_rk))
    for i in range(len(solution_rk)):
        x, y, z = solution_rk[i, 0:3]
        vx, vy, vz = solution_rk[i, 3:6]

        # Compute B at position
        r2 = x**2 + y**2 + z**2
        if r2 == 0:
            mu[i] = 0.0
            continue
        r5inv = r2**(-2.5)
        # Sign convention matches simulator (downward dipole moment, upward B at equator)
        B = np.array([
            -3 * x * z * r5inv,
            -3 * y * z * r5inv,
            -(3 * z**2 - r2) * r5inv
        ])

        B2 = np.dot(B, B)
        v = np.array([vx, vy, vz])
        v_par = (np.dot(v, B) / B2) * B
        v_perp = v - v_par
        mu[i] = mass * np.dot(v_perp, v_perp) / (2 * np.sqrt(B2))

    return mu


# ===================================================================
# ============ Gyro-window index helper =============================
# ===================================================================
def _gyro_window_indices(gyro_window, total_steps, window_steps):
    """Return (i0, i1) slice indices for first/last/all gyro window."""
    if gyro_window == "last":
        i1 = total_steps
        i0 = max(0, i1 - window_steps)
    elif gyro_window == "first":
        i0 = 0
        i1 = min(window_steps, total_steps)
    elif gyro_window == "all":
        i0 = 0
        i1 = total_steps
    else:
        raise ValueError("gyro_window must be 'first', 'last', or 'all'")
    return i0, i1


# ===================================================================
# ============ Mu deviation — RK solvers (in-memory) ================
# ===================================================================
def compute_mu_deviation_rk(
    solution, steps, dt, N_GYRO, N_STEPS_PER_GYRO,
    mass, gyro_window, time_factor,
    solver_type="rk4",
):
    """
    Compute magnetic moment deviation for an in-memory RK solver solution.

    Parameters
    ----------
    solution : ndarray
        For RK4/RK45: shape (6, N) — columns are time steps.
        For RKG: shape (N, 6) — rows are time steps (Hamiltonian format).
    steps : int
        Total number of integration steps.
    dt : float
        Step size (normalized time).
    N_GYRO : int
        Number of gyroperiods in the analysis window.
    N_STEPS_PER_GYRO : float
        Steps per gyroperiod for this solver.
    mass : float
        Relativistic mass (gamma * m_si).
    gyro_window : str
        "first", "last", or "all".
    time_factor : float
        Conversion from normalized time to gyroperiods (1/T_gyro).
    solver_type : str
        "rk4", "rk45", or "rkg". Controls data layout and mu computation.

    Returns
    -------
    dict with keys:
        "t"       : 1D time array in gyroperiods
        "mudrift" : 1D relative mu deviation array
        "mu0"     : float, initial magnetic moment
    """
    window_steps = int(round(N_GYRO * N_STEPS_PER_GYRO))
    i0, i1 = _gyro_window_indices(gyro_window, steps, window_steps)

    if solver_type == "rkg":
        # RKG stores (N, 6) with canonical momentum — need to convert to velocity
        r0 = solution[0, 0:3]
        p0 = solution[0, 3:6]
        A0 = vector_potential(r0)
        v0 = p0 - A0
        state0 = np.hstack((r0, v0))[None, :]
        mu0 = compute_mu_rk(state0, mass)[0]

        r_win = solution[i0:i1, 0:3]
        p_win = solution[i0:i1, 3:6]
        A_win = np.empty_like(r_win)
        for i in range(len(r_win)):
            A_win[i] = vector_potential(r_win[i])
        v_win = p_win - A_win
        state_win = np.hstack((r_win, v_win))
        mu_win = compute_mu_rk(state_win, mass)
    else:
        # RK4 and RK45: shape (6, N) — columns are time steps
        mu0 = compute_mu_rk(solution[:, 0:1].T, mass)[0]
        mu_win = compute_mu_rk(solution[:, i0:i1].T, mass)

    mudrift = np.abs(mu_win - mu0) / mu0
    t = (i0 + np.arange(mudrift.size, dtype=np.float64)) * dt * time_factor

    return {"t": t, "mudrift": mudrift, "mu0": mu0}


# ===================================================================
# ============ Mu deviation — PS (chunked h5) =======================
# ===================================================================
def compute_mu_deviation_ps(
    cache_path, steps_ps, ps_step, PS_decimate,
    N_GYRO, N_STEPS_PER_GYRO, mass, mu0_ps,
    gyro_window, time_factor,
    max_plot_points=1_000_000,
):
    """
    Compute magnetic moment deviation for PS data from chunked h5.

    Parameters
    ----------
    cache_path : str
        Path to the PS h5 file.
    steps_ps : int
        Total physical integration steps.
    ps_step : float
        PS step size (normalized time).
    PS_decimate : int
        Decimation factor used during streaming.
    N_GYRO : int
        Number of gyroperiods in the analysis window.
    N_STEPS_PER_GYRO : float
        Steps per gyroperiod (on the PS grid).
    mass : float
        Relativistic mass.
    mu0_ps : float
        Initial magnetic moment (PS).
    gyro_window : str
        "first", "last", or "all".
    time_factor : float
        Conversion from normalized time to gyroperiods.
    max_plot_points : int
        Cap on output array length for plotting.

    Returns
    -------
    dict with keys:
        "t"              : 1D time array in gyroperiods (decimated for plotting)
        "mudrift"        : 1D full mu deviation array
        "mudrift_plot"   : 1D decimated mu deviation for plotting
        "ps_order_label" : int, max PS order from h5 attrs
    """
    window_steps = N_GYRO * N_STEPS_PER_GYRO
    i0_phys, i1_phys = _gyro_window_indices(gyro_window, steps_ps, window_steps)

    ps_store_stride = PS_decimate if (PS_decimate > 1) else 1
    j0 = int(np.floor(i0_phys / ps_store_stride))
    j1 = int(np.ceil(i1_phys / ps_store_stride))

    with h5py.File(cache_path, "r") as ps_h5:
        ps_grp = ps_h5["ps"]
        ps_y = ps_grp["y"]
        ps_order_label = int(ps_grp.attrs["max_ps"])
        n_store = ps_y.shape[1]

        j0 = max(0, min(j0, n_store))
        j1 = max(0, min(j1, n_store))

        if j1 <= j0:
            raise RuntimeError("Empty PS mu window (chunked)")

        y_ps_win = expand_h5_to_full(ps_y[:, j0:j1])

    mu_ps = compute_mu_ps(y_ps_win, mass)
    mudrift = np.abs(mu_ps - mu0_ps) / mu0_ps

    dt_ps_store = ps_step * ps_store_stride
    t_store = np.arange(j0, j1, dtype=np.float64) * dt_ps_store
    moment_stride = max(1, round(len(mu_ps) // max_plot_points))
    t_plot = t_store[::moment_stride] * time_factor
    mudrift_plot = mudrift[::moment_stride]

    return {
        "t":              t_plot,
        "mudrift":        mudrift,
        "mudrift_plot":   mudrift_plot,
        "ps_order_label": ps_order_label,
    }
