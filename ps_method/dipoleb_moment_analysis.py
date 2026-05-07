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
from . import utils as ul
from . import writers as wr
from . import dipoleb_physics as dp


@ul.maybe_njit
def compute_mu_ps(solution_ps):
    # Uses the identity |v_perp|² = |v|² − (v·B)²/B² to avoid per-step
    # 3-vector allocations. Numerically equivalent to the v_par/v_perp
    # decomposition; saves N small allocations on long runs.
    #
    # Returns μ in NORMALIZED units: v_perp² / (2|B|), with v and B
    # already normalized by the dipoleb non-dimensionalization. The mass
    # factor that would convert this to SI J/T is left out — every
    # consumer uses μ as a baseline-relative drift, so the constant
    # cancels in (μ - μ₀)/μ₀.
    x = solution_ps[0]
    vx, vy, vz = solution_ps[3], solution_ps[4], solution_ps[5]
    Bx, By, Bz = solution_ps[14], solution_ps[15], solution_ps[16]

    mu = np.zeros_like(x)
    for i in range(len(x)):
        B2 = Bx[i]*Bx[i] + By[i]*By[i] + Bz[i]*Bz[i]
        if B2 == 0:
            mu[i] = 0.0
            continue
        v_dot_B = vx[i]*Bx[i] + vy[i]*By[i] + vz[i]*Bz[i]
        v2      = vx[i]*vx[i] + vy[i]*vy[i] + vz[i]*vz[i]
        v_perp2 = v2 - v_dot_B*v_dot_B / B2
        mu[i] = v_perp2 / (2.0 * np.sqrt(B2))
    return mu

@ul.maybe_njit
def compute_mu_rk(solution_rk):
    # Same scalar identity as compute_mu_ps; B is recomputed from position
    # (RK state doesn't carry it). Sign convention matches the simulator
    # (downward dipole moment, upward B at equator). μ is in normalized
    # units (see compute_mu_ps note).
    mu = np.zeros(len(solution_rk))
    for i in range(len(solution_rk)):
        x, y, z = solution_rk[i, 0:3]
        vx, vy, vz = solution_rk[i, 3:6]

        r2 = x*x + y*y + z*z
        if r2 == 0:
            mu[i] = 0.0
            continue
        r5inv = r2**(-2.5)
        Bx = -3.0 * x * z * r5inv
        By = -3.0 * y * z * r5inv
        Bz = -(3.0 * z*z - r2) * r5inv

        B2      = Bx*Bx + By*By + Bz*Bz
        v_dot_B = vx*Bx + vy*By + vz*Bz
        v2      = vx*vx + vy*vy + vz*vz
        v_perp2 = v2 - v_dot_B*v_dot_B / B2
        mu[i] = v_perp2 / (2.0 * np.sqrt(B2))

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
    solution, steps, dt, n_gyro, n_steps_per_gyro,
    gyro_window, time_factor,
    solver_type="rk4",
    y_initial=None,
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
    n_gyro : int
        Number of gyroperiods in the analysis window.
    n_steps_per_gyro : float
        Steps per gyroperiod for this solver.
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
        "mu0"     : float, initial magnetic moment (normalized units —
                    see compute_mu_rk doc).
    """
    window_steps = int(round(n_gyro * n_steps_per_gyro))
    i0, i1 = _gyro_window_indices(gyro_window, steps, window_steps)

    if solver_type == "rkg":
        # RKG stores (N, 6) with canonical momentum — need to convert to velocity.
        # For trimmed files, y_initial preserves the source IC so mu0 is the
        # true baseline rather than mu at trim-start.
        if y_initial is not None:
            r0 = y_initial[0:3]
            p0 = y_initial[3:6]
        else:
            r0 = solution[0, 0:3]
            p0 = solution[0, 3:6]
        A0 = dp.vector_potential(r0)
        v0 = p0 - A0
        state0 = np.hstack((r0, v0))[None, :]
        mu0 = compute_mu_rk(state0)[0]

        r_win = solution[i0:i1, 0:3]
        p_win = solution[i0:i1, 3:6]
        A_win = np.empty_like(r_win)
        for i in range(len(r_win)):
            A_win[i] = dp.vector_potential(r_win[i])
        v_win = p_win - A_win
        state_win = np.hstack((r_win, v_win))
        mu_win = compute_mu_rk(state_win)
    else:
        # RK4 and RK45: shape (6, N) — columns are time steps.
        # See y_initial note above (rkg branch).
        if y_initial is not None:
            state0_src = y_initial.reshape(1, 6)
        else:
            state0_src = solution[:, 0:1].T
        mu0 = compute_mu_rk(state0_src)[0]
        mu_win = compute_mu_rk(solution[:, i0:i1].T)

    mudrift = np.abs(mu_win - mu0) / mu0
    t = (i0 + np.arange(mudrift.size, dtype=ul.npfloat)) * dt * time_factor

    return {"t": t, "mudrift": mudrift, "mu0": mu0}


# ===================================================================
# ============ Mu deviation — PS (chunked h5) =======================
# ===================================================================
def compute_mu_deviation_ps(
    cache_path, steps_ps, ps_step, ps_decimate,
    n_gyro, n_steps_per_gyro, mu0_ps,
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
        Total PS step count (may be the trimmed value when reading a trimmed cache).
    ps_step : float
        PS step size (normalized time).
    ps_decimate : int
        Decimation factor used during streaming.
    n_gyro : int
        Number of gyroperiods in the analysis window.
    n_steps_per_gyro : float
        Steps per gyroperiod (on the PS grid).
    mu0_ps : float
        Initial magnetic moment (PS, in normalized units —
        see compute_mu_ps doc).
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
    window_steps = n_gyro * n_steps_per_gyro
    i0_phys, i1_phys = _gyro_window_indices(gyro_window, steps_ps, window_steps)

    ps_store_stride = ps_decimate if (ps_decimate > 1) else 1
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

        y_ps_win = wr.expand_h5_to_full(ps_y[:, j0:j1])

    mu_ps = compute_mu_ps(y_ps_win)
    mudrift = np.abs(mu_ps - mu0_ps) / mu0_ps

    dt_ps_store = ps_step * ps_store_stride
    t_store = np.arange(j0, j1, dtype=ul.npfloat) * dt_ps_store
    moment_stride = max(1, len(mu_ps) // max_plot_points)
    t_plot = t_store[::moment_stride] * time_factor
    mudrift_plot = mudrift[::moment_stride]

    return {
        "t":              t_plot,
        "mudrift":        mudrift,
        "mudrift_plot":   mudrift_plot,
        "ps_order_label": ps_order_label,
    }
