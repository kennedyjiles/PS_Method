"""
dipoleb_moment_analysis.py — Magnetic moment diagnostics for dipole trajectories.

    compute_mu_ps            — mu from PS solution array
    compute_mu_rk            — mu from RK solution array
    compute_mu_deviation_rk  — mu deviation over time (RK solvers)
    compute_mu_deviation_ps  — mu deviation over time (PS, chunked from h5)

Internal helpers:
    _gyro_window_indices          — index range for a gyro-window slice
    _equatorial_crossing_indices  — sample indices where z changes sign
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
# ============ Equatorial-crossing helper ===========================
# ===================================================================
def _equatorial_crossing_indices(z):
    """Sample indices where the orbit is on / crosses the magnetic equator (z = 0).

    Two cases, combined and returned sorted:
      1. A strict sign change between consecutive non-zero samples
         (z[i]*z[i+1] < 0) -> index i, the sample just before the crossing.
      2. A sample lying EXACTLY on the equator (z[i] == 0) that has a non-zero
         neighbour -> index i. This is what marks an equatorial LAUNCH
         (z_initial = 0), regardless of which hemisphere the particle heads
         into first. Requiring a non-zero neighbour excludes the degenerate
         90-degree-pitch orbit that stays at z == 0 forever (which would
         otherwise flag every sample). An off-equator launch (z[0] != 0) is
         never flagged.

    Uses exact zero (no tolerance) on purpose: a tolerance would produce
    false positives for near-equatorial orbits, and the launch value comes
    straight from the config so it is exactly 0.0 when it is meant to be.
    Sample-level accuracy (< one step); these feed plot markers only, and the
    values taken at these indices lie exactly on the solver's own curve.
    """
    z = np.asarray(z, dtype=float)
    n = z.size
    if n < 2:
        return np.zeros(0, dtype=np.int64)
    s = np.sign(z)                       # -1, 0, +1  (sign(-0.0) == 0)
    # case 1: strict crossings between non-zero samples
    strict = np.where(s[:-1] * s[1:] < 0)[0]
    # case 2: exact-zero samples with at least one non-zero neighbour
    on_eq = (s == 0)
    left_nz  = np.concatenate(([False], s[:-1] != 0))
    right_nz = np.concatenate((s[1:] != 0, [False]))
    exact = np.where(on_eq & (left_nz | right_nz))[0]
    return np.unique(np.concatenate((strict, exact))).astype(np.int64)


# ===================================================================
# ============ Mu deviation — RK solvers (in-memory) ================
# ===================================================================
def compute_mu_deviation_rk(
    solution, steps, dt, n_gyro, n_steps_per_gyro,
    gyro_window, time_factor,
    solver_type="rk4",
    y_initial=None,
    charge_sign=1,
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
        "t"        : 1D time array in gyroperiods
        "mudrift"  : 1D relative mu deviation array
        "mu0"      : float, initial magnetic moment (normalized units —
                     see compute_mu_rk doc).
        "mu_ratio" : 1D mu/mu0 (instantaneous normalized moment, for shape plot)
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
        # v = p - charge_sign * A (matches hamiltonian_rhs in dipoleb_physics)
        v0 = p0 - charge_sign * A0
        state0 = np.hstack((r0, v0))[None, :]
        mu0 = compute_mu_rk(state0)[0]

        r_win = solution[i0:i1, 0:3]
        p_win = solution[i0:i1, 3:6]
        A_win = np.empty_like(r_win)
        for i in range(len(r_win)):
            A_win[i] = dp.vector_potential(r_win[i])
        v_win = p_win - charge_sign * A_win
        state_win = np.hstack((r_win, v_win))
        mu_win = compute_mu_rk(state_win)
        z_win = r_win[:, 2]
    else:
        # RK4 and RK45: shape (6, N) — columns are time steps.
        # See y_initial note above (rkg branch).
        if y_initial is not None:
            state0_src = y_initial.reshape(1, 6)
        else:
            state0_src = solution[:, 0:1].T
        mu0 = compute_mu_rk(state0_src)[0]
        mu_win = compute_mu_rk(solution[:, i0:i1].T)
        z_win = solution[2, i0:i1]

    mudrift = np.abs(mu_win - mu0) / mu0
    mu_ratio = mu_win / mu0          # instantaneous mu, normalized (shape plot)
    t = (i0 + np.arange(mudrift.size, dtype=ul.npfloat)) * dt * time_factor

    # Equatorial crossings within the window — optional markers for the
    # mu_deviation / mu_shape figures (values sit exactly on this curve).
    eq_idx = _equatorial_crossing_indices(z_win)

    return {"t": t, "mudrift": mudrift, "mu0": mu0,
            "mu_ratio": mu_ratio,
            "eq_t": t[eq_idx], "eq_mudrift": mudrift[eq_idx],
            "eq_mu_ratio": mu_ratio[eq_idx]}


# Cap on the mu_deviation / mu_shape window, expanded to the 17-row layout.
# Guards gyro_window: "all" on long runs; raise it if you have the RAM.
MU_WINDOW_MAX_BYTES = 4 * 1024**3

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
        "mu_ratio_plot"  : 1D decimated mu/mu0 for the shape plot
        "ps_order_label" : int, plot label (mean PS order) from h5 attrs
    """
    window_steps = n_gyro * n_steps_per_gyro
    i0_phys, i1_phys = _gyro_window_indices(gyro_window, steps_ps, window_steps)

    ps_store_stride = ps_decimate if (ps_decimate > 1) else 1
    j0 = int(np.floor(i0_phys / ps_store_stride))
    j1 = int(np.ceil(i1_phys / ps_store_stride))

    with h5py.File(cache_path, "r") as ps_h5:
        ps_grp = ps_h5["ps"]
        ps_y = ps_grp["y"]
        ps_order_label = ul.ps_order_label_from_attrs(ps_grp.attrs)
        n_store = ps_y.shape[1]
        # Absolute run-time offset: for a segment, t0 = ps_step * start_global_index
        # so τ/T reflects the segment's position in the full run. 0.0 for a whole run.
        t0_ps = float(ps_grp.attrs.get("t0", 0.0))

        j0 = max(0, min(j0, n_store))
        j1 = max(0, min(j1, n_store))

        if j1 <= j0:
            raise RuntimeError("Empty PS mu window (chunked)")

        # The window is read at FULL resolution (the equatorial-crossing markers
        # below rely on that) and then expanded 9 -> 17 rows, so the transient
        # cost is ~26 rows x window x 8 bytes. With gyro_window "first"/"last"
        # that window is n_gyro gyroperiods and stays small. gyro_window "all"
        # sets it to the WHOLE run, which on a long segmented run is tens of GB
        # — refuse it rather than being OOM-killed mid-plot.
        _win_cols = j1 - j0
        _win_bytes = _win_cols * 26 * 8
        if _win_bytes > MU_WINDOW_MAX_BYTES:
            raise RuntimeError(
                f"PS mu window is {_win_cols:,} stored steps "
                f"(~{_win_bytes / 1024**3:.1f} GB once expanded to the 17-row "
                f"layout), over the {MU_WINDOW_MAX_BYTES / 1024**3:.1f} GB cap. "
                f"This is the mu_deviation / mu_shape window only — the full-run "
                f"mu error plot (compute_mu_errors) is chunked and unaffected. "
                f"Use gyro_window 'first' or 'last' with a smaller n_gyro, or "
                f"raise MU_WINDOW_MAX_BYTES if you really have the RAM.")

        y_ps_win = wr.expand_h5_to_full(ps_y[:, j0:j1])

    mu_ps = compute_mu_ps(y_ps_win)
    mudrift = np.abs(mu_ps - mu0_ps) / mu0_ps
    mu_ratio = mu_ps / mu0_ps        # instantaneous mu, normalized (shape plot)

    dt_ps_store = ps_step * ps_store_stride
    t_store = np.arange(j0, j1, dtype=ul.npfloat) * dt_ps_store + t0_ps
    moment_stride = max(1, len(mu_ps) // max_plot_points)
    t_plot = t_store[::moment_stride] * time_factor
    # .copy() so the decimated plot arrays don't stay views onto the full-window
    # mudrift / mu_ratio arrays — mu_ratio has no other consumer and can be
    # freed once the shape plot has its points.
    mudrift_plot = mudrift[::moment_stride].copy()
    mu_ratio_plot = mu_ratio[::moment_stride].copy()

    # Equatorial crossings within the window, detected at FULL window
    # resolution (before moment_stride decimation) so the marker values sit
    # exactly on the true curve. Optional markers for the mu figures.
    eq_idx = _equatorial_crossing_indices(y_ps_win[2])

    return {
        "t":              t_plot,
        "mudrift":        mudrift,
        "mudrift_plot":   mudrift_plot,
        "mu_ratio_plot":  mu_ratio_plot,
        "ps_order_label": ps_order_label,
        "eq_t":           t_store[eq_idx] * time_factor,
        "eq_mudrift":     mudrift[eq_idx],
        "eq_mu_ratio":    mu_ratio[eq_idx],
    }
