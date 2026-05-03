"""
Shared utilities for all field drivers (constb, hyperb, dipoleb).

Numba:
    maybe_njit                  — compile with njit for float64, pass through for float128

Solver:
    rk4_fixed_step              — classical 4th-order Runge-Kutta integrator

Plotting:
    f64                         — float128 → float64 conversion for matplotlib
    plt_config                  — global matplotlib rcParams (fonts, DPI)
    sparse_labels               — log-axis tick formatter (label every other decade)
    data_to_fig                 — data coordinates to figure-fraction coordinates
    place_endpoint_labels       — collision-free endpoint labels at axes edge
    setup_log_axes              — standard log-log axis formatting

Slicing (single-solution):
    slice_solution_constb_hyperb — extract a time window from one solution array (constb/hyperb)
    slice_solution_dipoleb       — extract a time window from one solution array (dipoleb)

Slicing (multi-solver orchestration, dipoleb only):
    prepare_slice_dipoleb        — calls slice_solution_dipoleb for each enabled solver,
                                   reads PS data from h5 cache, returns one dict ready to plot

IMPORTANT: modules that use @maybe_njit must be imported AFTER
builtins.npfloat has been set.
"""

import builtins
import os
import numpy as np
from numba import njit
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, LogFormatterSciNotation, NullFormatter, FuncFormatter

try:
    npfloat = builtins.npfloat
except Exception:
    npfloat = np.float64  # fallback if not defined globally

has_float128 = getattr(np, "float128", None) is not None

def maybe_njit(func):
    """Skip njit when using float128 (numba doesn't support it), else compile.

    IMPORTANT: modules using @maybe_njit must be imported AFTER
    builtins.npfloat has been set, otherwise the check sees the
    float64 fallback and always compiles with njit.
    """
    try:
        live_npfloat = builtins.npfloat
    except AttributeError:
        live_npfloat = np.float64
    if has_float128 and live_npfloat == np.float128:
        return func
    else:
        return njit(func)


two = npfloat(2.0)
six = npfloat(6.0)
half = npfloat(0.5)

# ================================================================
# =============== Runge Kutta 4th Order Fixed Step ===============
# ================================================================

@maybe_njit
def rk4_fixed_step(func, d0, dt, steps, args=()):
    d_out = np.zeros((steps + 1, len(d0)), dtype=npfloat)
    d_out[0] = d0

    t = npfloat(0.0)

    for i in range(1, steps + 1):
        y = d_out[i-1]

        k1 = func(t, y, *args)
        k2 = func(t + dt/two, y + dt/two * k1, *args)
        k3 = func(t + dt/two, y + dt/two * k2, *args)
        k4 = func(t + dt,      y + dt * k3, *args)

        d_out[i] = y + (dt/six)*(k1 + two*k2 + two*k3 + k4)
        t += dt

    return d_out.T

# =======================================================
# ============== Misc Assists for Plotting ==============
# =======================================================
def f64(arr):
    """Convert array to float64 (needed when plotting float128 data)."""
    return np.asarray(arr, dtype=np.float64)

def plt_config(scale=1):
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['mathtext.fontset'] = 'cm' # Computer Modern
    plt.rcParams['axes.titlesize'] = int(18*scale)
    plt.rcParams['axes.labelsize'] = int(16*scale)
    plt.rcParams['xtick.labelsize'] = int(14*scale)
    plt.rcParams['ytick.labelsize'] = int(14*scale)
    plt.rcParams['legend.fontsize'] = int(14*scale)
    plt.rcParams['figure.dpi'] = 200
    plt.rcParams['savefig.dpi'] = 600
    # plt.rcParams['figure.constrained_layout.use'] = True

def sparse_labels(val, pos):
    # ignore invalid/nonpositive values (tight_layout may probe these)
    if not np.isfinite(val) or val <= 0:
        return ""
    exp = int(np.round(np.log10(val)))                 # candidate decade
    # only label if it's (numerically) exactly a power of 10
    if not np.isclose(val, 10.0**exp, rtol=0, atol=1e-12):
        return ""
    # keep only every Nth decade; change 3 -> 1 for all decades, 2 for every other, etc.
    return rf"$10^{{{exp}}}$" if (exp % 2 == 0) else ""


def data_to_fig(x, y, ax, fig):
    """Convert a data-coordinate point (x, y) to figure-fraction coordinates.

    Useful for placing annotations (e.g. fig.text) at a position that
    corresponds to a specific data value on a given axes.
    """
    x64 = float(np.asarray(x, dtype=np.float64))
    y64 = float(np.asarray(y, dtype=np.float64))
    px, py = ax.transData.transform(np.array([[x64, y64]], dtype=np.float64))[0]
    fx, fy = fig.transFigure.inverted().transform([[px, py]])[0]
    return fx, fy

def place_endpoint_labels(fig, ax, endpoints, fontsize=11, min_gap=0.025):
    """Place non-overlapping endpoint labels to the right of the axes.

    Parameters
    ----------
    fig : Figure
    ax  : Axes
    endpoints : list of (x_data, y_data, label_str, color)
    fontsize  : int
    min_gap   : float  — minimum vertical spacing in figure coords
    """
    ax_pos = ax.get_position()
    x_fig_label = ax_pos.x1

    labels = []
    for x, y, label, color in endpoints:
        _, fy = data_to_fig(x, y, ax, fig)
        fy = min(max(fy, ax_pos.y0), ax_pos.y1)
        labels.append([fy, label, color])

    labels.sort(key=lambda v: v[0])

    # Push overlapping labels apart (bottom-up)
    for i in range(1, len(labels)):
        if labels[i][0] - labels[i - 1][0] < min_gap:
            labels[i][0] = labels[i - 1][0] + min_gap

    # Clamp back down from the top
    for i in range(len(labels) - 2, -1, -1):
        if labels[i + 1][0] - labels[i][0] < min_gap:
            labels[i][0] = labels[i + 1][0] - min_gap

    for fy, label, color in labels:
        fig.text(x_fig_label, fy, label, color=color,
                 va="center", ha="left", fontsize=fontsize)

def setup_log_axes(ax):
    """Configure log-log axes with the standard formatting used across all plots."""
    ax.margins(x=0.01)
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=100))
    ax.yaxis.set_major_formatter(LogFormatterSciNotation(base=10.0))
    ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=[]))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.xaxis.set_major_locator(LogLocator(base=10.0, numticks=100))
    ax.xaxis.set_major_formatter(LogFormatterSciNotation(base=10.0))
    ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=[]))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.grid(True, which="major", linestyle="--", linewidth=0.7)
    ax.yaxis.set_major_formatter(FuncFormatter(sparse_labels))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

def slice_solution_constb_hyperb(t_eval, sol, window_duration, norm_time, mode="last"):
    """Extract a time window from a solution array (constb/hyperb drivers).

    Parameters
    ----------
    t_eval          : 1-D time array.
    sol             : 2-D solution array, either (nvars, npts) or (npts, nvars).
    window_duration : length of the window in normalised time units.
    norm_time       : total simulation time (used to locate the tail end).
    mode            : "last" returns the final `window_duration` of the run;
                      "first" returns from t=0 up to `window_duration`.

    Returns a list of 1-D arrays, one per variable (x, y, z, vx, …).
    """
    t_eval = np.asarray(t_eval)
    if mode == "first":
        # Slice from start up to first N gyroperiods
        end_t = min(t_eval[-1], window_duration)
        end_idx = np.searchsorted(t_eval, end_t, side="right")
        if sol.shape[0] <= sol.shape[1]:  # (nvars, npts)
            return [s[:end_idx] for s in sol]
        else:  # (npts, nvars)
            return [sol[:end_idx, i] for i in range(sol.shape[1])]

    elif mode == "last":
        # Slice last N gyroperiods
        start_t = max(t_eval[0], norm_time - window_duration)
        start_idx = np.searchsorted(t_eval, start_t, side="left")
        if sol.shape[0] <= sol.shape[1]:
            return [s[start_idx:] for s in sol]
        else:
            return [sol[start_idx:, i] for i in range(sol.shape[1])]

    else:
        raise ValueError("mode must be 'first' or 'last'")


def slice_solution_dipoleb(t, sol, window_duration, norm_time, mode="last"):
    """Extract a time window from a dipole solution array.

    Parameters
    ----------
    t               : 1-D time array.
    sol             : 2-D solution array, either (nvars, npts) or (npts, nvars).
                      Pass None to return indices only.
    window_duration : length of the window in normalised time units.
    norm_time       : total simulation time (used to locate the tail end).
    mode            : "last" returns the final `window_duration` of the run;
                      "first" returns from t=0 up to `window_duration`.

    Returns (x, y, z) arrays, or index array if sol is None.
    """
    if mode == "last":
        t_end = norm_time
        t_start = max(t[0], t_end - window_duration)
    elif mode == "first":
        t_start = t[0]
        t_end = min(t[-1], t_start + window_duration)
    else:
        raise ValueError(f"Unknown slice mode: {mode}")

    idx = np.where((t >= t_start) & (t <= t_end))[0]

    if sol is None:
        return idx

    if sol.shape[0] <= sol.shape[1]:
        arr = sol
    else:
        arr = sol.T

    x = arr[0, idx]
    y = arr[1, idx]
    z = arr[2, idx]

    return x, y, z


def prepare_slice_dipoleb(
    slice_mode, window_duration, norm_time,
    # PS-specific
    USE_PS=False, cache_path=None, ps_step=None, steps_ps=None,
    PS_decimate=1, MAX_PLOT_POINTS=1_000_000,
    # RK4
    USE_RK4=False, solution_rk4=None, rk4_step=None,
    # RKG
    USE_RKG=False, solution_rkg=None, rkg_step=None,
    # RK45
    USE_RK45=False, y_rk45_common=None,
):
    """Compute time-windowed trajectory slices for each enabled solver (dipoleb).

    Reads PS data directly from the h5 cache (to avoid loading the full array)
    and delegates RK slicing to slice_solution_dipoleb.

    Returns a dict with keys like ``ps_x_slice``, ``rk4_y_slice``, etc.
    Missing solvers get ``None`` values.
    """
    import h5py

    if slice_mode == "first":
        t_start = 0.0
        t_end   = min(norm_time, window_duration)
    elif slice_mode == "last":
        t_end   = norm_time
        t_start = max(0.0, norm_time - window_duration)
    else:
        raise ValueError("slice_mode must be 'first' or 'last'")

    result = dict(
        ps_x_slice=None, ps_y_slice=None, ps_z_slice=None,
        rk4_x_slice=None, rk4_y_slice=None, rk4_z_slice=None,
        rkg_x_slice=None, rkg_y_slice=None, rkg_z_slice=None,
        rk45_x_slice=None, rk45_y_slice=None, rk45_z_slice=None,
        ps_order_label=None,
    )

    # ---------- PS ----------
    if USE_PS:
        i0_phys = int(np.floor(t_start / ps_step))
        i1_phys = int(np.floor(t_end   / ps_step))
        i0_phys = max(0, i0_phys)
        i1_phys = min(i1_phys, steps_ps)
        if i1_phys < i0_phys:
            raise RuntimeError("Empty PS slice window")

        ps_store_stride = PS_decimate if PS_decimate > 1 else 1
        j0 = int(np.ceil(i0_phys / ps_store_stride))
        j1 = int(np.floor(i1_phys / ps_store_stride))
        if j1 < j0:
            raise RuntimeError("Empty PS stored slice window")

        with h5py.File(cache_path, "r") as ps_h5:
            ps_grp = ps_h5["ps"]
            ps_y   = ps_grp["y"]
            n_store = ps_y.shape[1]
            j0 = max(0, min(j0, n_store - 1))
            j1 = max(0, min(j1, n_store - 1))
            if j1 < j0:
                raise RuntimeError("Empty PS stored slice")
            y_win = ps_y[:, j0:j1+1]
            result["ps_order_label"] = int(ps_grp.attrs["max_ps"])

        plot_stride = max(1, y_win.shape[1] // MAX_PLOT_POINTS)
        result["ps_x_slice"] = y_win[0, ::plot_stride]
        result["ps_y_slice"] = y_win[1, ::plot_stride]
        result["ps_z_slice"] = y_win[2, ::plot_stride]

    # ---------- RK4 ----------
    if USE_RK4:
        t_rk4 = rk4_step * np.arange(solution_rk4.shape[1], dtype=npfloat)
        rk4_x, rk4_y, rk4_z = slice_solution_dipoleb(
            t_rk4, solution_rk4, window_duration, norm_time, mode=slice_mode)[:3]
        result["rk4_x_slice"] = rk4_x
        result["rk4_y_slice"] = rk4_y
        result["rk4_z_slice"] = rk4_z

    # ---------- RKG ----------
    if USE_RKG:
        t_rkg = rkg_step * np.arange(solution_rkg.shape[0], dtype=npfloat)
        rkg_x, rkg_y, rkg_z = slice_solution_dipoleb(
            t_rkg, solution_rkg.T, window_duration, norm_time, mode=slice_mode)[:3]
        result["rkg_x_slice"] = rkg_x
        result["rkg_y_slice"] = rkg_y
        result["rkg_z_slice"] = rkg_z

    # ---------- RK45 ----------
    if USE_RK45:
        t_rk45 = ps_step * np.arange(y_rk45_common.shape[1], dtype=npfloat)
        rk45_x, rk45_y, rk45_z = slice_solution_dipoleb(
            t_rk45, y_rk45_common, window_duration, norm_time, mode=slice_mode)[:3]
        result["rk45_x_slice"] = rk45_x
        result["rk45_y_slice"] = rk45_y
        result["rk45_z_slice"] = rk45_z

    return result
