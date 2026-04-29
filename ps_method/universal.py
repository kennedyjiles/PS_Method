import builtins
import logging
import numpy as np
from numba import njit
import matplotlib.pyplot as plt

try:
    npfloat = builtins.npfloat
except Exception:
    npfloat = np.float64  # fallback if not defined globally

has_float128 = getattr(np, "float128", None) is not None

def maybe_njit(func):
    if has_float128 and npfloat == np.float128:
        return func
    else:
        return njit(func)


two = npfloat(2.0)
six = npfloat(6.0)
half = npfloat(0.5)

# =========================================
# ============ Relative KE Error ==========
# =========================================

@maybe_njit
def kinetic_energy(vx, vy, vz, m=npfloat(1.0)):
    return half * m * (vx**two + vy**two + vz**two)

@maybe_njit
def compute_energy_drift(vx, vy, vz):
    KE = kinetic_energy(vx, vy, vz)
    return (KE - KE[0]) / KE[0]

@maybe_njit
def extract_v(sol):  # assumes PS output has x, y, z, vx, vy, vz as initial entries
    return sol[3], sol[4], sol[5]

# =========================================
# ============= Cauchy Related ============
# =========================================
@maybe_njit
def cauchy_sum(a, b, n):
    result = 0.0
    for j in range(n + 1):
        result += a[j] * b[n - j]
    return result  
    
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

# ===========================================
# ============== Misc Plotting ==============
# ===========================================
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
    x64 = float(np.asarray(x, dtype=np.float64))
    y64 = float(np.asarray(y, dtype=np.float64))
    px, py = ax.transData.transform(np.array([[x64, y64]], dtype=np.float64))[0]
    fx, fy = fig.transFigure.inverted().transform([[px, py]])[0]
    return fx, fy

def slice_solution(t_eval, sol, window_duration, norm_time, mode="last"):
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

# ========================================
# ============= Logging ==================
# ========================================

def setup_logger(name="dipole_logger", filename="dipole_run.log", level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)

    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    formatter = logging.Formatter('%(levelname)s — %(message)s')

    file_handler = logging.FileHandler(filename, mode="w")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger