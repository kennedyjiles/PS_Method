import builtins
import numpy as np
from numba import njit
from matplotlib import transforms
import matplotlib.pyplot as plt

try:
    npfloat = builtins.npfloat
except AttributeError:
    npfloat = np.float64  # fallback if driver didn't set it

two = npfloat(2.0)
six = npfloat(6.0)
half = npfloat(0.5)

# #########################################
# ============= KE fo Drift  ============
# #########################################

@((lambda f: f) if npfloat == np.float128 else njit)
def kinetic_energy(vx, vy, vz, m=npfloat(1.0)):
    return half * m * (vx**two + vy**two + vz**two)

@((lambda f: f) if npfloat == np.float128 else njit)
def compute_energy_drift(vx, vy, vz):
    KE = kinetic_energy(vx, vy, vz)
    return (KE - KE[0]) / KE[0]

@((lambda f: f) if npfloat == np.float128 else njit)
def extract_v(sol):  # assumes PS output has x, y, z, vx, vy, vz as initial entries
    return sol[3], sol[4], sol[5]

# #########################################
# ============= Cauchy Related ============
# #########################################
@((lambda f: f) if npfloat == np.float128 else njit)
def cauchy_sum(a, b, n):
    result = 0.0
    for j in range(n + 1):
        result += a[j] * b[n - j]
    return result  
    
# ================================================================
# =============== Runge Kutta 4th Order Fixed Step ===============
# ================================================================

@((lambda f: f) if npfloat == np.float128 else njit)
def rk4_fixed_step(func, d0, t, args=()):
    """
    func will generally be the lorentz force function
    d0 is initial values for all variables in lorenz force
    t is a time array
    args will vary by problem but defined in
    """
    d_out = np.zeros((len(t), len(d0)), dtype=npfloat)  
    d_out[0] = d0
    dt = npfloat(t[1] - t[0])  

    for i in range(1, len(t)):
        k1 = func(npfloat(t[i-1]), d_out[i-1], *args)
        k2 = func(npfloat(t[i-1]) + dt/two, d_out[i-1] + dt/two * k1, *args)
        k3 = func(npfloat(t[i-1]) + dt/two, d_out[i-1] + dt/two * k2, *args)
        k4 = func(npfloat(t[i-1]) + dt,   d_out[i-1] + dt * k3, *args)
        d_out[i] = d_out[i-1] + (dt/six)*(k1 + two*k2 + two*k3 + k4)

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


def _last_finite_pos(y):
    """Return last finite, positive y (for log axes)."""
    y = np.asarray(y, dtype=np.float64)
    for v in y[::-1]:
        if np.isfinite(v) and v > 0:
            return float(v)
    # fallback: tiniest positive to keep label on-axes
    return np.finfo(np.float64).tiny

def label_right_collision_free(ax, lines, names, x=1.01, min_sep=0.05, fontsize=9):
    """
    Place labels just outside the right edge (x in axes coords) while aligning
    vertically to each line's endpoint (y in data coords), with collision avoidance
    done in axes-fraction space. No Matplotlib transforms on your data arrays.
    """
    trans = transforms.blended_transform_factory(ax.transAxes, ax.transData)

    # 1) get each line's last finite, positive y
    y_end = np.array([_last_finite_pos(ln.get_ydata()) for ln in lines], dtype=np.float64)

    # 2) normalize to axes-fraction y for spacing (works for linear or log)
    y0, y1 = ax.get_ylim()
    if ax.get_yscale() == 'log':
        # guard log range
        tiny = np.finfo(np.float64).tiny
        y0, y1 = max(y0, tiny), max(y1, tiny)
        lo, hi = np.log10(y0), np.log10(y1)
        y_norm = (np.log10(y_end) - lo) / (hi - lo)
    else:
        lo, hi = y0, y1
        y_norm = (y_end - lo) / (hi - lo)

    # 3) sort and enforce minimum separation in axes space
    order = np.argsort(y_norm)
    y_norm = y_norm[order]
    names  = [names[i] for i in order]
    colors = [lines[i].get_color() for i in order]

    for i in range(1, len(y_norm)):
        if y_norm[i] - y_norm[i-1] < min_sep:
            y_norm[i] = y_norm[i-1] + min_sep

    # keep within axes vertically
    y_norm = np.clip(y_norm, 0.02, 0.98)

    # 4) map back to data y (still without transforms)
    if ax.get_yscale() == 'log':
        y_adj = 10 ** (lo + y_norm * (hi - lo))
    else:
        y_adj = lo + y_norm * (hi - lo)

    # 5) draw labels just outside the axes
    for yy, nm, col in zip(y_adj, names, colors):
        ax.text(x, float(yy), nm, transform=trans, va='center', ha='left',
                color=col, clip_on=False, fontsize=fontsize)

def interp_to_grid(t_src, y_src, t_target, *, fill_value=np.nan):
    # Cast to float64
    t_src64 = np.asarray(t_src, dtype=np.float64).ravel()
    y_src64 = np.asarray(y_src, dtype=np.float64).ravel()
    t_tgt64 = np.asarray(t_target, dtype=np.float64).ravel()

    # Remove NaN/inf pairs
    mask = np.isfinite(t_src64) & np.isfinite(y_src64)
    t_src64 = t_src64[mask]
    y_src64 = y_src64[mask]

    # Need at least 2 points to interpolate
    if t_src64.size < 2:
        return np.full_like(t_tgt64, fill_value, dtype=np.float64)

    # Sort by t_src and deduplicate (np.interp requires ascending xp)
    order = np.argsort(t_src64)
    t_src64 = t_src64[order]
    y_src64 = y_src64[order]
    t_src64, uniq_idx = np.unique(t_src64, return_index=True)
    y_src64 = y_src64[uniq_idx]

    # Interpolate (no extrapolation by default)
    out = np.interp(t_tgt64, t_src64, y_src64, left=fill_value, right=fill_value)
    return out

def data_to_fig(x, y, ax, fig):
    # Force float64 scalars
    x64 = float(np.asarray(x, dtype=np.float64))
    y64 = float(np.asarray(y, dtype=np.float64))

    # data -> display (shape must be Nx2)
    px, py = ax.transData.transform(np.array([[x64, y64]], dtype=np.float64))[0]
    # display -> figure
    fx, fy = fig.transFigure.inverted().transform([[px, py]])[0]
    return fx, fy
