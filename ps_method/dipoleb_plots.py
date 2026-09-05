"""
Plotting functions for dipoleb.py — trajectory visualizations.

Each function receives only the data it needs (no globals). Called from
dipoleb.py after data loading / slice setup is complete.

Trajectory:
    full_2d              — full-run 2D trajectory (x vs y)
    full_3d              — full-run 3D trajectory
    slice_2d             — windowed 2D trajectory slice
    slice_3d             — windowed 3D trajectory slice

Energy:
    ke_error             — kinetic energy relative error vs time

Dragt (plotted in his non-dimensionless units):
    poincare             — Poincaré surface of section (rho vs rho_dot),
                           Dragt units with a second R_E axis pair on top/right
    gyrophase_mu         — gyrophase vs magnetic moment at crossings
    polar_phase_space    — polar phase space (mu, gyrophase)
    meridian_plane       — meridian plane projection (rho/L vs z/L, Dragt units)
    meridian_plane_RE    — meridian plane projection (rho vs z, in R_E)
    adiabaticity         — adiabaticity parameter epsilon vs time

Conservation:
    pphi_error           — P_phi relative error vs time
    mu_error             — full-run magnetic moment relative error vs time
    invariants_overlay   — E, P_phi, mu relative errors on one axis (PS only)
    mu_deviation         — magnetic moment conservation error vs time
    mu_shape             — instantaneous magnetic moment μ/μ₀ vs time (shape)

Internal helpers:
    _log_envelope        — per-log-bin upper envelope for readable log-log curves
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, LogFormatterSciNotation, NullFormatter
from ps_method import writers as wr
from ps_method import utils as ul
import os

# =====================================================
# Centralized solver styling — change once, applies everywhere
# =====================================================
COLORS = {
    "rk45": "#E69F00",   # orange
    "rk4":  "#CC79A7",   # reddish purple
    "rkg":  "#CC0000",   # red
    "ps":   "#009E73",   # bluish green
}

LINESTYLES = {
    "rk45": "--",
    "rk4":  "-.",
    "rkg":  "-.",
    "ps":   "-",     # solid — PS is the reference solver, drawn on top
}

# PS line thickness, shared by every PS curve. Change once, applies everywhere.
PS_LINEWIDTH = 0.45


# =====================================================
# Interactive review mode (mu_deviation plot only)
# =====================================================
# When enabled, the mu_deviation figure (<hash>_mu.png) is still saved to disk
# exactly as normal, but is then ALSO opened in a matplotlib window and the run
# PAUSES until you close that window — so you can zoom / pan on the live figure
# before the script moves on. Off by default (batch runs unaffected). Enable
# from the shell without touching any code:
#
#     PS_INTERACTIVE=1 python dipoleb.py demo
#
# or set INTERACTIVE_PLOTS = True here for a session. To extend to another
# figure, replace its trailing plt.close(fig) with _close_or_show(fig).
INTERACTIVE_PLOTS = os.environ.get("PS_INTERACTIVE", "").strip().lower() in ("1", "true", "yes", "on")


def _close_or_show(fig):
    """Finish a figure: pop it up (blocking) in interactive mode, then close it.

    Used by mu_deviation in place of plt.close(fig), AFTER the savefig, so the
    saved PNG is identical whether or not you review it.
    """
    if INTERACTIVE_PLOTS:
        plt.show()      # blocking — the run resumes when you close the window
    plt.close(fig)


def _log_envelope(x, y, n_bins=1500):
    """Upper envelope of (x, y) for readable log-log plotting.

    Bins x uniformly in log10 and takes max(|y|) per bin. Collapses a dense,
    oscillatory band (e.g. the symplectic RKG energy error) into a clean upper
    bound without distorting the peak, while leaving smooth curves essentially
    unchanged. Returns (x_centers, y_max) over non-empty bins.
    """
    x = np.asarray(x); y = np.abs(np.asarray(y))
    pos = x > 0
    x, y = x[pos], y[pos]
    if x.size <= n_bins:
        return x, y
    lx = np.log10(x)
    edges = np.linspace(lx.min(), lx.max(), n_bins + 1)
    idx = np.clip(np.searchsorted(edges, lx, side="right") - 1, 0, n_bins - 1)
    ymax = np.full(n_bins, -np.inf)
    np.maximum.at(ymax, idx, y)
    centers = 10.0 ** (0.5 * (edges[:-1] + edges[1:]))
    keep = np.isfinite(ymax) & (ymax > 0)
    return centers[keep], ymax[keep]


# =====================================================
# ============== Full 2D Trajectory Plot ==============
# =====================================================
def full_2d(
    run_folder, stem, particle_type, plotbounds, ps_order_label,
    USE_PLOT_TITLES, USE_RK45, USE_RK4, USE_RKG, USE_PS,
    solution_rk45=None, solution_rk4=None, solution_rkg=None,
    x_ps_plot=None, y_ps_plot=None,
):
    """Full-run 2D trajectory (x vs y) for all enabled solvers."""
    fig, ax = plt.subplots(figsize=(10, 8))

    if USE_RK45:
        ax.plot(solution_rk45.y[0], solution_rk45.y[1], label='RK45', color=COLORS["rk45"], linestyle=LINESTYLES["rk45"], linewidth=0.5)
    if USE_RK4:
        ax.plot(solution_rk4[0], solution_rk4[1], label='RK4', alpha=0.8, color=COLORS["rk4"], linestyle=LINESTYLES["rk4"], linewidth=0.5)
    if USE_RKG:
        ax.plot(solution_rkg[:, 0], solution_rkg[:, 1], label='RKG', alpha=0.8, color=COLORS["rkg"], linestyle=LINESTYLES["rkg"], linewidth=0.5)
    if USE_PS:
        ax.plot(x_ps_plot, y_ps_plot, label=f"PS{ps_order_label}", alpha=0.7, color=COLORS["ps"], linestyle=LINESTYLES["ps"], linewidth=PS_LINEWIDTH)

    ax.set_xlabel(r"x")
    ax.set_ylabel(r"y")
    ax.ticklabel_format(style='plain', useOffset=False, axis='both')
    if USE_PLOT_TITLES:
        ax.set_title(f"2D {particle_type} Trajectory in Dipole B Field")

    leg = ax.legend(loc="upper right")
    for line in leg.get_lines():
        line.set_linewidth(2.5)
    ax.set_xlim(-plotbounds, plotbounds)
    ax.set_ylim(-plotbounds, plotbounds)
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True)

    fig_path = wr.build_filename(run_folder, stem, figure_tag="2D", ext="png")
    plt.savefig(fig_path, dpi=600, bbox_inches="tight")
    plt.close(fig)


# =====================================================
# ============== Full 3D Trajectory Plot ==============
# =====================================================
def full_3d(
    run_folder, stem, particle_type, plotbounds, ps_order_label,
    USE_PLOT_TITLES, USE_RK45, USE_RK4, USE_RKG, USE_PS,
    solution_rk45=None, solution_rk4=None, solution_rkg=None,
    x_ps_plot=None, y_ps_plot=None, z_ps_plot=None,
):
    """Full-run 3D trajectory for all enabled solvers."""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    if USE_RK45:
        ax.plot(solution_rk45.y[0], solution_rk45.y[1], solution_rk45.y[2], label="RK45", color=COLORS["rk45"], linestyle=LINESTYLES["rk45"], linewidth=0.5)
    if USE_RK4:
        ax.plot(solution_rk4[0], solution_rk4[1], solution_rk4[2], label='RK4', alpha=0.8, color=COLORS["rk4"], linestyle=LINESTYLES["rk4"], linewidth=0.5)
    if USE_RKG:
        ax.plot(solution_rkg[:, 0], solution_rkg[:, 1], solution_rkg[:, 2], label='RKG', alpha=0.8, color=COLORS["rkg"], linestyle=LINESTYLES["rkg"], linewidth=0.5)
    if USE_PS:
        ax.plot(x_ps_plot, y_ps_plot, z_ps_plot, label=f"PS{ps_order_label}", alpha=0.7, color=COLORS["ps"], linestyle=LINESTYLES["ps"], linewidth=PS_LINEWIDTH)

    ax.set_xlim(-plotbounds, plotbounds)
    ax.set_ylim(-plotbounds, plotbounds)
    ax.set_zlim(-plotbounds, plotbounds)

    ax.set_xlabel(r'X')
    ax.set_ylabel(r'Y')
    ax.set_zlabel(r'Z')
    if USE_PLOT_TITLES:
        ax.set_title(f"3D {particle_type} Trajectory in Dipole B Field")
    leg = ax.legend(loc="upper right")
    for line in leg.get_lines():
        line.set_linewidth(2.5)

    fig_path = wr.build_filename(run_folder, stem, figure_tag="3D", ext="png")
    plt.savefig(fig_path, dpi=600, bbox_inches="tight")
    plt.close(fig)


# =====================================================
# ================ 2D Trajectory Slice ================
# =====================================================
def slice_2d(
    run_folder, stem, particle_type, ps_order_label,
    USE_PLOT_TITLES, USE_RK45, USE_RK4, USE_RKG, USE_PS,
    rk45_x_slice=None, rk45_y_slice=None,
    rk4_x_slice=None, rk4_y_slice=None,
    rkg_x_slice=None, rkg_y_slice=None,
    ps_x_slice=None, ps_y_slice=None,
):
    """Time-windowed 2D trajectory slice for all enabled solvers."""
    fig, ax = plt.subplots(figsize=(10, 7))

    if USE_RK45:
        ax.plot(rk45_x_slice, rk45_y_slice, label='RK45', color=COLORS["rk45"], linestyle=LINESTYLES["rk45"], linewidth=0.5)
    if USE_RK4:
        ax.plot(rk4_x_slice, rk4_y_slice, label='RK4', alpha=0.8, color=COLORS["rk4"], linestyle=LINESTYLES["rk4"], linewidth=0.5)
    if USE_RKG:
        ax.plot(rkg_x_slice, rkg_y_slice, label='RKG', alpha=0.8, color=COLORS["rkg"], linestyle=LINESTYLES["rkg"], linewidth=0.5)
    if USE_PS:
        ax.plot(ps_x_slice, ps_y_slice, label=f"PS{ps_order_label}", alpha=0.8, color=COLORS["ps"], linestyle=LINESTYLES["ps"], linewidth=PS_LINEWIDTH)



    ax.set_xlabel(r"x")
    ax.set_ylabel(r"y")
    if USE_PLOT_TITLES:
        ax.set_title(f"2D Trajectory of Slice {particle_type} Orbits in Dipole B Field")
    ax.axis('equal')
    leg = ax.legend(loc="upper right")
    for line in leg.get_lines():
        line.set_linewidth(2.5)
    ax.grid(True)

    fig_path = wr.build_filename(run_folder, stem, figure_tag="2Dslice", ext="png")
    plt.savefig(fig_path, dpi=600, bbox_inches="tight")
    plt.close(fig)


# =====================================================
# ================ 3D Trajectory Slice ================
# =====================================================
def slice_3d(
    run_folder, stem, particle_type, plotbounds, ps_order_label,
    USE_PLOT_TITLES, USE_RK45, USE_RK4, USE_RKG, USE_PS,
    rk45_x_slice=None, rk45_y_slice=None, rk45_z_slice=None,
    rk4_x_slice=None, rk4_y_slice=None, rk4_z_slice=None,
    rkg_x_slice=None, rkg_y_slice=None, rkg_z_slice=None,
    ps_x_slice=None, ps_y_slice=None, ps_z_slice=None,
):
    """Time-windowed 3D trajectory slice for all enabled solvers."""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    if USE_RK45:
        ax.plot(rk45_x_slice, rk45_y_slice, rk45_z_slice, label='RK45', color=COLORS["rk45"], linestyle=LINESTYLES["rk45"], linewidth=0.5)
    if USE_RK4:
        ax.plot(rk4_x_slice, rk4_y_slice, rk4_z_slice, label='RK4', alpha=0.8, color=COLORS["rk4"], linestyle=LINESTYLES["rk4"], linewidth=0.5)
    if USE_RKG:
        ax.plot(rkg_x_slice, rkg_y_slice, rkg_z_slice, label='RKG', alpha=0.8, color=COLORS["rkg"], linestyle=LINESTYLES["rkg"], linewidth=0.5)
    if USE_PS:
        ax.plot(ps_x_slice, ps_y_slice, ps_z_slice, label=f"PS{ps_order_label}", alpha=0.8, color=COLORS["ps"], linestyle=LINESTYLES["ps"], linewidth=PS_LINEWIDTH)

    ax.set_xlim(-plotbounds, plotbounds)
    ax.set_ylim(-plotbounds, plotbounds)
    ax.set_zlim(-plotbounds, plotbounds)
    ax.grid(True)

    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$y$')
    ax.set_zlabel(r'$z$')
    if USE_PLOT_TITLES:
        ax.set_title(f'3D Trajectory Slice of {particle_type} Orbits in Dipole B Field')
    leg = ax.legend(loc="upper right")
    for line in leg.get_lines():
        line.set_linewidth(2.5)

    fig_path = wr.build_filename(run_folder, stem, figure_tag="3Dslice", ext="png")
    plt.savefig(fig_path, dpi=600, bbox_inches="tight")
    plt.close(fig)


# =====================================================
# ============== KE Relative Error Plot ===============
# =====================================================
def ke_error(
    run_folder, stem, particle_type, ps_order_label,
    USE_PLOT_TITLES, time_factor, norm_time,
    ps_data=None, rk4_data=None, rk45_data=None, rkg_data=None,
    ext_ps_data=None, ext_rk4_data=None, ext_rk45_data=None, ext_rkg_data=None,
    envelope=True,
):
    """
    Kinetic energy relative error (log-log) for all enabled solvers.

    Each *_data argument is a tuple (t_array, rel_drift_array) or None.
    External comparisons also include an order label for PS:
        ext_ps_data = (t_array, rel_drift_array, PS_order_ext)

    envelope : when True (default), each curve is shown as its upper envelope
        (max |error| per log-spaced bin) so the dense symplectic RKG oscillation
        reads as a clean bound instead of a noisy band. Smooth curves (PS/RK4/
        RK45) are essentially unchanged. Set False to plot every raw sample.
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    def _env(x, y):
        return _log_envelope(x, y) if envelope else (np.asarray(x), np.abs(np.asarray(y)))

    # ---- external comparison lines (plotted first, behind main lines) ----
    ln_ext = ln_extrk4 = ln_extb = ln_extc = None
    ye_ps = ye_rk4 = ye_rk45 = ye_rkg = None
    if ext_ps_data is not None:
        t_ext, drift_ext, _ = ext_ps_data
        xe, ye_ps = _env(t_ext[1:] * time_factor, drift_ext[1:])
        ln_ext, = ax.semilogy(xe, ye_ps,
                              alpha=0.8, color=COLORS["ps"], linestyle=LINESTYLES["ps"],
                              linewidth=PS_LINEWIDTH,
                              zorder=9)   # external PS also above the RKG band
    if ext_rk4_data is not None:
        t_ext, drift_ext = ext_rk4_data
        xe, ye_rk4 = _env(t_ext[1:] * time_factor, drift_ext[1:])
        ln_extrk4, = ax.semilogy(xe, ye_rk4,
                                 alpha=0.8, color=COLORS["rk4"], linestyle=LINESTYLES["rk4"])
    if ext_rk45_data is not None:
        t_ext, drift_ext = ext_rk45_data
        xe, ye_rk45 = _env(t_ext[1:] * time_factor, drift_ext[1:])
        ln_extb, = ax.semilogy(xe, ye_rk45,
                               alpha=0.8, color=COLORS["rk45"], linestyle=LINESTYLES["rk45"])
    if ext_rkg_data is not None:
        t_ext, drift_ext = ext_rkg_data
        xe, ye_rkg = _env(t_ext[1:] * time_factor, drift_ext[1:])
        ln_extc, = ax.semilogy(xe, ye_rkg,
                               alpha=0.8, color=COLORS["rkg"], linestyle=LINESTYLES["rkg"])

    # ---- main solver lines ----
    lnps = lnrk4 = lnrkg = lnrk45 = None
    y_ps = y_rk4 = y_rkg = y_rk45 = None
    if ps_data is not None:
        t_ps, drift_ps = ps_data
        x_ps, y_ps = _env(t_ps[1:] * time_factor, drift_ps[1:])
        lnps, = ax.semilogy(x_ps, y_ps,
                            label=f"PS{ps_order_label}", alpha=0.8, color=COLORS["ps"], linestyle=LINESTYLES["ps"],
                            linewidth=PS_LINEWIDTH,
                            zorder=10)   # draw PS on top so the noisy RKG band can't obscure it
    if rk4_data is not None:
        t_rk4, drift_rk4 = rk4_data
        x_rk4, y_rk4 = _env(t_rk4[1:] * time_factor, drift_rk4[1:])
        lnrk4, = ax.semilogy(x_rk4, y_rk4,
                             label='RK4', alpha=0.8, color=COLORS["rk4"], linestyle=LINESTYLES["rk4"])
    if rkg_data is not None:
        t_rkg, drift_rkg = rkg_data
        x_rkg, y_rkg = _env(t_rkg[1:] * time_factor, drift_rkg[1:])
        lnrkg, = ax.semilogy(x_rkg, y_rkg,
                             label='RKG', alpha=0.8, color=COLORS["rkg"], linestyle=LINESTYLES["rkg"])
    if rk45_data is not None:
        t_rk45, drift_rk45 = rk45_data
        x_rk45, y_rk45 = _env(t_rk45[1:] * time_factor, drift_rk45[1:])
        lnrk45, = ax.semilogy(x_rk45, y_rk45,
                              label='RK45', color=COLORS["rk45"], linestyle=LINESTYLES["rk45"])

    # ---- axis formatting ----
    ul.setup_log_axes(ax)

    # X-axis: keep the tick marks exactly as-is, drop only the label and the
    # numeric tick labels.
    ax.set_xlabel("")
    ax.tick_params(axis='x', labelbottom=False)
    ax.set_ylabel(r"$|\Delta E|/E_0$")

    if USE_PLOT_TITLES:
        ax.set_title(f"{particle_type} Relative Kinetic Energy Error in Dipole B Field")

    fig.subplots_adjust(right=0.9)
    fig.canvas.draw()

    # ---- endpoint labels (anchored at each curve's own final point, so a
    # solver that stops early is labelled where it stops, not at the axis edge) ----
    endpoints = []
    if ps_data is not None:
        endpoints.append((x_ps[-1], y_ps[-1],
                         f"PS{ps_order_label}", lnps.get_color()))
    # if ps_data is not None:
    #     endpoints.append((x_ps[-1], y_ps[-1],
    #                      f"PS13", lnps.get_color()))
    if rk4_data is not None:
        endpoints.append((x_rk4[-1], y_rk4[-1],
                         "RK4", lnrk4.get_color()))
    if rkg_data is not None:
        endpoints.append((x_rkg[-1], y_rkg[-1],
                         "RKG", lnrkg.get_color()))
    if rk45_data is not None:
        endpoints.append((x_rk45[-1], y_rk45[-1],
                         "RK45", lnrk45.get_color()))

    if ext_ps_data is not None:
        order_ext = ext_ps_data[2]
        endpoints.append((ext_ps_data[0][-1] * time_factor, ye_ps[-1],
                         f"PS{order_ext}", ln_ext.get_color()))
    if ext_rk4_data is not None:
        endpoints.append((ext_rk4_data[0][-1] * time_factor, ye_rk4[-1],
                         "RK4", ln_extrk4.get_color()))
    if ext_rk45_data is not None:
        endpoints.append((ext_rk45_data[0][-1] * time_factor, ye_rk45[-1],
                         "RK45", ln_extb.get_color()))
    if ext_rkg_data is not None:
        endpoints.append((ext_rkg_data[0][-1] * time_factor, ye_rkg[-1],
                         "RKG", ln_extc.get_color()))

    xmin, xmax = ax.get_xlim()
    ax.set_xlim(xmin, xmax * 1.05)

    ul.place_endpoint_labels(fig, ax, endpoints)

    # === Save and Close ===
    fig_path = wr.build_filename(run_folder, stem, figure_tag="KEerror", ext="png")
    plt.savefig(fig_path, dpi=600, bbox_inches="tight")
    plt.close(fig)


# =============================================================
# ============== Dragt Poincaré Surface of Section ============
# =============================================================
def poincare(
    run_folder, L_shell_dragt, gamma,
    rho_bnd, rho_dot_bnd, rho_0_sim, rho_dot_0_sim,
    crossings=None, stem="", use_titles=True, show_sim_units=True,
):
    """
    Poincaré surface of section at z=0 in Dragt dimensionless units, with a
    second set of axes (top / right) carrying the same points in the paper's
    units — rho in R_E and rho_dot in R_E per tau_0.

    crossings: tuple (rho_dragt, rho_dot_dragt, ...) collected by
               analysis_chunked, or None if no equatorial crossings found.
    show_sim_units: draw the R_E / (R_E/tau_0) axes on top and right.
    """
    fig, ax = plt.subplots(figsize=(10, 7))

    # --- Analytical boundary ---
    if rho_bnd is not None:
        ax.plot(rho_bnd,  rho_dot_bnd, 'k-', linewidth=1.5, label="Boundary")
        ax.plot(rho_bnd, -rho_dot_bnd, 'k-', linewidth=1.5)
    else:
        print("WARNING: No accessible boundary region for this energy and launch angle!")

    # --- Launch point (t=0) ---
    ax.plot(rho_0_sim / L_shell_dragt, rho_dot_0_sim * L_shell_dragt**2 / gamma, 'D',
            markerfacecolor='blue', markeredgecolor='black', markersize=6, label="Launch (t=0)")

    # --- Equatorial (z=0) crossings ---
    if crossings is None:
        print("WARNING: No equatorial crossings (z=0) found!")
    else:
        rho_dragt, rho_dot_dragt = crossings[0], crossings[1]
        ax.plot(rho_dragt, rho_dot_dragt, 'D', markerfacecolor='none',
                markeredgecolor=COLORS["ps"], markersize=4, label="Crossings")

    ax.set_xlabel(r"$\rho$  (Dragt)")
    ax.set_ylabel(r"$\dot{\rho}$  (Dragt)")
    # ax.set_xlim(0.925, 1.10)
    # ax.set_ylim(-.1, .1)

    # --- Second set of axes in the paper's units (top / right) ---
    # Exact inverse of the Dragt scaling applied to the sim values above:
    #     rho    [R_E]        = rho_dragt     * L
    #     rhodot [R_E/tau_0]  = rhodot_dragt  * gamma / L^2
    if show_sim_units and L_shell_dragt > 0 and gamma > 0:
        _L = float(L_shell_dragt)
        _g = float(gamma)
        sec_x = ax.secondary_xaxis(
            "top", functions=(lambda rd: rd * _L, lambda rr: rr / _L))
        sec_x.set_xlabel(r"$\rho$  ($R_E$)")
        sec_y = ax.secondary_yaxis(
            "right", functions=(lambda vd: vd * _g / _L**2,
                                lambda vs: vs * _L**2 / _g))
        sec_y.set_ylabel(r"$\dot{\rho}$  ($R_E/\tau_0$)")

    if use_titles:
        ax.set_title("Poincaré Surface of Section at z=0 "
                     r"(Dragt units; $R_E$ units on top/right)")
    ax.grid(True)
    ax.legend(loc="upper right", fontsize=9)
    fig.savefig(os.path.join(run_folder, f"{stem}_dragt_surface_section.png"), dpi=300)
    plt.close(fig)


# =============================================================
# ============== Gyrophase vs Magnetic Moment =================
# =============================================================
def gyrophase_mu(run_folder, gyrophase, mu_cross, stem="", use_titles=True):
    """Scatter plot of gyrophase vs magnetic moment at equatorial crossings."""

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(mu_cross, gyrophase, s=10, c='blue', alpha=0.6, edgecolors='none')
    ax.set_xlabel(r"Magnetic Moment $\mu$")
    ax.set_ylabel(r"Gyrophase $\Phi_g$ (Degrees)")
    if use_titles:
        ax.set_title("Gyrophase vs. Adiabatic Invariance at Equator")
    ax.set_ylim(-180, 180)
    ax.grid(True)
    fig.savefig(os.path.join(run_folder, f"{stem}_phase_vs_mu.png"), dpi=300)
    plt.close(fig)


# =============================================================
# ============== Polar Phase Space ============================
# =============================================================
def polar_phase_space(run_folder, gyrophase, mu_cross, stem="", use_titles=True):
    """Polar plot of gyrophase vs magnetic moment."""

    gyrophase_rad = np.radians(gyrophase)
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'projection': 'polar'})
    ax.scatter(gyrophase_rad, mu_cross, s=10, c='blue', alpha=0.6, edgecolors='none')
    if use_titles:
        ax.set_title("Phi vs Mu", va='bottom')
    fig.savefig(os.path.join(run_folder, f"{stem}_polar_phase_space.png"), dpi=300)
    plt.close(fig)


# =============================================================
# ============== Meridian Plane (Dragt Fig. 3) ================
# =============================================================
def meridian_plane(run_folder, rho_arr, z_arr, stem="", use_titles=True):
    """Trajectory in the meridian plane (rho vs z) in Dragt dimensionless units
    (lengths normalized by the shell parameter L). Kept in Dragt's units on
    purpose for direct sanity checks against his figures; see meridian_plane_RE
    for the same picture in the paper's units."""

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.plot(rho_arr, z_arr, color='blue', linewidth=0.5, alpha=0.6, label='Trajectory')
    ax.axhline(0, color='black', lw=1, ls='--', label='Equator ($z=0$)')
    ax.set_xlabel(r"$\rho$")
    ax.set_ylabel(r"$z$")
    if use_titles:
        ax.set_title(r"Meridian Plane Comparison ")
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')
    fig.savefig(os.path.join(run_folder, f"{stem}_dragt_z_vs_rho.png"), dpi=300)
    plt.close(fig)


# =============================================================
# ============== Meridian Plane (paper units, R_E) ============
# =============================================================
def meridian_plane_RE(run_folder, rho_arr, z_arr, stem="", use_titles=True):
    """Trajectory in the meridian plane (rho vs z) in R_E — the paper's units.
    Separate from ``meridian_plane`` (which stays in Dragt's L-normalized units
    for sanity checks against his figures). 
    Saved as <stem>_meridian.png.
    """
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.plot(rho_arr, z_arr, color=COLORS["ps"], linestyle=LINESTYLES["ps"],
            linewidth=PS_LINEWIDTH, alpha=0.8)
    ax.axhline(0, color='black', lw=0.8, ls='--')
    ax.set_xlabel(r"$\rho$")
    ax.set_ylabel(r"$z$")
    if use_titles:
        ax.set_title("Meridian Plane")
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig_path = wr.build_filename(run_folder, stem, figure_tag="meridian", ext="png")
    plt.savefig(fig_path, dpi=600, bbox_inches="tight")
    plt.close(fig)


# =============================================================
# ============== Adiabaticity Parameter vs Time ===============
# =============================================================
def adiabaticity(run_folder, t_arr, eps_arr, eps_initial, eps_mean, eps_max, stem="",
                 use_titles=True):
    """Adiabaticity parameter epsilon vs time (semilogy)."""

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.semilogy(t_arr, eps_arr, color=COLORS["ps"], linewidth=0.6, alpha=0.8, label=r"$\varepsilon(t)$")
    ax.axhline(1, color='k', linestyle='--', linewidth=1.0, label=r"$\varepsilon = 1$")
    ax.set_xlabel(r"$\tau / T$")
    ax.set_ylabel(r"$\varepsilon$")
    if use_titles:
        ax.set_title(r"Adiabaticity")
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    # ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1), borderaxespad=0)
    print(f"(Adiabaticity parameter, <.1 stable) epsilon:\n   initial={eps_initial:.4f}, mean={eps_mean:.4f}, max={eps_max:.4f}\n")
    fig.savefig(os.path.join(run_folder, f"{stem}_dragt_adiabaticity.png"), dpi=300, bbox_inches='tight')
    plt.close(fig)


# =============================================================
# ============== P_phi Relative Error =========================
# =============================================================
def pphi_error(
    run_folder, stem, particle_type, ps_order_label,
    USE_PLOT_TITLES, time_factor, norm_time,
    ylabel_str=r"$|\Delta P_\phi|/|P_{\phi,0}|$",
    ps_data=None, rk4_data=None, rk45_data=None, rkg_data=None,
    ext_ps_data=None, ext_rk4_data=None, ext_rk45_data=None, ext_rkg_data=None,
):
    """
    Canonical angular momentum (P_phi) relative error (log-log) for all
    enabled solvers.  Mirrors the styling of ``ke_error``.

    Each *_data argument is a tuple ``(t_array, drift_array)`` or ``None``.
    External comparisons also include an order label for PS:
        ext_ps_data = (t_array, drift_array, PS_order_ext)
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    # ---- external comparison lines (plotted first, behind main lines) ----
    ln_ext = ln_extrk4 = ln_extb = ln_extc = None
    ye_ps = ye_rk4 = ye_rk45 = ye_rkg = None
    if ext_ps_data is not None:
        t_ext, drift_ext, _ = ext_ps_data
        ye_ps = np.abs(drift_ext[1:])
        ln_ext, = ax.semilogy(t_ext[1:] * time_factor, ye_ps,
                              alpha=0.8, color=COLORS["ps"], linestyle=LINESTYLES["ps"],
                              linewidth=PS_LINEWIDTH,
                              zorder=9)   # external PS also above the RKG band
    if ext_rk4_data is not None:
        t_ext, drift_ext = ext_rk4_data
        ye_rk4 = np.abs(drift_ext[1:])
        ln_extrk4, = ax.semilogy(t_ext[1:] * time_factor, ye_rk4,
                                 alpha=0.8, color=COLORS["rk4"], linestyle=LINESTYLES["rk4"])
    if ext_rk45_data is not None:
        t_ext, drift_ext = ext_rk45_data
        ye_rk45 = np.abs(drift_ext[1:])
        ln_extb, = ax.semilogy(t_ext[1:] * time_factor, ye_rk45,
                               alpha=0.8, color=COLORS["rk45"], linestyle=LINESTYLES["rk45"])
    if ext_rkg_data is not None:
        t_ext, drift_ext = ext_rkg_data
        ye_rkg = np.abs(drift_ext[1:])
        ln_extc, = ax.semilogy(t_ext[1:] * time_factor, ye_rkg,
                               alpha=0.8, color=COLORS["rkg"], linestyle=LINESTYLES["rkg"])

    # ---- main solver lines ----
    lnps = lnrk4 = lnrkg = lnrk45 = None
    if ps_data is not None:
        t_ps, drift_ps = ps_data
        lnps, = ax.semilogy(t_ps[1:] * time_factor, np.abs(drift_ps[1:]),
                            label=f"PS{ps_order_label}", alpha=0.8,
                            color=COLORS["ps"], linestyle=LINESTYLES["ps"],
                            linewidth=PS_LINEWIDTH,
                            zorder=10)   # draw PS on top so the noisy RKG band can't obscure it
    if rk4_data is not None:
        t_rk4, drift_rk4 = rk4_data
        lnrk4, = ax.semilogy(t_rk4[1:] * time_factor, np.abs(drift_rk4[1:]),
                             label='RK4', alpha=0.8,
                             color=COLORS["rk4"], linestyle=LINESTYLES["rk4"])
    if rkg_data is not None:
        t_rkg, drift_rkg = rkg_data
        lnrkg, = ax.semilogy(t_rkg[1:] * time_factor, np.abs(drift_rkg[1:]),
                             label='RKG', alpha=0.8,
                             color=COLORS["rkg"], linestyle=LINESTYLES["rkg"])
    if rk45_data is not None:
        t_rk45, drift_rk45 = rk45_data
        lnrk45, = ax.semilogy(t_rk45[1:] * time_factor, np.abs(drift_rk45[1:]),
                              label='RK45',
                              color=COLORS["rk45"], linestyle=LINESTYLES["rk45"])

    # ---- axis formatting ----
    ul.setup_log_axes(ax)
    ax.set_xlabel(r"$\tau/T$")
    ax.set_ylabel(ylabel_str)

    if USE_PLOT_TITLES:
        ax.set_title(f"{particle_type} Relative Canonical Angular Momentum Error in Dipole B Field")

    fig.subplots_adjust(right=0.9)
    fig.canvas.draw()

    # ---- endpoint labels (anchored at each curve's own final point) ----
    endpoints = []
    # if ps_data is not None:
    #     endpoints.append((t_ps[-1] * time_factor, np.abs(drift_ps[-1]),
    #                      f"PS13", lnps.get_color()))
    if ps_data is not None:
        endpoints.append((t_ps[-1] * time_factor, np.abs(drift_ps[-1]),
                         f"PS{ps_order_label}", lnps.get_color()))        
    if rk4_data is not None:
        endpoints.append((t_rk4[-1] * time_factor, np.abs(drift_rk4[-1]),
                         "RK4", lnrk4.get_color()))
    if rkg_data is not None:
        endpoints.append((t_rkg[-1] * time_factor, np.abs(drift_rkg[-1]),
                         "RKG", lnrkg.get_color()))
    if rk45_data is not None:
        endpoints.append((t_rk45[-1] * time_factor, np.abs(drift_rk45[-1]),
                         "RK45", lnrk45.get_color()))

    if ext_ps_data is not None:
        order_ext = ext_ps_data[2]
        endpoints.append((ext_ps_data[0][-1] * time_factor, ye_ps[-1],
                         f"PS{order_ext}", ln_ext.get_color()))
    if ext_rk4_data is not None:
        endpoints.append((ext_rk4_data[0][-1] * time_factor, ye_rk4[-1],
                         "RK4", ln_extrk4.get_color()))
    if ext_rk45_data is not None:
        endpoints.append((ext_rk45_data[0][-1] * time_factor, ye_rk45[-1],
                         "RK45", ln_extb.get_color()))
    if ext_rkg_data is not None:
        endpoints.append((ext_rkg_data[0][-1] * time_factor, ye_rkg[-1],
                         "RKG", ln_extc.get_color()))

    xmin, xmax = ax.get_xlim()
    ax.set_xlim(xmin, xmax * 1.05)

    ul.place_endpoint_labels(fig, ax, endpoints)

    # === Save and Close ===
    fig_path = wr.build_filename(run_folder, stem, figure_tag="Pphierror", ext="png")
    plt.savefig(fig_path, dpi=600, bbox_inches="tight")
    plt.close(fig)


# =============================================================
# ============== Mu Relative Error (full run) =================
# =============================================================
def mu_error(
    run_folder, stem, particle_type, ps_order_label,
    USE_PLOT_TITLES, time_factor, norm_time,
    ylabel_str=r"$|\Delta \mu_n|/\mu_0$",
    ps_data=None, rk4_data=None, rk45_data=None, rkg_data=None,
    ext_ps_data=None, ext_rk4_data=None, ext_rk45_data=None, ext_rkg_data=None,
):
    """
    Magnetic moment relative error (log-log) over the FULL run for all
    enabled solvers.  Mirrors the styling of ``pphi_error`` / ``ke_error``.
    (The windowed ``mu_deviation`` plot below covers only the gyro window.)

    Each *_data argument is a tuple ``(t_array, drift_array)`` or ``None``.
    External comparisons also include an order label for PS:
        ext_ps_data = (t_array, drift_array, PS_order_ext)
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    # ---- external comparison lines (plotted first, behind main lines) ----
    ln_ext = ln_extrk4 = ln_extb = ln_extc = None
    ye_ps = ye_rk4 = ye_rk45 = ye_rkg = None
    if ext_ps_data is not None:
        t_ext, drift_ext, _ = ext_ps_data
        ye_ps = np.abs(drift_ext[1:])
        ln_ext, = ax.semilogy(t_ext[1:] * time_factor, ye_ps,
                              alpha=0.8, color=COLORS["ps"], linestyle=LINESTYLES["ps"],
                              linewidth=PS_LINEWIDTH,
                              zorder=9)   # external PS also above the RKG band
    if ext_rk4_data is not None:
        t_ext, drift_ext = ext_rk4_data
        ye_rk4 = np.abs(drift_ext[1:])
        ln_extrk4, = ax.semilogy(t_ext[1:] * time_factor, ye_rk4,
                                 alpha=0.8, color=COLORS["rk4"], linestyle=LINESTYLES["rk4"])
    if ext_rk45_data is not None:
        t_ext, drift_ext = ext_rk45_data
        ye_rk45 = np.abs(drift_ext[1:])
        ln_extb, = ax.semilogy(t_ext[1:] * time_factor, ye_rk45,
                               alpha=0.8, color=COLORS["rk45"], linestyle=LINESTYLES["rk45"])
    if ext_rkg_data is not None:
        t_ext, drift_ext = ext_rkg_data
        ye_rkg = np.abs(drift_ext[1:])
        ln_extc, = ax.semilogy(t_ext[1:] * time_factor, ye_rkg,
                               alpha=0.8, color=COLORS["rkg"], linestyle=LINESTYLES["rkg"])

    # ---- main solver lines ----
    lnps = lnrk4 = lnrkg = lnrk45 = None
    if ps_data is not None:
        t_ps, drift_ps = ps_data
        lnps, = ax.semilogy(t_ps[1:] * time_factor, np.abs(drift_ps[1:]),
                            label=f"PS{ps_order_label}", alpha=0.8,
                            color=COLORS["ps"], linestyle=LINESTYLES["ps"],
                            linewidth=PS_LINEWIDTH,
                            zorder=10)   # draw PS on top so the noisy RKG band can't obscure it
    if rk4_data is not None:
        t_rk4, drift_rk4 = rk4_data
        lnrk4, = ax.semilogy(t_rk4[1:] * time_factor, np.abs(drift_rk4[1:]),
                             label='RK4', alpha=0.8,
                             color=COLORS["rk4"], linestyle=LINESTYLES["rk4"])
    if rkg_data is not None:
        t_rkg, drift_rkg = rkg_data
        lnrkg, = ax.semilogy(t_rkg[1:] * time_factor, np.abs(drift_rkg[1:]),
                             label='RKG', alpha=0.8,
                             color=COLORS["rkg"], linestyle=LINESTYLES["rkg"])
    if rk45_data is not None:
        t_rk45, drift_rk45 = rk45_data
        lnrk45, = ax.semilogy(t_rk45[1:] * time_factor, np.abs(drift_rk45[1:]),
                              label='RK45',
                              color=COLORS["rk45"], linestyle=LINESTYLES["rk45"])

    # ---- axis formatting ----
    ul.setup_log_axes(ax)
    ax.set_xlabel(r"$\tau/T$")
    ax.set_ylabel(ylabel_str)

    if USE_PLOT_TITLES:
        ax.set_title(f"{particle_type} Relative Magnetic Moment Error in Dipole B Field")

    fig.subplots_adjust(right=0.9)
    fig.canvas.draw()

    # ---- endpoint labels (anchored at each curve's own final point) ----
    endpoints = []
    if ps_data is not None:
        endpoints.append((t_ps[-1] * time_factor, np.abs(drift_ps[-1]),
                         f"PS{ps_order_label}", lnps.get_color()))
    if rk4_data is not None:
        endpoints.append((t_rk4[-1] * time_factor, np.abs(drift_rk4[-1]),
                         "RK4", lnrk4.get_color()))
    if rkg_data is not None:
        endpoints.append((t_rkg[-1] * time_factor, np.abs(drift_rkg[-1]),
                         "RKG", lnrkg.get_color()))
    if rk45_data is not None:
        endpoints.append((t_rk45[-1] * time_factor, np.abs(drift_rk45[-1]),
                         "RK45", lnrk45.get_color()))

    if ext_ps_data is not None:
        order_ext = ext_ps_data[2]
        endpoints.append((ext_ps_data[0][-1] * time_factor, ye_ps[-1],
                         f"PS{order_ext}", ln_ext.get_color()))
    if ext_rk4_data is not None:
        endpoints.append((ext_rk4_data[0][-1] * time_factor, ye_rk4[-1],
                         "RK4", ln_extrk4.get_color()))
    if ext_rk45_data is not None:
        endpoints.append((ext_rk45_data[0][-1] * time_factor, ye_rk45[-1],
                         "RK45", ln_extb.get_color()))
    if ext_rkg_data is not None:
        endpoints.append((ext_rkg_data[0][-1] * time_factor, ye_rkg[-1],
                         "RKG", ln_extc.get_color()))

    xmin, xmax = ax.get_xlim()
    ax.set_xlim(xmin, xmax * 1.05)

    ul.place_endpoint_labels(fig, ax, endpoints)

    # === Save and Close ===
    fig_path = wr.build_filename(run_folder, stem, figure_tag="MUerror", ext="png")
    plt.savefig(fig_path, dpi=600, bbox_inches="tight")
    plt.close(fig)


# =============================================================
# ============== Invariants Overlay (E, P_phi, mu) ============
# =============================================================
# Colours here encode the QUANTITY, not the solver, so they deliberately avoid
# every entry in COLORS (PS green, RK45 orange, RK4 reddish-purple, RKG red).
INVARIANT_STYLE = {
    "ke":   dict(color="#0072B2", linestyle="-", label=r"$|\Delta E|/E_0$"),               # blue
    "pphi": dict(color="#D51900", linestyle="-", label=r"$|\Delta P_\phi|/|P_{\phi,0}|$"),  # vermillion
    "mu":   dict(color="#000000", linestyle="-",  label=r"$|\Delta \mu|/\mu_\emptyset$"),          # black
}


def invariants_overlay(
    run_folder, stem, particle_type, ps_order_label,
    USE_PLOT_TITLES, time_factor,
    ke_data=None, pphi_data=None, mu_data=None,
):
    """
    Single-solver (PS) overview: relative errors of all three invariants —
    kinetic energy, canonical angular momentum, and magnetic moment — on ONE
    log-log axis over the full run. The point of the shared axis is the
    contrast: E and P_phi sit at round-off while mu is bounded but ripples
    orders of magnitude above them.

    Each *_data argument is a (t_array, drift_array) tuple (the ``*_ps``
    log-spaced tuples from compute_ke_errors / compute_pphi_errors /
    compute_mu_errors) or None to omit that curve.
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    lines = []
    for key, data in (("ke", ke_data), ("pphi", pphi_data), ("mu", mu_data)):
        if data is None:
            continue
        t, drift = data
        st = INVARIANT_STYLE[key]
        ln, = ax.semilogy(t[1:] * time_factor, np.abs(drift[1:]),
                          color=st["color"], linestyle=st["linestyle"],
                          linewidth=0.6, alpha=0.9, label=st["label"],
                          zorder=10 if key == "mu" else 5)
        lines.append((t[-1] * time_factor, float(np.abs(drift[-1])), st["label"], st["color"]))

    # ---- axis formatting ----
    ul.setup_log_axes(ax)
    ax.set_xlabel(r"$\tau/T$")
    ax.set_ylabel("Relative Variations")

    if USE_PLOT_TITLES:
        ax.set_title(f"{particle_type} Invariant Variations (PS{ps_order_label}) in Dipole B Field")

    fig.subplots_adjust(right=0.9)
    fig.canvas.draw()

    # ---- endpoint labels (house style; no legend, matching the other error plots) ----
    xmin, xmax = ax.get_xlim()
    ax.set_xlim(xmin, xmax * 1.05)
    ul.place_endpoint_labels(fig, ax, lines)

    # === Save and Close ===
    fig_path = wr.build_filename(run_folder, stem, figure_tag="invariants", ext="png")
    plt.savefig(fig_path, dpi=600, bbox_inches="tight")
    plt.close(fig)


# =====================================================
# ============== Magnetic Moment Deviations ===========
# =====================================================
def mu_deviation(
    run_folder, stem, particle_type, ps_order_label,
    USE_PLOT_TITLES,
    ps_data=None, rk4_data=None, rk45_data=None, rkg_data=None,
    eq_data=None,
):
    """
    Magnetic moment relative deviation (semilogy) for all enabled solvers.

    Each *_data argument is a tuple (t_array, mudrift_array) or None.
    eq_data is an optional (t_array, mudrift_array) of equatorial-crossing
    samples, drawn as black dots with their own legend entry.
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    # ---- solver lines ----
    lnps = lnrk4 = lnrkg = lnrk45 = None
    if rk45_data is not None:
        t_rk45, mudrift_rk45 = rk45_data
        lnrk45, = ax.semilogy(t_rk45, mudrift_rk45, label="RK45", color=COLORS["rk45"], linestyle=LINESTYLES["rk45"])
    if rk4_data is not None:
        t_rk4, mudrift_rk4 = rk4_data
        lnrk4, = ax.semilogy(t_rk4, mudrift_rk4, label="RK4", alpha=0.3, color=COLORS["rk4"], linestyle=LINESTYLES["rk4"])
    if rkg_data is not None:
        t_rkg, mudrift_rkg = rkg_data
        lnrkg, = ax.semilogy(t_rkg, mudrift_rkg, label="RKG", alpha=0.3, color=COLORS["rkg"], linestyle=LINESTYLES["rkg"])
    if ps_data is not None:
        t_ps, mudrift_ps = ps_data
        lnps, = ax.semilogy(t_ps, mudrift_ps, label=f"PS{ps_order_label}", linewidth=PS_LINEWIDTH, color=COLORS["ps"], linestyle=LINESTYLES["ps"],
                            zorder=10)   # draw PS on top so the noisy RKG band can't obscure it

    # ---- equatorial-crossing dots (optional; omitted when eq_data is None) ----
    ln_eq = None
    if eq_data is not None:
        t_eq, mudrift_eq = eq_data
        ln_eq, = ax.semilogy(t_eq, mudrift_eq, linestyle="none", marker="o",
                             markersize=3.5, color="black",
                             label="Equatorial Crossing", zorder=11)

    # ---- axis formatting ----
    ax.margins(x=0.01)
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=100))
    ax.yaxis.set_major_formatter(LogFormatterSciNotation(base=10.0))
    ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=[]))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.grid(True, which="major", linestyle="--", linewidth=0.7)
    ax.get_xaxis().get_major_formatter().set_useOffset(False)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.set_xlabel(r"$\tau/T$")
    ax.set_ylabel(r"$|\Delta \mu|/\mu_\emptyset$")

    if USE_PLOT_TITLES:
        ax.set_title(f"{particle_type} Magnetic Moment Variations in Dipole B Field")

    fig.subplots_adjust(right=0.9)
    fig.canvas.draw()

    # ---- endpoint labels ----
    endpoints = []
    if rk45_data is not None:
        endpoints.append((t_rk45[-1], float(np.abs(mudrift_rk45[-1])), "RK45", lnrk45.get_color()))
    if rk4_data is not None:
        endpoints.append((t_rk4[-1], float(np.abs(mudrift_rk4[-1])), "RK4", lnrk4.get_color()))
    if rkg_data is not None:
        endpoints.append((t_rkg[-1], float(np.abs(mudrift_rkg[-1])), "RKG", lnrkg.get_color()))
    if ps_data is not None:
        endpoints.append((t_ps[-1], float(np.abs(mudrift_ps[-1])), f"PS{ps_order_label}", lnps.get_color()))

    ul.place_endpoint_labels(fig, ax, endpoints)

    # Solvers are labelled by endpoint text, so the legend carries only the
    # equatorial-crossing marker (when present). Boxed with an opaque white
    # background so it stays legible over the data; loc="best" lets
    # matplotlib pick the position with the least data overlap.
    if ln_eq is not None:
        ax.legend(handles=[ln_eq], loc="best", frameon=True,
                  facecolor="white", framealpha=1.0, edgecolor="0.8")

    # === Save and Close ===
    fig_path_mu = wr.build_filename(run_folder, stem, figure_tag="mu", ext="png")
    plt.savefig(fig_path_mu, dpi=600, bbox_inches="tight")
    _close_or_show(fig)   # pops up for zooming when PS_INTERACTIVE=1 (this plot only)


# =====================================================
# ====== Magnetic Moment Shape (instantaneous μ/μ0) ===
# =====================================================
def mu_shape(
    run_folder, stem, particle_type, ps_order_label,
    USE_PLOT_TITLES,
    ps_data=None, rk4_data=None, rk45_data=None, rkg_data=None,
    eq_data=None,
):
    """
    Instantaneous magnetic moment μ/μ₀ vs time over the analysis window,
    on linear axes. Unlike ``mu_deviation`` (which shows the conservation
    error), this shows the gyration/bounce-scale *shape* of μ as the particle
    samples the field along its orbit. Initial gyrophase shifts where the
    ripple starts, not its amplitude, so the shape is φ-robust.

    Each *_data argument is a tuple (t_array, mu_ratio_array) or None.
    eq_data is an optional (t_array, mu_ratio_array) of equatorial-crossing
    samples, drawn as black dots with their own legend entry.
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    lnps = lnrk4 = lnrkg = lnrk45 = None
    if rk45_data is not None:
        t_rk45, mu_rk45 = rk45_data
        lnrk45, = ax.plot(t_rk45, mu_rk45, label="RK45",
                          color=COLORS["rk45"], linestyle=LINESTYLES["rk45"])
    if rk4_data is not None:
        t_rk4, mu_rk4 = rk4_data
        lnrk4, = ax.plot(t_rk4, mu_rk4, label="RK4", alpha=0.7,
                         color=COLORS["rk4"], linestyle=LINESTYLES["rk4"])
    if rkg_data is not None:
        t_rkg, mu_rkg = rkg_data
        lnrkg, = ax.plot(t_rkg, mu_rkg, label="RKG", alpha=0.7,
                         color=COLORS["rkg"], linestyle=LINESTYLES["rkg"])
    if ps_data is not None:
        t_ps, mu_ps = ps_data
        lnps, = ax.plot(t_ps, mu_ps, label=f"PS{ps_order_label}", linewidth=PS_LINEWIDTH,
                        color=COLORS["ps"], linestyle=LINESTYLES["ps"],
                        zorder=10)   # PS on top

    # ---- equatorial-crossing dots (optional; omitted when eq_data is None) ----
    ln_eq = None
    if eq_data is not None:
        t_eq, mu_eq = eq_data
        ln_eq, = ax.plot(t_eq, mu_eq, linestyle="none", marker="o",
                         markersize=3.5, color="black",
                         label="Equatorial Crossing", zorder=11)

    # =================== ONE-OFF: 20-YEAR RUN, FINAL WINDOW ===================
    # COMMENT THIS BLOCK OUT FOR EVERY OTHER RUN.
    # Re-references the abscissa to the END of the record, so that 0 is the last
    # step of the run and the panel reads as "the final gyroperiods" rather than
    # as six unreadable labels near 2.7e10. Shifts the plotted x values only; the
    # underlying data is untouched. _T_END must equal the run's total tau/T.
    #
    # _T_END = 2.7e10
    # for _ln in ax.get_lines():
    #     _ln.set_xdata(np.asarray(_ln.get_xdata(), dtype=float) - _T_END)
    # ax.relim()
    # ax.autoscale_view()
    # _XLABEL_ONEOFF = r"$\tau/T - 2.7\times10^{10}$"
    # ==========================================================================

    # ---- axis formatting (linear axes — the shape, not a log error) ----
    ax.margins(x=0.01)
    ax.grid(True, which="major", linestyle="--", linewidth=0.7)
    ax.get_xaxis().get_major_formatter().set_useOffset(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.set_xlabel(locals().get("_XLABEL_ONEOFF", r"$\tau/T$"))
    ax.set_ylabel(r"$\mu/\mu_\emptyset$")

    if USE_PLOT_TITLES:
        ax.set_title(f"{particle_type} Magnetic Moment in Dipole B Field")

    fig.subplots_adjust(right=0.9)
    fig.canvas.draw()

    # ---- trailing-edge labels (same style as mu_deviation, no legend) ----
    # endpoints = []
    # if rk45_data is not None:
    #     endpoints.append((t_rk45[-1], float(mu_rk45[-1]), "RK45", lnrk45.get_color()))
    # if rk4_data is not None:
    #     endpoints.append((t_rk4[-1], float(mu_rk4[-1]), "RK4", lnrk4.get_color()))
    # if rkg_data is not None:
    #     endpoints.append((t_rkg[-1], float(mu_rkg[-1]), "RKG", lnrkg.get_color()))
    # if ps_data is not None:
    #     endpoints.append((t_ps[-1], float(mu_ps[-1]), f"PS{ps_order_label}", lnps.get_color()))
    # ul.place_endpoint_labels(fig, ax, endpoints)


    if ln_eq is not None:
        ax.legend(handles=[ln_eq], bbox_to_anchor=(1.0, 1.1), loc="upper right", frameon=True,
                  facecolor="white", framealpha=1.0, edgecolor="0.8")
        
    # === Save and Close ===
    fig_path = wr.build_filename(run_folder, stem, figure_tag="muShape", ext="png")
    plt.savefig(fig_path, dpi=600, bbox_inches="tight")
    plt.close(fig)

