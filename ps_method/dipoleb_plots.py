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

Dragt:
    poincare             — Poincaré surface of section (rho vs rho_dot)
    gyrophase_mu         — gyrophase vs magnetic moment at crossings
    polar_phase_space    — polar phase space (mu, gyrophase)
    meridian_plane       — meridian plane projection (rho/L vs z/L)
    adiabaticity         — adiabaticity parameter epsilon vs time

Conservation:
    pphi_error           — P_phi relative error vs time
    mu_deviation         — magnetic moment deviation vs time

Helpers:
    prepare_slice_window — extract a time window for slice plots
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, LogFormatterSciNotation, NullFormatter, FuncFormatter
from ps_method.writers import build_filename
from ps_method.utils import sparse_labels, data_to_fig, setup_log_axes, place_endpoint_labels, npfloat
from ps_method.utils import slice_solution_dipoleb as slice_solution
import h5py
import os

# =====================================================
# ============== Full 2D Trajectory Plot ==============
# =====================================================
def full_2d(
    summary, run_folder, stem, particle_type, plotbounds, ps_order_label,
    USE_PLOT_TITLES, USE_RK45, USE_RK4, USE_RKG, USE_PS,
    solution_rk45=None, solution_rk4=None, solution_rkg=None,
    x_ps_plot=None, y_ps_plot=None,
):
    """Full-run 2D trajectory (x vs y) for all enabled solvers."""
    fig, ax = plt.subplots(figsize=(10, 8))

    if USE_RK45:
        ax.plot(solution_rk45.y[0], solution_rk45.y[1], label='RK45', color='#E69F00', linestyle='--')
    if USE_RK4:
        ax.plot(solution_rk4[0], solution_rk4[1], label='RK4', alpha=0.8, color='#CC79A7', linestyle='-.')
    if USE_RKG:
        ax.plot(solution_rkg[:, 0], solution_rkg[:, 1], label='RKG', alpha=0.8, color='#CC0000', linestyle='-.')
    if USE_PS:
        ax.plot(x_ps_plot, y_ps_plot, label=f"PS{ps_order_label}", alpha=0.7, color="#009E73", linestyle=":")

    ax.set_xlabel(r"x")
    ax.set_ylabel(r"y")
    ax.ticklabel_format(style='plain', useOffset=False, axis='both')
    if USE_PLOT_TITLES:
        ax.set_title(f"2D {particle_type} Trajectory in Dipole B Field")

    ax.legend(loc="upper right")
    ax.set_xlim(-plotbounds, plotbounds)
    ax.set_ylim(-plotbounds, plotbounds)
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True)

    fig.canvas.draw()
    fig_path = build_filename(summary, run_folder, stem, figure_tag="2D", ext="png")
    plt.savefig(fig_path, dpi=600, bbox_inches="tight")
    plt.close(fig)


# =====================================================
# ============== Full 3D Trajectory Plot ==============
# =====================================================
def full_3d(
    summary, run_folder, stem, particle_type, plotbounds, ps_order_label,
    USE_PLOT_TITLES, USE_RK45, USE_RK4, USE_RKG, USE_PS,
    solution_rk45=None, solution_rk4=None, solution_rkg=None,
    x_ps_plot=None, y_ps_plot=None, z_ps_plot=None,
):
    """Full-run 3D trajectory for all enabled solvers."""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    if USE_RK45:
        ax.plot(solution_rk45.y[0], solution_rk45.y[1], solution_rk45.y[2], label="RK45", color='#E69F00', linestyle='--')
    if USE_RK4:
        ax.plot(solution_rk4[0], solution_rk4[1], solution_rk4[2], label='RK4', alpha=0.8, color='#CC79A7', linestyle='-.')
    if USE_RKG:
        ax.plot(solution_rkg[:, 0], solution_rkg[:, 1], solution_rkg[:, 2], label='RKG', alpha=0.8, color='#CC0000', linestyle='-.')
    if USE_PS:
        ax.plot(x_ps_plot, y_ps_plot, z_ps_plot, label=f"PS{ps_order_label}", alpha=0.7, color="#009E73", linestyle=":")

    ax.set_xlim(-plotbounds, plotbounds)
    ax.set_ylim(-plotbounds, plotbounds)
    ax.set_zlim(-plotbounds, plotbounds)

    ax.set_xlabel(r'X')
    ax.set_ylabel(r'Y')
    ax.set_zlabel(r'Z')
    if USE_PLOT_TITLES:
        ax.set_title(f"3D {particle_type} Trajectory in Dipole B Field")
    ax.legend(loc="upper right")

    fig.canvas.draw()
    fig_path = build_filename(summary, run_folder, stem, figure_tag="3D", ext="png")
    plt.savefig(fig_path, dpi=600, bbox_inches="tight")
    plt.close(fig)


# =====================================================
# ================ 2D Trajectory Slice ================
# =====================================================
def slice_2d(
    summary, run_folder, stem, particle_type, ps_order_label,
    USE_PLOT_TITLES, USE_RK45, USE_RK4, USE_RKG, USE_PS,
    rk45_x_slice=None, rk45_y_slice=None,
    rk4_x_slice=None, rk4_y_slice=None,
    rkg_x_slice=None, rkg_y_slice=None,
    ps_x_slice=None, ps_y_slice=None,
):
    """Time-windowed 2D trajectory slice for all enabled solvers."""
    fig, ax = plt.subplots(figsize=(10, 7))

    if USE_RK45:
        ax.plot(rk45_x_slice, rk45_y_slice, label='RK45', color='#E69F00', linestyle='--')
    if USE_RK4:
        ax.plot(rk4_x_slice, rk4_y_slice, label='RK4', alpha=0.8, color='#CC79A7', linestyle='-.')
    if USE_RKG:
        ax.plot(rkg_x_slice, rkg_y_slice, label='RKG', alpha=0.8, color='#CC0000', linestyle='-.')
    if USE_PS:
        ax.plot(ps_x_slice, ps_y_slice, label=f"PS{ps_order_label}", alpha=0.8, color='#009E73', linestyle=':')

    ax.set_xlabel(r"x")
    ax.set_ylabel(r"y")
    if USE_PLOT_TITLES:
        ax.set_title(f"2D Trajectory of Slice {particle_type} Orbits in Dipole B Field")
    ax.axis('equal')
    ax.legend(loc="upper right")
    ax.grid(True)

    fig.canvas.draw()
    fig_path = build_filename(summary, run_folder, stem, figure_tag="2Dslice", ext="png")
    plt.savefig(fig_path, dpi=600, bbox_inches="tight")
    plt.close(fig)


# =====================================================
# ================ 3D Trajectory Slice ================
# =====================================================
def slice_3d(
    summary, run_folder, stem, particle_type, plotbounds, ps_order_label,
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
        ax.plot(rk45_x_slice, rk45_y_slice, rk45_z_slice, label='RK45', color='#E69F00', linestyle='--')
    if USE_RK4:
        ax.plot(rk4_x_slice, rk4_y_slice, rk4_z_slice, label='RK4', alpha=0.8, color='#CC79A7', linestyle='-.')
    if USE_RKG:
        ax.plot(rkg_x_slice, rkg_y_slice, rkg_z_slice, label='RKG', alpha=0.8, color='#CC0000', linestyle='-.')
    if USE_PS:
        ax.plot(ps_x_slice, ps_y_slice, ps_z_slice, label=f"PS{ps_order_label}", alpha=0.8, color='#009E73', linestyle=':')

    ax.set_xlim(-plotbounds, plotbounds)
    ax.set_ylim(-plotbounds, plotbounds)
    ax.set_zlim(-plotbounds, plotbounds)
    ax.legend(loc="upper right")
    ax.grid(True)

    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$y$')
    ax.set_zlabel(r'$z$')
    if USE_PLOT_TITLES:
        ax.set_title(f'3D Trajectory Slice of {particle_type} Orbits in Dipole B Field')
    ax.legend(loc="upper right")

    fig.canvas.draw()
    fig_path = build_filename(summary, run_folder, stem, figure_tag="3Dslice", ext="png")
    plt.savefig(fig_path, dpi=600, bbox_inches="tight")
    plt.close(fig)


# =====================================================
# ============== KE Relative Error Plot ===============
# =====================================================
def ke_error(
    summary, run_folder, stem, particle_type, ps_order_label,
    USE_PLOT_TITLES, time_factor, norm_time,
    ps_data=None, rk4_data=None, rk45_data=None, rkg_data=None,
    ext_ps_data=None, ext_rk4_data=None, ext_rk45_data=None, ext_rkg_data=None,
):
    """
    Kinetic energy relative error (log-log) for all enabled solvers.

    Each *_data argument is a tuple (t_array, rel_drift_array) or None.
    External comparisons also include an order label for PS:
        ext_ps_data = (t_array, rel_drift_array, PS_order_ext)
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    # ---- external comparison lines (plotted first, behind main lines) ----
    ln_ext = ln_extrk4 = ln_extb = ln_extc = None
    if ext_ps_data is not None:
        t_ext, drift_ext, _ = ext_ps_data
        ln_ext, = ax.semilogy(t_ext[1:] * time_factor, np.abs(drift_ext[1:]),
                              alpha=0.8, color='#009E73', linestyle=':')
    if ext_rk4_data is not None:
        t_ext, drift_ext = ext_rk4_data
        ln_extrk4, = ax.semilogy(t_ext[1:] * time_factor, np.abs(drift_ext[1:]),
                                 alpha=0.8, color='#CC79A7', linestyle='-.')
    if ext_rk45_data is not None:
        t_ext, drift_ext = ext_rk45_data
        ln_extb, = ax.semilogy(t_ext[1:] * time_factor, np.abs(drift_ext[1:]),
                               alpha=0.8, color='#E69F00', linestyle='--')
    if ext_rkg_data is not None:
        t_ext, drift_ext = ext_rkg_data
        ln_extc, = ax.semilogy(t_ext[1:] * time_factor, np.abs(drift_ext[1:]),
                               alpha=0.8, color='#CC0000', linestyle='-.')

    # ---- main solver lines ----
    lnps = lnrk4 = lnrkg = lnrk45 = None
    if ps_data is not None:
        t_ps, drift_ps = ps_data
        lnps, = ax.semilogy(t_ps[1:] * time_factor, np.abs(drift_ps[1:]),
                            label=f"PS{ps_order_label}", alpha=0.8, color="#009E73", linestyle=":")
    if rk4_data is not None:
        t_rk4, drift_rk4 = rk4_data
        lnrk4, = ax.semilogy(t_rk4[1:] * time_factor, np.abs(drift_rk4[1:]),
                             label='RK4', alpha=0.8, color='#CC79A7', linestyle='-.')
    if rkg_data is not None:
        t_rkg, drift_rkg = rkg_data
        lnrkg, = ax.semilogy(t_rkg[1:] * time_factor, np.abs(drift_rkg[1:]),
                             label='RKG', alpha=0.8, color='#CC0000', linestyle='-.')
    if rk45_data is not None:
        t_rk45, drift_rk45 = rk45_data
        lnrk45, = ax.semilogy(t_rk45[1:] * time_factor, np.abs(drift_rk45[1:]),
                              label='RK45', color='#E69F00', linestyle='--')

    # ---- axis formatting ----
    setup_log_axes(ax)

    ax.set_xlabel(r"$\tau/T$")
    ax.set_ylabel(r"$|\Delta E|/E_0$")

    if USE_PLOT_TITLES:
        ax.set_title(f"{particle_type} Relative Kinetic Energy Error in Dipole B Field")

    fig.subplots_adjust(right=0.9)
    fig.canvas.draw()

    # ---- endpoint labels ----
    endpoints = []
    if ps_data is not None:
        endpoints.append((norm_time * time_factor, np.abs(drift_ps[-1]),
                         f"PS{ps_order_label}", lnps.get_color()))
    if rk4_data is not None:
        endpoints.append((norm_time * time_factor, np.abs(drift_rk4[-1]),
                         "RK4", lnrk4.get_color()))
    if rkg_data is not None:
        endpoints.append((norm_time * time_factor, np.abs(drift_rkg[-1]),
                         "RKG", lnrkg.get_color()))
    if rk45_data is not None:
        endpoints.append((norm_time * time_factor, np.abs(drift_rk45[-1]),
                         "RK45", lnrk45.get_color()))

    if ext_ps_data is not None:
        t_ext, drift_ext, order_ext = ext_ps_data
        endpoints.append((t_ext[-1] * time_factor, np.abs(drift_ext[-1]),
                         f"PS{order_ext}", ln_ext.get_color()))
    if ext_rk4_data is not None:
        t_ext, drift_ext = ext_rk4_data
        endpoints.append((t_ext[-1] * time_factor, np.abs(drift_ext[-1]),
                         "RK4", ln_extrk4.get_color()))
    if ext_rk45_data is not None:
        t_ext, drift_ext = ext_rk45_data
        endpoints.append((t_ext[-1] * time_factor, np.abs(drift_ext[-1]),
                         "RK45", ln_extb.get_color()))
    if ext_rkg_data is not None:
        t_ext, drift_ext = ext_rkg_data
        endpoints.append((t_ext[-1] * time_factor, np.abs(drift_ext[-1]),
                         "RKG", ln_extc.get_color()))

    xmin, xmax = ax.get_xlim()
    ax.set_xlim(xmin, xmax * 1.05)

    place_endpoint_labels(fig, ax, endpoints)

    # === Save and Close ===
    fig.canvas.draw()
    fig_path = build_filename(summary, run_folder, stem, figure_tag="KEerror", ext="png")
    plt.savefig(fig_path, dpi=600, bbox_inches="tight")
    plt.close(fig)


# =============================================================
# ============== Dragt Poincaré Surface of Section ============
# =============================================================
def poincare(
    run_folder, L_shell_dragt, gamma,
    rho_bnd, rho_dot_bnd, rho_0_sim, rho_dot_0_sim,
    crossings=None, stem="",
):
    """
    Poincaré surface of section at z=0 in Dragt dimensionless units.

    crossings: tuple (rho_dragt, rho_dot_dragt, ...) from compute_z_crossings,
               or None if no equatorial crossings found.
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
                markeredgecolor='#009E73', markersize=4, label="Crossings")

    ax.set_xlabel(r"$\rho$ (Dimensionless)")
    ax.set_ylabel(r"$\dot{\rho}$ (Dimensionless)")
    ax.set_title("Dragt Poincaré Surface of Section at z=0")
    ax.grid(True)
    ax.legend(loc="upper right", fontsize=9)
    fig.canvas.draw()
    fig.savefig(os.path.join(run_folder, f"{stem}_dragt_surface_section.png"), dpi=300)
    plt.close(fig)


# =============================================================
# ============== Gyrophase vs Magnetic Moment =================
# =============================================================
def gyrophase_mu(run_folder, gyrophase, mu_cross, stem=""):
    """Scatter plot of gyrophase vs magnetic moment at equatorial crossings."""

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(mu_cross, gyrophase, s=10, c='blue', alpha=0.6, edgecolors='none')
    ax.set_xlabel(r"Magnetic Moment $\mu$ (Normalized)")
    ax.set_ylabel(r"Gyrophase $\Phi_g$ (Degrees)")
    ax.set_title("Gyrophase vs. Adiabatic Invariance at Equator")
    ax.set_ylim(-180, 180)
    ax.grid(True)
    fig.savefig(os.path.join(run_folder, f"{stem}_phase_vs_mu.png"), dpi=300)
    plt.close(fig)


# =============================================================
# ============== Polar Phase Space ============================
# =============================================================
def polar_phase_space(run_folder, gyrophase, mu_cross, stem=""):
    """Polar plot of gyrophase vs magnetic moment."""

    gyrophase_rad = np.radians(gyrophase)
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'projection': 'polar'})
    ax.scatter(gyrophase_rad, mu_cross, s=10, c='blue', alpha=0.6, edgecolors='none')
    ax.set_title("Phi vs Mu", va='bottom')
    fig.savefig(os.path.join(run_folder, f"{stem}_polar_phase_space.png"), dpi=300)
    plt.close(fig)


# =============================================================
# ============== Meridian Plane (Dragt Fig. 3) ================
# =============================================================
def meridian_plane(run_folder, rho_arr, z_arr, stem=""):
    """Trajectory in the meridian plane (rho vs z) in Dragt dimensionless units."""

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.plot(rho_arr, z_arr, color='blue', linewidth=0.5, alpha=0.6, label='Trajectory')
    ax.axhline(0, color='black', lw=1, ls='--', label='Equator ($z=0$)')
    ax.set_xlabel(r"$\rho$ (Dragt Dimensionless)")
    ax.set_ylabel(r"$z$ (Dragt Dimensionless)")
    ax.set_title(r"Meridian Plane Comparison ")
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')
    fig.savefig(os.path.join(run_folder, f"{stem}_dragt_z_vs_rho.png"), dpi=300)
    plt.close(fig)


# =============================================================
# ============== Adiabaticity Parameter vs Time ===============
# =============================================================
def adiabaticity(run_folder, t_arr, eps_arr, eps_initial, eps_mean, eps_max, stem=""):
    """Adiabaticity parameter epsilon vs time (semilogy)."""

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.semilogy(t_arr, eps_arr, color='#009E73', linewidth=0.6, alpha=0.8, label=r"$\epsilon(t)$")
    ax.axhline(0.1, color='k', linestyle='--', linewidth=1.0, label=r"$\epsilon = 0.1$ (GC limit)")
    ax.set_xlabel(r"$\tau / T$ (Equatorial Gyroperiods)")
    ax.set_ylabel(r"$\epsilon = r_g \cdot |\nabla_\perp B| / B$")
    ax.set_title(r"Adiabaticity Parameter $\epsilon \approx 3 r_g / r$ vs Time")
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1), borderaxespad=0)
    print(f"(Adiabaticity parameter, <.1 stable) epsilon:\n   initial={eps_initial:.4f}, mean={eps_mean:.4f}, max={eps_max:.4f}\n")
    fig.canvas.draw()
    fig.savefig(os.path.join(run_folder, f"{stem}_dragt_adiabaticity.png"), dpi=300, bbox_inches='tight')
    plt.close(fig)


# =============================================================
# ============== P_phi Relative Error =========================
# =============================================================
def pphi_error(run_folder, t_pphi_gyro, rel_error_log, P_phi_initial, max_err, ylabel_str, stem=""):
    """Log-log plot of canonical angular momentum conservation error."""

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t_pphi_gyro[1:], rel_error_log[1:], color='crimson', linewidth=1.5)

    ax.set_xscale('log')
    ax.set_yscale('log')

    ax.text(0.02, 0.85, f"Initial $P_\\phi$: {P_phi_initial:.6f}\nMax Relative Error: {max_err:.2e}",
            transform=ax.transAxes, fontsize=11, color='black',
            bbox=dict(facecolor='white', alpha=0.9, edgecolor='black'))

    ax.set_xlabel(r"$\tau/T$")
    ax.set_ylabel(ylabel_str)
    ax.set_title("Relative Error of Canonical Angular Momentum")
    ax.grid(True, which="both", ls="--", alpha=0.5)

    fig.tight_layout()
    fig.savefig(os.path.join(run_folder, f"{stem}_P_phi_rel_error.png"), dpi=300)
    plt.close(fig)


# =====================================================
# ============== Magnetic Moment Deviations ===========
# =====================================================
def mu_deviation(
    summary, run_folder, stem, particle_type, ps_order_label,
    USE_PLOT_TITLES,
    ps_data=None, rk4_data=None, rk45_data=None, rkg_data=None,
):
    """
    Magnetic moment relative deviation (semilogy) for all enabled solvers.

    Each *_data argument is a tuple (t_array, mudrift_array) or None.
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    # ---- solver lines ----
    lnps = lnrk4 = lnrkg = lnrk45 = None
    if rk45_data is not None:
        t_rk45, mudrift_rk45 = rk45_data
        lnrk45, = ax.semilogy(t_rk45, mudrift_rk45, label="RK45", color="#E69F00", linestyle="--")
    if rk4_data is not None:
        t_rk4, mudrift_rk4 = rk4_data
        lnrk4, = ax.semilogy(t_rk4, mudrift_rk4, label="RK4", alpha=0.3, color="#CC79A7", linestyle="-.")
    if rkg_data is not None:
        t_rkg, mudrift_rkg = rkg_data
        lnrkg, = ax.semilogy(t_rkg, mudrift_rkg, label="RKG", alpha=0.3, color="#CC0000", linestyle="-.")
    if ps_data is not None:
        t_ps, mudrift_ps = ps_data
        lnps, = ax.semilogy(t_ps, mudrift_ps, label=f"PS{ps_order_label}", linewidth=0.3, color="#009E73", linestyle="-")

    # ---- axis formatting ----
    ax.margins(x=0.01)
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=100))
    ax.yaxis.set_major_formatter(LogFormatterSciNotation(base=10.0))
    ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=[]))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.grid(True, which="major", linestyle="--", linewidth=0.7)
    ax.get_xaxis().get_major_formatter().set_useOffset(False)

    # # for top slices of mu
    # ax.set_ylim(5e-3, 2e-1)
    # ax.set_yscale('linear')

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

    place_endpoint_labels(fig, ax, endpoints)

    # === Save and Close ===
    fig_path_mu = build_filename(summary, run_folder, stem, figure_tag="mu", ext="png")
    plt.savefig(fig_path_mu, dpi=600, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------
#  prepare_slice_window  — extract time-windowed trajectory slices
# ------------------------------------------------------------------
def prepare_slice_window(
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
    """Compute time-windowed trajectory slices for each enabled solver.

    Returns a dict with keys like ``ps_x_slice``, ``rk4_y_slice``, etc.
    Missing solvers get ``None`` values.
    """
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
        rk4_x, rk4_y, rk4_z = slice_solution(
            t_rk4, solution_rk4, window_duration, norm_time, mode=slice_mode)[:3]
        result["rk4_x_slice"] = rk4_x
        result["rk4_y_slice"] = rk4_y
        result["rk4_z_slice"] = rk4_z

    # ---------- RKG ----------
    if USE_RKG:
        t_rkg = rkg_step * np.arange(solution_rkg.shape[0], dtype=npfloat)
        rkg_x, rkg_y, rkg_z = slice_solution(
            t_rkg, solution_rkg.T, window_duration, norm_time, mode=slice_mode)[:3]
        result["rkg_x_slice"] = rkg_x
        result["rkg_y_slice"] = rkg_y
        result["rkg_z_slice"] = rkg_z

    # ---------- RK45 ----------
    if USE_RK45:
        t_rk45 = ps_step * np.arange(y_rk45_common.shape[1], dtype=npfloat)
        rk45_x, rk45_y, rk45_z = slice_solution(
            t_rk45, y_rk45_common, window_duration, norm_time, mode=slice_mode)[:3]
        result["rk45_x_slice"] = rk45_x
        result["rk45_y_slice"] = rk45_y
        result["rk45_z_slice"] = rk45_z

    return result
