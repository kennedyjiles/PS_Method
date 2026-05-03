"""
Shared plotting functions for constb and hyperb drivers.

Each function is self-contained: creates a figure, plots, saves, and closes.
Drivers call these with pre-computed data — no globals are referenced.

Functions
---------
High-level plots:
    full_2d          — full-run 2D trajectory
    full_3d          — full-run 3D trajectory
    ke_error         — single KE error over time (RK4, RK45, PS max order)
    slice_2d         — sliced 2D trajectory (last/first N gyroperiods)
    slice_3d         — sliced 3D trajectory
    ke_error_multi   — multi-PS-order KE error comparison
    trajectory_error — position error vs analytical (constb only)
"""

import numpy as np
import matplotlib.pyplot as plt
from ps_method import utils as ul


# =====================================================================
# Color palette (colorblind-friendly, consistent across all field types)
# =====================================================================
COLORS = {
    "rk45":       "#E69F00",   # orange
    "rk4":        "#CC79A7",   # reddish purple
    "ps":         "#009E73",   # bluish green
    "analytical": "black",
    "ps4":        "crimson",
    "ps5":        "#0072B2",   # blue
    "ps6":        "#56B4E9",   # sky blue
    "ps7":        "#D55E00",   # vermillion
    "ps10":       "#000000",   # black
    "ps15":       "#999999",   # gray
    "ext":        "black",
    "extb":       "#6A3D9A",   # purple
}

LINESTYLES = {
    "rk45":       "--",
    "rk4":        "-.",
    "ps":         ":",
    "analytical": "-",
}

# =====================================================================
# ================ Full 2D Trajectory Plot ============================
# =====================================================================
def full_2d(
    save_path, *,
    solution_ps, orders_used,
    solution_rk45=None, solution_rk4=None, solution_analytical=None,
    use_rk45=False, use_rk4=False, use_analytical=False,
    particle_type="", field_label="", use_plot_titles=True,
):
    """Full-run 2D trajectory (x vs y)."""
    fig, ax = plt.subplots(figsize=(10, 8))

    if use_analytical and solution_analytical is not None:
        ax.plot(solution_analytical[0], solution_analytical[1],
                color=COLORS["analytical"], linestyle=LINESTYLES["analytical"], linewidth=0.3, label="Exact")
    if use_rk45 and solution_rk45 is not None:
        ax.plot(solution_rk45.y[0], solution_rk45.y[1],
                color=COLORS["rk45"], linestyle=LINESTYLES["rk45"], label="RK45")
    if use_rk4 and solution_rk4 is not None:
        ax.plot(solution_rk4[0], solution_rk4[1],
                color=COLORS["rk4"], linestyle=LINESTYLES["rk4"], linewidth=0.75, label="RK4")
    ax.plot(solution_ps[0], solution_ps[1],
            color=COLORS["ps"], linestyle=LINESTYLES["ps"], label=f"PS{orders_used.max()}")

    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    if use_plot_titles:
        ax.set_title(f"2D {particle_type} Trajectory in {field_label}")
    ax.legend(loc="upper right")
    ax.axis("equal")
    ax.grid(True)
    plt.tight_layout()

    fig.canvas.draw()
    fig.savefig(save_path, dpi=600, bbox_inches="tight")
    plt.close(fig)


# =====================================================================
# ================ Full 3D Trajectory Plot ============================
# =====================================================================
def full_3d(
    save_path, *,
    solution_ps, orders_used,
    solution_rk45=None, solution_rk4=None, solution_analytical=None,
    use_rk45=False, use_rk4=False, use_analytical=False,
    particle_type="", field_label="", use_plot_titles=True,
):
    """Full-run 3D trajectory."""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    if use_analytical and solution_analytical is not None:
        ax.plot(solution_analytical[0], solution_analytical[1], solution_analytical[2],
                color=COLORS["analytical"], linestyle=LINESTYLES["analytical"], linewidth=0.3, label="Exact")
    if use_rk45 and solution_rk45 is not None:
        ax.plot(solution_rk45.y[0], solution_rk45.y[1], solution_rk45.y[2],
                label="RK45", color=COLORS["rk45"], linestyle=LINESTYLES["rk45"])
    if use_rk4 and solution_rk4 is not None:
        ax.plot(solution_rk4[0], solution_rk4[1], solution_rk4[2],
                label="RK4", color=COLORS["rk4"], linestyle=LINESTYLES["rk4"])
    ax.plot(solution_ps[0], solution_ps[1], solution_ps[2],
            label=f"PS{orders_used.max()}", color=COLORS["ps"], linestyle=LINESTYLES["ps"])

    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    ax.set_zlabel(r"$z$")
    if use_plot_titles:
        ax.set_title(f"3D {particle_type} Trajectory in {field_label}")
    ax.legend(loc="upper right")
    plt.tight_layout()

    fig.canvas.draw()
    fig.savefig(save_path, dpi=600, bbox_inches="tight")
    plt.close(fig)


# =====================================================================
# ================ KE Relative Error Plot =============================
# =====================================================================
def ke_error(
    save_path, *,
    t_eval_ps, rel_drift_ps, orders_used,
    t_eval_rk4=None, rel_drift_rk4=None,
    t_eval_rk45=None, rel_drift_rk45=None,
    use_rk4=False, use_rk45=False,
    particle_type="", field_label="", use_plot_titles=True,
    time_factor=None,
    **_ignored,
):
    """Relative kinetic energy error over time (log-log)."""
    if time_factor is None:
        time_factor = 1.0 / (2.0 * np.pi)   

    fig, ax = plt.subplots(figsize=(10, 5))

    lines = {}
    if use_rk45 and rel_drift_rk45 is not None:
        lines["rk45"], = ax.semilogy(
            ul.f64(t_eval_rk45) * time_factor, np.abs(ul.f64(rel_drift_rk45)),
            color=COLORS["rk45"], linestyle=LINESTYLES["rk45"])
    if use_rk4 and rel_drift_rk4 is not None:
        lines["rk4"], = ax.semilogy(
            ul.f64(t_eval_rk4) * time_factor, np.abs(ul.f64(rel_drift_rk4)),
            color=COLORS["rk4"], linestyle=LINESTYLES["rk4"])
    lines["ps"], = ax.semilogy(
        ul.f64(t_eval_ps) * time_factor, np.abs(ul.f64(rel_drift_ps)),
        color=COLORS["ps"], linestyle=LINESTYLES["ps"])

    ul.setup_log_axes(ax)
    ax.set_xlabel(r"$\tau/T$")
    ax.set_ylabel(r"$|\Delta E|/E_0$")
    if use_plot_titles:
        ax.set_title(f"{particle_type} Relative Kinetic Energy Error in {field_label}")

    fig.subplots_adjust(right=0.9)
    fig.canvas.draw()

    endpoints = []
    if use_rk45 and "rk45" in lines:
        endpoints.append((t_eval_rk45[-1], np.abs(rel_drift_rk45[-1]), "RK45", lines["rk45"].get_color()))
    if use_rk4 and "rk4" in lines:
        endpoints.append((t_eval_rk4[-1], np.abs(rel_drift_rk4[-1]), "RK4", lines["rk4"].get_color()))
    endpoints.append((t_eval_ps[-1], np.abs(rel_drift_ps[-1]),
                       f"PS{orders_used.max()}", lines["ps"].get_color()))

    ul.place_endpoint_labels(fig, ax, endpoints)

    fig.canvas.draw()
    fig.savefig(save_path, dpi=600, bbox_inches="tight")
    plt.close(fig)


# =====================================================================
# ================ 2D Trajectory Slice ================================
# =====================================================================
def slice_2d(
    save_path, *,
    ps_x, ps_y, orders_used,
    rk45_x=None, rk45_y=None,
    rk4_x=None, rk4_y=None,
    ana_x=None, ana_y=None,
    use_rk45=False, use_rk4=False, use_analytical=False,
    skip_rk4_slice=False,
    slice_ylim=None, slice_ylim_top=None, slice_equal_aspect=False,
    particle_type="", field_label="", use_plot_titles=True,
):
    """Sliced 2D trajectory (last/first N gyroperiods)."""
    fig, ax = plt.subplots(figsize=(10, 5))

    if use_analytical and ana_x is not None:
        ax.plot(ana_x, ana_y, label="Exact", color=COLORS["analytical"],
                linestyle=LINESTYLES["analytical"], linewidth=0.4)
    if use_rk45 and rk45_x is not None:
        ax.plot(rk45_x, rk45_y, label="RK45", color=COLORS["rk45"], linestyle=LINESTYLES["rk45"])
    if use_rk4 and rk4_x is not None and not skip_rk4_slice:
        ax.plot(rk4_x, rk4_y, label="RK4", color=COLORS["rk4"], linestyle=LINESTYLES["rk4"])
    ax.plot(ps_x, ps_y, label=f"PS{orders_used.max()}", color=COLORS["ps"], linestyle=LINESTYLES["ps"])

    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    if use_plot_titles:
        ax.set_title(f"2D Trajectory of Final {particle_type} Orbits in {field_label}")

    ax.ticklabel_format(style="plain", useOffset=False, axis="both")
    ax.axis("equal")
    if slice_ylim is not None:
        ax.set_ylim(slice_ylim[0], slice_ylim[1])
    if slice_ylim_top is not None:
        ax.set_ylim(top=slice_ylim_top)
    if slice_equal_aspect:
        ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="upper right")
    ax.grid(True)

    fig.canvas.draw()
    fig.savefig(save_path, dpi=600, bbox_inches="tight")
    plt.close(fig)

# =====================================================================
# ================ 3D Trajectory Slice ================================
# =====================================================================
def slice_3d(
    save_path, *,
    ps_x, ps_y, ps_z, orders_used,
    rk45_x=None, rk45_y=None, rk45_z=None,
    rk4_x=None, rk4_y=None, rk4_z=None,
    ana_x=None, ana_y=None, ana_z=None,
    use_rk45=False, use_rk4=False, use_analytical=False,
    particle_type="", field_label="", use_plot_titles=True,
):
    """Sliced 3D trajectory."""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    if use_analytical and ana_x is not None:
        ax.plot(ana_x, ana_y, ana_z, label="Exact",
                color=COLORS["analytical"], linestyle=LINESTYLES["analytical"], linewidth=0.4)
    if use_rk45 and rk45_x is not None:
        ax.plot(rk45_x, rk45_y, rk45_z, label="RK45",
                color=COLORS["rk45"], linestyle=LINESTYLES["rk45"])
    if use_rk4 and rk4_x is not None:
        ax.plot(rk4_x, rk4_y, rk4_z, label="RK4",
                color=COLORS["rk4"], linestyle=LINESTYLES["rk4"])
    ax.plot(ps_x, ps_y, ps_z, label=f"PS{orders_used.max()}",
            color=COLORS["ps"], linestyle=LINESTYLES["ps"])

    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    ax.set_zlabel(r"$z$")
    if use_plot_titles:
        ax.set_title(f"3D Trajectory of Final {particle_type} Orbits in {field_label}")
    ax.legend(loc="upper right")

    fig.canvas.draw()
    fig.savefig(save_path, dpi=600, bbox_inches="tight")
    plt.close(fig)

# =====================================================================
# ================ Multi-Order KE Error Comparison ====================
# =====================================================================
def ke_error_multi(
    save_path, *,
    t_eval_ps, orders_used,
    ps_drifts,
    t_eval_rk4=None, rel_drift_rk4=None,
    t_eval_rk45=None, rel_drift_rk45=None,
    use_rk4=False, use_rk45=False,
    ext_data=None, extb_data=None,
    particle_type="", field_label="", use_plot_titles=True,
    time_factor=None, energy_xlim_left=None,
):
    """
    Multi-PS-order KE error comparison plot.

    Parameters
    ----------
    ps_drifts : list of (order_int, rel_drift_array, color, linestyle)
        Each entry is a PS order to plot.
    ext_data : tuple (t, rel_drift, ps_order_label) or None
        External h5 comparison data.
    extb_data : tuple (t, rel_drift, ps_order_label) or None
        Second external h5 comparison data.
    """
    if time_factor is None:
        time_factor = 1.0 / (2.0 * np.pi)   # default T_gyro for constb/hyperb

    fig, ax = plt.subplots(figsize=(10, 5))

    lines = {}
    # RK methods
    if use_rk45 and rel_drift_rk45 is not None:
        lines["rk45"], = ax.semilogy(
            ul.f64(t_eval_rk45[1:]) * time_factor,
            np.abs(ul.f64(rel_drift_rk45[1:])),
            linestyle=LINESTYLES["rk45"], color=COLORS["rk45"])
    if use_rk4 and rel_drift_rk4 is not None:
        lines["rk4"], = ax.semilogy(
            ul.f64(t_eval_rk4[1:]) * time_factor,
            np.abs(ul.f64(rel_drift_rk4[1:])),
            linestyle=LINESTYLES["rk4"], color=COLORS["rk4"])

    # PS orders
    ps_lines = {}
    for order, drift, color, ls in ps_drifts:
        key = f"ps{order}"
        ps_lines[key], = ax.semilogy(
            ul.f64(t_eval_ps[1:]) * time_factor,
            np.abs(ul.f64(drift[1:])),
            linestyle=ls, color=color)

    # Main PS (max order) — last entry in ps_drifts is assumed to be the main one
    # (already plotted above)

    # External h5 overlays
    if ext_data is not None:
        t_ext, drift_ext, ps_order_ext = ext_data
        lines["ext"], = ax.semilogy(
            ul.f64(t_ext[1:]) * time_factor,
            np.abs(ul.f64(drift_ext[1:])),
            linestyle="-.", linewidth=1.2, color=COLORS["ext"])
    if extb_data is not None:
        t_extb, drift_extb, ps_order_extb = extb_data
        lines["extb"], = ax.semilogy(
            ul.f64(t_extb[1:]) * time_factor,
            np.abs(ul.f64(drift_extb[1:])),
            linestyle="-", linewidth=1.2, color=COLORS["extb"])

    ul.setup_log_axes(ax)
    if energy_xlim_left is not None:
        ax.set_xlim(left=energy_xlim_left)
    ax.set_xlabel(r"$\tau/T$")
    ax.set_ylabel(r"$|\Delta E|/E_0$")
    if use_plot_titles:
        ax.set_title(f"{particle_type} Relative Kinetic Energy Error in {field_label}")

    # --- Endpoint labels ---
    fig.subplots_adjust(right=0.9)
    fig.canvas.draw()

    endpoints = []
    if use_rk45 and "rk45" in lines:
        endpoints.append((t_eval_rk45[-1], np.abs(rel_drift_rk45[-1]),
                          "RK45", lines["rk45"].get_color()))
    if use_rk4 and "rk4" in lines:
        endpoints.append((t_eval_rk4[-1], np.abs(rel_drift_rk4[-1]),
                          "RK4", lines["rk4"].get_color()))

    for order, drift, color, ls in ps_drifts:
        key = f"ps{order}"
        endpoints.append((t_eval_ps[-1], np.abs(drift[-1]),
                          f"PS{order}", ps_lines[key].get_color()))

    if ext_data is not None:
        endpoints.append((t_ext[-1], np.abs(drift_ext[-1]),
                          f"PS{ps_order_ext}*", lines["ext"].get_color()))
    if extb_data is not None:
        endpoints.append((t_extb[-1], np.abs(drift_extb[-1]),
                          f"PS{ps_order_extb}*", lines["extb"].get_color()))

    ul.place_endpoint_labels(fig, ax, endpoints)

    fig.savefig(save_path, dpi=600, bbox_inches="tight")
    plt.close(fig)


# =====================================================================
# ================ Trajectory Error vs Analytical (constb only) =======
# =====================================================================
def trajectory_error(
    save_path, *,
    t_eval_ps, rel_err_ps, orders_used,
    t_eval_rk4=None, rel_err_rk4=None,
    t_eval_rk45=None, rel_err_rk45=None,
    t_ext=None, rel_err_ext=None, ps_order_ext=None,
    use_rk4=False, use_rk45=False, use_external_h5=False,
    use_full_plot=False,
    particle_type="", field_label="", use_plot_titles=True,
    time_factor=None,
):
    """
    Position error (gyro-radius normalized) vs analytical solution.

    Only applicable for constb where an exact solution exists.
    """
    if time_factor is None:
        time_factor = 1.0 / (2.0 * np.pi)  

    fig, ax = plt.subplots(figsize=(10, 5))

    lines = {}
    if use_rk45 and rel_err_rk45 is not None:
        lines["rk45"], = ax.semilogy(
            ul.f64(t_eval_rk45) * time_factor, np.abs(ul.f64(rel_err_rk45)),
            label="RK45", linestyle=LINESTYLES["rk45"], color=COLORS["rk45"])
    if use_rk4 and rel_err_rk4 is not None:
        lines["rk4"], = ax.semilogy(
            ul.f64(t_eval_rk4) * time_factor, np.abs(ul.f64(rel_err_rk4)),
            label="RK4", linestyle=LINESTYLES["rk4"], color=COLORS["rk4"])
    lines["ps"], = ax.semilogy(
        ul.f64(t_eval_ps) * time_factor, np.abs(ul.f64(rel_err_ps)),
        label=f"PS{orders_used.max()}", linestyle=LINESTYLES["ps"], color=COLORS["ps"])
    if use_external_h5 and rel_err_ext is not None:
        lines["ext"], = ax.semilogy(
            ul.f64(t_ext) * time_factor, np.abs(ul.f64(rel_err_ext)),
            label=f"PS{ps_order_ext}*", linestyle="-.", color=COLORS["ext"])

    ul.setup_log_axes(ax)
    if not use_full_plot:
        ax.tick_params(axis="x", which="both", labelbottom=False)
    else:
        ax.set_xlabel(r"$\tau/T$")
        ax.tick_params(axis="x", which="both", labelbottom=True)

    ax.set_ylabel(r"$\Delta \mathbf{r}_\perp/\rho_L$")
    if use_plot_titles:
        ax.set_title(f"{particle_type} Gyro-Radius Error in {field_label}")

    fig.subplots_adjust(right=0.9)
    fig.canvas.draw()

    endpoints = []
    if use_rk45 and "rk45" in lines:
        endpoints.append((t_eval_rk45[-1], np.abs(rel_err_rk45[-1]),
                          "RK45", lines["rk45"].get_color()))
    if use_rk4 and "rk4" in lines:
        endpoints.append((t_eval_rk4[-1], np.abs(rel_err_rk4[-1]),
                          "RK4", lines["rk4"].get_color()))
    if use_external_h5 and "ext" in lines:
        endpoints.append((t_ext[-1], np.abs(rel_err_ext[-1]),
                          f"PS{ps_order_ext}*", lines["ext"].get_color()))
    endpoints.append((t_eval_ps[-1], np.abs(rel_err_ps[-1]),
                      f"PS{orders_used.max()}", lines["ps"].get_color()))

    ul.place_endpoint_labels(fig, ax, endpoints)

    fig.canvas.draw()
    fig.savefig(save_path, dpi=600, bbox_inches="tight")
    plt.close(fig)
