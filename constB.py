"""
constb.py — Driver for charged particle trajectory simulation in a
            uniform magnetic field using power series, RK4, RK45, and
            analytical solutions.

Usage:
    python constb.py                          # default config (demo)
    python constb.py demo                     # named config → configs/constb/demo.yml
    python constb.py paper                    # named config → configs/constb/paper.yml
    python constb.py configs/constb/my.yml    # direct path to a custom YAML config
"""

import numpy as np
import builtins
import os
import time
import sys
from types import SimpleNamespace
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib as mpl

from configs.config_loader import (
    load_config, compute_derived_constb, copy_config_to_output,
    apply_manual_h5_overrides, physics_hash,
)
from ps_method.constants import q_e, evtoj


def main(cfg_path, replot=False):
    """Run a constant-B simulation from a YAML config file path.

    Parameters
    ----------
    cfg_path : str – path to a YAML config file.
    replot   : bool – if True, force READ_DATA=True (skip solvers,
               regenerate plots from cached h5 data).
    """

    print(f"Loading YAML config: {cfg_path}\n")
    cfg = load_config(cfg_path)

    # --- Manual h5 mode: peek at the cached file, override identity in cfg
    #     so all downstream derived quantities use the h5's values rather
    #     than whatever the yml had. Path is tried yml-relative first, then
    #     cwd-relative (so e.g. companion ymls next to a trimmed h5 work). ---
    _raw_manual = cfg.get("manual_h5_path")
    if _raw_manual and not os.path.isabs(_raw_manual):
        _yml_relative = os.path.join(os.path.dirname(os.path.abspath(cfg_path)), _raw_manual)
        if os.path.exists(_yml_relative):
            cfg["manual_h5_path"] = _yml_relative
    manual_h5_path = cfg.get("manual_h5_path")
    USE_MANUAL_FILE = manual_h5_path is not None and os.path.exists(manual_h5_path)
    if USE_MANUAL_FILE:
        print(f"Manual h5 mode: loading identity from {manual_h5_path}")
        apply_manual_h5_overrides(cfg, manual_h5_path, field="constb")

    # --- Resolve float type BEFORE compute_derived (needs to be set for builtins) ---
    USE_FLOAT128 = cfg.get("use_float128", False)
    if USE_FLOAT128:
        npfloat = np.float128
    else:
        npfloat = np.float64
    builtins.npfloat = npfloat

    # --- Import physics modules AFTER builtins.npfloat is set so @maybe_njit
    #     sees the correct float type (float128 skips njit, float64 compiles). ---
    from ps_method import constb_physics as cp
    from ps_method import utils as ul
    from ps_method import constb_hyperb_energy_analysis as ea
    from ps_method import constb_hyperb_plots as fplt
    from ps_method import writers as wr

    p = compute_derived_constb(cfg, npfloat=npfloat)

    # === Unpack Config ===
    READ_DATA       = p["READ_DATA"]
    WRITE_DATA      = p["WRITE_DATA"]
    if replot:
        READ_DATA = True
    USE_RK45        = p["USE_RK45"]
    USE_RK4         = p["USE_RK4"]
    USE_ANALYTICAL  = p["USE_ANALYTICAL"]
    USE_PLOT_TITLES = p["USE_PLOT_TITLES"]
    USE_FULL_PLOT   = p["USE_FULL_PLOT"]
    gyro_plot_slice = p["gyro_plot_slice"]

    USE_EXTERNAL_H5  = p["USE_EXTERNAL_H5"]
    USE_EXTERNAL_H5b = p["USE_EXTERNAL_H5b"]
    external_h5      = p["external_h5"]
    external_h5b     = p["external_h5b"]
    PS_order_ext     = p["PS_order_ext"]
    PS_order_extb    = p["PS_order_extb"]

    output_folder = p["output_folder"]
    run_storage   = p["run_storage"]

    pitch_deg   = p["pitch_deg"]
    phi_deg     = p["phi_deg"]
    KE_particle = p["KE_particle"]
    mass        = p["mass"]
    Bfield_si   = p["Bfield_si"]
    x_initial   = p["x_initial"]
    y_initial   = p["y_initial"]
    z_initial   = p["z_initial"]

    gyroperiods = p["gyroperiods"]
    norm_time   = p["norm_time"]
    T_gyro      = p["T_gyro"]
    ps_step     = p["ps_step"]
    rk4_step    = p["rk4_step"]

    PS_order    = p["PS_order"]
    tol         = p["tol"]
    rtol_rk45   = p["rtol_rk45"]
    atol_rk45   = p["atol_rk45"]

    # === Misc Odds and Ends ===
    os.makedirs(output_folder, exist_ok=True)
    ul.plt_config(scale=1)
    plt.ioff()

    if USE_FLOAT128:
        mpl.rcParams['agg.path.chunksize'] = 100000
    else:
        mpl.rcParams['agg.path.chunksize'] = 1000

    particle_type = p["particle"].capitalize()

    charge_sign = npfloat(-1) if p["particle"].lower() in ("electron", "e") else npfloat(1)

    # === Misc Normalizing  ===
    B_0 = np.linalg.norm(Bfield_si)  # Magnitude of the field
    Bfield = Bfield_si / B_0         # normalized B field
    v_si = npfloat(np.sqrt(npfloat(2 * KE_particle * evtoj / mass)))
    tau_time = mass / (abs(q_e) * B_0)
    v_tau = v_si * tau_time
    # physical_time computed below, after norm_time is rounded to an exact multiple of ps_step

    # === Velocity Config ===
    pitch_rad = np.radians(pitch_deg)
    phi_rad = np.radians(phi_deg)
    v_par = v_tau * np.cos(pitch_rad)
    v_perp = v_tau * np.sin(pitch_rad)
    vx_initial = v_perp * np.cos(phi_rad)
    vy_initial = v_perp * np.sin(phi_rad)
    vz_initial = v_par
    if abs(vx_initial) < tol: vx_initial = npfloat(0.0)
    if abs(vy_initial) < tol: vy_initial = npfloat(0.0)
    if abs(vz_initial) < tol: vz_initial = npfloat(0.0)

    gyro_radius_si = v_si * np.sin(pitch_rad) * mass / (abs(q_e) * B_0)

    initial_pos_vel = np.array([x_initial, y_initial, z_initial, vx_initial, vy_initial, vz_initial], dtype=npfloat)

    # === Ensures that Total Time Elapsed is the Same ===
    # Note: norm_time is rounded so steps are an exact integer multiple of the
    # PS step size.  This prevents phase errors when comparing trajectories.
    # If PS and RK use different step sizes you may need interpolation.
    # This has no impact on the energy calculations.
    steps_ps = int(round(norm_time / ps_step))
    norm_time = steps_ps * ps_step           # <-- adjust total time to be exact multiple
    physical_time = norm_time * tau_time     # compute after rounding so values stay consistent
    t_eval_ps = np.linspace(0, norm_time, steps_ps + 1, dtype=npfloat)

    if USE_RK4:
        steps_rk4 = int(round(norm_time / rk4_step))
        t_eval_rk4 = np.linspace(0, norm_time, steps_rk4 + 1, dtype=npfloat)

    if USE_RK45:
        steps_rk45 = steps_ps   # just for plotting consistency
        t_eval_rk45 = np.float64(t_eval_ps)

    if USE_RK4 and not np.isclose(ps_step, rk4_step, rtol=1e-12):
        print(
            "⚠️  Warning: PS and RK4 step sizes or total times do not align exactly.\n"
            "    → Energy drift comparisons are fine.\n"
            "    → Trajectory and phase comparisons may show artificial offsets."
        )

    if USE_ANALYTICAL:
        start_time_analytical = time.time()
        solution_analytical = cp.analytical(
            t_eval_ps,
            initial_pos_vel, charge_sign)
        end_time_analytical = time.time()

    # === Build parameter signature & check cache ===
    params = wr.get_run_params_constb(USE_RK45, USE_RK4, KE_particle, rtol_rk45, atol_rk45,
                       mass, q_e, B_0,
                       x_initial, y_initial, z_initial,
                       pitch_deg, phi_deg,
                       norm_time, ps_step, rk4_step,
                       PS_order, tol, charge_sign,
                       dtype=npfloat.__name__)
    if USE_MANUAL_FILE:
        cache_path = manual_h5_path
    else:
        cache_path = wr.h5_path_for(physics_hash(cfg), run_storage)
    # In manual mode nothing new is written to run_storage — skip creating
    # an empty stub folder.
    if not USE_MANUAL_FILE:
        os.makedirs(run_storage, exist_ok=True)

    if os.path.exists(cache_path) and READ_DATA:
        print(f"Found existing results: {os.path.basename(cache_path)} — loading.\n")
        cached = wr.load_results_h5_constb(cache_path)

        # Rehydrate what you need for plotting/analysis:
        solution_ps = cached["ps"]["y"] if cached["ps"] else None
        orders_used = cached["ps"]["orders"] if cached["ps"] else None
        t_eval_ps = cached["ps"]["t"] if cached["ps"] else None

        if USE_RK4 and cached["rk4"]:
            solution_rk4 = cached["rk4"]["y"]
            t_eval_rk4 = cached["rk4"]["t"]
        if USE_RK45 and cached["rk45"]:
            solution_rk45 = SimpleNamespace(t=cached["rk45"]["t"], y=cached["rk45"]["y"])
            t_eval_rk45 = cached["rk45"]["t"]

        timing = cached.get("meta", {}).get("timing", {})
        stem = wr.stem_from_h5(cache_path)


    else:
        print("No matching file or 'Read Data' skipped. Running solvers...\n")
        # ====== Run RK45 ======
        if USE_RK45:
            start_time_rk45 = time.time()
            solution_rk45 = solve_ivp(
                cp.lorentz_force, (0, norm_time),
                initial_pos_vel, method='RK45',
                t_eval=t_eval_rk45, args=(Bfield, charge_sign),
                rtol=rtol_rk45,
                atol=atol_rk45)
            end_time_rk45 = time.time()

        # ====== Run RK4 ======
        if USE_RK4:
            start_time_rk4 = time.time()
            rk4_dt = npfloat(t_eval_rk4[1] - t_eval_rk4[0])
            solution_rk4 = ul.rk4_fixed_step(
                cp.lorentz_force, initial_pos_vel,
                rk4_dt, steps_rk4, args=(Bfield, charge_sign))
            end_time_rk4 = time.time()

        # ===== Run PS Method ====
        start_time_ps = time.time()
        solution_ps, orders_used = cp.ps_integrate(
            PS_order, steps_ps, initial_pos_vel,
            ps_step, Bfield, charge_sign, tol)
        end_time_ps = time.time()

        # Prepare a results dict for saving
        results = {
            "ps": {
                "t": t_eval_ps,
                "y": solution_ps,
                "orders": orders_used,
            },
            "rk4": None,
            "rk45": None,
            "meta": {
                "timing": {},
                "physical_time": float(physical_time),
                "norm_time": float(norm_time),
                "particle_label": f"{KE_particle:.1e} eV {particle_type.lower()}",
            }
        }

        if USE_RK4:
            results["rk4"] = {"t": t_eval_rk4, "y": solution_rk4}
            results["meta"]["timing"]["rk4"] = end_time_rk4 - start_time_rk4
        if USE_RK45:
            results["rk45"] = {"t": solution_rk45.t, "y": solution_rk45.y}
            results["meta"]["timing"]["rk45"] = end_time_rk45 - start_time_rk45

        results["meta"]["timing"]["ps"] = end_time_ps - start_time_ps

        timing = results["meta"]["timing"]

        # Save to cache
        if WRITE_DATA:
            wr.save_results_h5_constb(cache_path, params, results)
            print(f"Saved results → {os.path.basename(cache_path)}")
        stem = wr.stem_from_h5(cache_path)


    # === Sanity Check ===
    print(f"Particle        : {KE_particle:.1e} eV {particle_type}")
    if USE_RK45 and "rk45" in timing:
        print(f"Run Time RK45   : {timing['rk45']:.2f} s")
    if USE_RK4 and "rk4" in timing:
        print(f"Run Time RK4    : {timing['rk4']:.2f} s")
    if "ps" in timing:
        print(f"Run Time PS     : {timing['ps']:.2f} s")
    print(f"Norm Time       : {norm_time:.2e}")
    print(f"Physical Time   : {physical_time:.2e} s")
    if orders_used is not None:
        print(f"PS Orders       : max={orders_used.max()}, mean={orders_used.mean():.1f}\n")

    # === Create run-specific output subfolders ===
    # data/constb/<config>/<stem>/figures/   ← plots
    # data/constb/<config>/<stem>/           ← summary, config copy
    run_folder = os.path.join(output_folder, stem)
    fig_folder = os.path.join(run_folder, "figures")
    os.makedirs(fig_folder, exist_ok=True)

    # --- Copy config YAML to run folder (with git hash) ---
    copy_config_to_output(cfg_path, run_folder, cfg=cfg)

    # --- Filename helper (shared stem for all plots) ---
    _base = f"{fig_folder}/{stem}"
    _field_label = "Constant B Field"
    _plot_kw = dict(particle_type=particle_type, field_label=_field_label, use_plot_titles=USE_PLOT_TITLES)

    # ===================================================================
    # ================Full 2D & 3D Trajectory Plots======================
    # ===================================================================
    if USE_FULL_PLOT:
        _traj_kw = dict(
            solution_ps=solution_ps, orders_used=orders_used,
            solution_rk45=solution_rk45 if USE_RK45 else None,
            solution_rk4=solution_rk4 if USE_RK4 else None,
            solution_analytical=solution_analytical if USE_ANALYTICAL else None,
            use_rk45=USE_RK45, use_rk4=USE_RK4, use_analytical=USE_ANALYTICAL,
            **_plot_kw,
        )
        fplt.full_2d(f"{_base}_2D.png", **_traj_kw)
        fplt.full_3d(f"{_base}_3D.png", **_traj_kw)

    # ================================================================
    # ==================KE Error Plot Over time Only =================
    # ================================================================
    rel_drift_ps = ea.energy_drift(*ea.extract_v(solution_ps))

    rel_drift_rk4 = None
    rel_drift_rk45 = None

    if USE_RK45:
        rel_drift_rk45 = ea.energy_drift(*ea.extract_v(solution_rk45.y))

    if USE_RK4:
        rel_drift_rk4 = ea.energy_drift(*ea.extract_v(solution_rk4))

    # --- External h5 KE drift overlays (used by ke_error AND ke_error_multi) ---
    # Loaded here (before either plot) so the simple ke_error gets the same
    # overlay as the multi-PS plot. yml > h5 saved attr > computed from orders.
    def _resolve_ext_order(yml_value, ext_grp):
        if yml_value is not None:
            return yml_value
        saved = ext_grp.get("max_ps")
        if saved is not None:
            return int(saved)
        if "orders" in ext_grp:
            return int(np.max(ext_grp["orders"]))
        return None

    ext_data = extb_data = None
    if USE_EXTERNAL_H5:
        external = wr.load_results_h5_constb(external_h5)
        ext_ps = external["ps"]
        t_ext, y_ext = ext_ps["t"], ext_ps["y"]
        rel_drift_ext = ea.energy_drift_pure(*ea.extract_v(y_ext))
        ext_data = (t_ext, rel_drift_ext, _resolve_ext_order(PS_order_ext, ext_ps))

    if USE_EXTERNAL_H5b:
        externalb = wr.load_results_h5_constb(external_h5b)
        ext_psb = externalb["ps"]
        t_extb, y_extb = ext_psb["t"], ext_psb["y"]
        rel_drift_extb = ea.energy_drift_pure(*ea.extract_v(y_extb))
        extb_data = (t_extb, rel_drift_extb, _resolve_ext_order(PS_order_extb, ext_psb))

    fplt.ke_error(
        f"{_base}_KEerror.png",
        t_eval_ps=t_eval_ps, rel_drift_ps=rel_drift_ps, orders_used=orders_used,
        t_eval_rk4=t_eval_rk4 if USE_RK4 else None, rel_drift_rk4=rel_drift_rk4,
        t_eval_rk45=t_eval_rk45 if USE_RK45 else None, rel_drift_rk45=rel_drift_rk45,
        use_rk4=USE_RK4, use_rk45=USE_RK45,
        ext_data=ext_data, extb_data=extb_data,
        **_plot_kw,
    )

    # ======================================
    # ============= Slicing ================
    # ======================================
    window_duration = gyro_plot_slice * 2 * np.pi

    if USE_RK4:
        rk4_x, rk4_y, rk4_z, *_ = ul.slice_solution_constb_hyperb(t_eval_rk4, solution_rk4, window_duration, norm_time)

    if USE_RK45:
        rk45_x, rk45_y, rk45_z, *_ = ul.slice_solution_constb_hyperb(t_eval_rk45, solution_rk45.y, window_duration, norm_time)

    if USE_ANALYTICAL:
        ana_x, ana_y, ana_z, *_ = ul.slice_solution_constb_hyperb(t_eval_ps, solution_analytical, window_duration, norm_time)

    ps_x, ps_y, ps_z, *_ = ul.slice_solution_constb_hyperb(t_eval_ps, solution_ps, window_duration, norm_time)

    _slice_kw = dict(
        ps_x=ps_x, ps_y=ps_y, orders_used=orders_used,
        rk45_x=rk45_x if USE_RK45 else None, rk45_y=rk45_y if USE_RK45 else None,
        rk4_x=rk4_x if USE_RK4 else None, rk4_y=rk4_y if USE_RK4 else None,
        ana_x=ana_x if USE_ANALYTICAL else None, ana_y=ana_y if USE_ANALYTICAL else None,
        use_rk45=USE_RK45, use_rk4=USE_RK4, use_analytical=USE_ANALYTICAL,
        **_plot_kw,
    )

    if USE_FULL_PLOT:
        fplt.slice_2d(f"{_base}_2Dslice.png", **_slice_kw)
        fplt.slice_3d(
            f"{_base}_3Dslice.png",
            ps_x=ps_x, ps_y=ps_y, ps_z=ps_z, orders_used=orders_used,
            rk45_x=rk45_x if USE_RK45 else None, rk45_y=rk45_y if USE_RK45 else None, rk45_z=rk45_z if USE_RK45 else None,
            rk4_x=rk4_x if USE_RK4 else None, rk4_y=rk4_y if USE_RK4 else None, rk4_z=rk4_z if USE_RK4 else None,
            ana_x=ana_x if USE_ANALYTICAL else None, ana_y=ana_y if USE_ANALYTICAL else None, ana_z=ana_z if USE_ANALYTICAL else None,
            use_rk45=USE_RK45, use_rk4=USE_RK4, use_analytical=USE_ANALYTICAL,
            **_plot_kw,
        )

    # ===============================================================
    # === Multi-PS-order KE error comparison ========================
    # ===============================================================
    if USE_FULL_PLOT and not USE_FLOAT128:
        # ext_data / extb_data already loaded above (so the simple ke_error
        # plot can use them too). They're reused here unchanged.

        # --- Recompute PS at various orders ---
        # Skip any order that matches the main run's PS order — otherwise
        # the comparison plot would draw two lines for the same order.
        _main_order = int(orders_used.max())
        _ps_orders = [n for n in (4, 5, 6, 7, 10) if n != _main_order]
        ps_drifts = []
        for order in _ps_orders:
            sol, _ = cp.ps_integrate(order, steps_ps, initial_pos_vel, ps_step, Bfield, charge_sign, tol)
            drift = ea.energy_drift(*ea.extract_v(sol))
            ps_drifts.append((order, drift,
                              fplt.COLORS[f"ps{order}"],
                              fplt.LINESTYLES[f"ps{order}"]))

        ps_drifts.append((orders_used.max(), rel_drift_ps,
                          fplt.COLORS["ps"], fplt.LINESTYLES["ps"]))

        fplt.ke_error_multi(
            f"{_base}_KEerror_manyPS.png",
            t_eval_ps=t_eval_ps, orders_used=orders_used,
            ps_drifts=ps_drifts,
            t_eval_rk4=t_eval_rk4 if USE_RK4 else None, rel_drift_rk4=rel_drift_rk4,
            t_eval_rk45=t_eval_rk45 if USE_RK45 else None, rel_drift_rk45=rel_drift_rk45,
            use_rk4=USE_RK4, use_rk45=USE_RK45,
            ext_data=ext_data, extb_data=extb_data,
            **_plot_kw,
        )

    # =================================================================
    # === Trajectory error vs analytical ==============================
    # =================================================================
    if USE_ANALYTICAL:
        # Each method has its own time grid (steps_per_gyro can differ between
        # PS, RK4, and any external h5). Re-evaluate the analytical closed form
        # on each grid so trajectory differences are sampled at matching times.
        x_ana = solution_analytical[0]
        y_ana = solution_analytical[1]
        rel_traj_err_ps = ea.trajectory_error_xy(solution_ps, x_ana, y_ana, gyro_radius_si)

        rel_traj_err_rk4 = None
        if USE_RK4:
            sol_ana_rk4 = cp.analytical(t_eval_rk4, initial_pos_vel, charge_sign)
            rel_traj_err_rk4 = ea.trajectory_error_xy(
                solution_rk4, sol_ana_rk4[0], sol_ana_rk4[1], gyro_radius_si)

        rel_traj_err_rk45 = None
        if USE_RK45:
            # rk45 uses the PS grid (t_eval_rk45 = t_eval_ps), so x_ana works
            rel_traj_err_rk45 = ea.trajectory_error_xy(solution_rk45.y, x_ana, y_ana, gyro_radius_si)

        # External h5 trajectory error
        t_ext_traj = None
        rel_traj_err_ext = None
        ps_order_ext_label = PS_order_ext
        if USE_EXTERNAL_H5:
            external = wr.load_results_h5_constb(external_h5)
            ext_ps = external["ps"]
            t_ext_traj = ext_ps["t"]
            y_ext = ext_ps["y"]
            sol_ana_ext = cp.analytical(t_ext_traj, initial_pos_vel, charge_sign)
            rel_traj_err_ext = ea.trajectory_error_xy(
                y_ext, sol_ana_ext[0], sol_ana_ext[1], gyro_radius_si)
            if ps_order_ext_label is None:
                # h5's saved attr > computed max from orders array
                ps_order_ext_label = ext_ps.get("max_ps")
                if ps_order_ext_label is None and "orders" in ext_ps:
                    ps_order_ext_label = int(np.max(ext_ps["orders"]))

        fplt.trajectory_error(
            f"{_base}_TrajError.png",
            t_eval_ps=t_eval_ps, rel_traj_err_ps=rel_traj_err_ps, orders_used=orders_used,
            t_eval_rk4=t_eval_rk4 if USE_RK4 else None, rel_traj_err_rk4=rel_traj_err_rk4,
            t_eval_rk45=t_eval_rk45 if USE_RK45 else None, rel_traj_err_rk45=rel_traj_err_rk45,
            t_ext=t_ext_traj, rel_traj_err_ext=rel_traj_err_ext, ps_order_ext=ps_order_ext_label,
            use_rk4=USE_RK4, use_rk45=USE_RK45, use_external_h5=USE_EXTERNAL_H5,
            use_full_plot=USE_FULL_PLOT,
            field_label="a Constant Magnetic Field", **{k: v for k, v in _plot_kw.items() if k != "field_label"},
        )


    # ============================================
    # ======= Write Summary Output to File =======
    # ============================================
    output_filename = f"{run_folder}/{stem}_summary.txt"

    wr.summary_txt_constb(
        output_filename,
        stem=stem, WRITE_DATA=WRITE_DATA, READ_DATA=READ_DATA,
        particle_type=particle_type, KE_particle=KE_particle, mass=mass,
        pitch_deg=pitch_deg, phi_deg=phi_deg,
        tau_time=tau_time, v_tau=v_tau, gyro_radius_si=gyro_radius_si,
        x_initial=x_initial, y_initial=y_initial, z_initial=z_initial,
        vx_initial=vx_initial, vy_initial=vy_initial, vz_initial=vz_initial,
        Bfield=Bfield, B_0=B_0,
        npfloat_name=npfloat.__name__,
        norm_time=norm_time, physical_time=physical_time, gyroperiods=gyroperiods,
        ps_step=ps_step, rk4_step=rk4_step,
        steps_ps=steps_ps, steps_rk4=steps_rk4 if USE_RK4 else None,
        orders_used=orders_used,
        USE_RK4=USE_RK4, USE_RK45=USE_RK45, USE_ANALYTICAL=USE_ANALYTICAL,
        timing=timing,
        analytical_time=(end_time_analytical - start_time_analytical) if USE_ANALYTICAL else None,
        rel_drift_ps=rel_drift_ps,
        rel_drift_rk4=rel_drift_rk4,
        rel_drift_rk45=rel_drift_rk45,
    )

    print(f"\nRun Complete → {run_folder}")


# === CLI entry point ===
if __name__ == "__main__":
    run = "demo"
    if len(sys.argv) > 1:
        run = sys.argv[1]
        print(f"Run mode set from command line: {run}\n")
    else:
        print(f"Using default run mode: {run}\n")

    _configs_dir = os.path.join(os.path.dirname(__file__), "configs", "constb")

    if run.endswith((".yml", ".yaml")) and os.path.isfile(run):
        _yaml_path = run
    elif os.path.isfile(os.path.join(_configs_dir, f"{run}.yml")):
        _yaml_path = os.path.join(_configs_dir, f"{run}.yml")
    else:
        raise FileNotFoundError(
            f"No YAML config found for '{run}'. "
            f"Expected configs/constb/{run}.yml or a direct path to a .yml file.\n"
            f"Available configs: {[f.replace('.yml','') for f in os.listdir(_configs_dir) if f.endswith('.yml') and f != 'base.yml']}"
        )

    main(_yaml_path)
