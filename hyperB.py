"""
hyperb.py — Driver for charged particle trajectory simulation in a
            hyperbolic tangent magnetic field using power series, RK4,
            and RK45 solvers.

Usage:
    python hyperb.py                          # default config (demo)
    python hyperb.py demo                     # named config → configs/hyperb/demo.yml
    python hyperb.py paper1                   # named config → configs/hyperb/paper1.yml
    python hyperb.py configs/hyperb/my.yml    # direct path to a custom YAML config
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

from configs.config_loader import load_config, compute_derived_hyperb, copy_config_to_output
from ps_method.constants import q_e, evtoj


def main(cfg_path, replot=False):
    """Run a hyperbolic-B simulation from a YAML config file path.

    Parameters
    ----------
    cfg_path : str – path to a YAML config file.
    replot   : bool – if True, force READ_DATA=True (skip solvers,
               regenerate plots from cached h5 data).
    """
    cfg = load_config(cfg_path)
    _config_log = cfg.pop("_config_log", [])

    # --- Resolve float type BEFORE compute_derived (needs to be set for builtins) ---
    USE_FLOAT128 = cfg.get("use_float128", False)
    if USE_FLOAT128:
        npfloat = np.float128
    else:
        npfloat = np.float64
    builtins.npfloat = npfloat

    # --- Import physics modules AFTER builtins.npfloat is set so @maybe_njit
    #     sees the correct float type (float128 skips njit, float64 compiles). ---
    from ps_method import hyperb_physics as hp
    from ps_method import utils as ul
    from ps_method import constb_hyperb_energy_analysis as ea
    from ps_method import constb_hyperb_plots as fplt
    from ps_method import writers as wr

    p = compute_derived_hyperb(cfg, npfloat=npfloat)

    # === Unpack Config ===
    READ_DATA       = p["READ_DATA"]
    WRITE_DATA      = p["WRITE_DATA"]
    if replot:
        READ_DATA = True
    USE_RK45        = p["USE_RK45"]
    USE_RK4         = p["USE_RK4"]
    USE_PLOT_TITLES    = p["USE_PLOT_TITLES"]
    USE_FULL_PLOT      = p["USE_FULL_PLOT"]
    window_duration    = p["window_duration"]
    slice_mode         = p["slice_mode"]
    skip_rk4_slice     = p["skip_rk4_slice"]
    slice_ylim         = p["slice_ylim"]
    slice_ylim_top     = p["slice_ylim_top"]
    slice_equal_aspect = p["slice_equal_aspect"]
    energy_xlim_left   = p["energy_xlim_left"]

    USE_EXTERNAL_H5  = p["USE_EXTERNAL_H5"]
    USE_EXTERNAL_H5b = p["USE_EXTERNAL_H5b"]
    external_h5      = p["external_h5"]
    external_h5b     = p["external_h5b"]
    PS_order_ext     = p["PS_order_ext"]
    PS_order_extb    = p["PS_order_extb"]

    output_folder = p["output_folder"]
    run_storage   = p["run_storage"]

    pitch_deg    = p["pitch_deg"]
    phi_deg      = p["phi_deg"]
    KE_particle  = p["KE_particle"]
    mass_si      = p["mass_si"]
    delta        = p["delta"]
    B_0          = p["B_0"]
    x_initial_si = p["x_initial_si"]
    y_initial_si = p["y_initial_si"]
    z_initial_si = p["z_initial_si"]

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
    os.makedirs(run_storage, exist_ok=True)
    os.makedirs(output_folder, exist_ok=True)
    ul.plt_config(scale=1)
    plt.ioff()

    if USE_FLOAT128:
        mpl.rcParams['agg.path.chunksize'] = 100000
    else:
        mpl.rcParams['agg.path.chunksize'] = 100

    particle_type = p["particle"].capitalize()
    qoverm = npfloat(-1) if p["particle"].lower() in ("electron", "e") else npfloat(1)

    # === Misc Normalization  ===
    pitch_rad = np.radians(pitch_deg)
    phi_rad = np.radians(phi_deg)

    v_si = npfloat(np.sqrt(npfloat(2 * KE_particle * evtoj / mass_si))) / 1000  # /1000 puts things in km
    tau_time = mass_si / (abs(q_e) * B_0)

    gyro_radius_si = abs(v_si * np.sin(pitch_rad) * mass_si / (q_e * B_0))
    r_normalization = delta
    v_tau = v_si * tau_time / r_normalization
    gamma = 1 / (delta / r_normalization)  # if normalizing by delta this should be 1

    physical_time = norm_time * tau_time

    x_initial = npfloat(x_initial_si / r_normalization)
    y_initial = npfloat(y_initial_si / r_normalization)
    z_initial = npfloat(z_initial_si / r_normalization)

    # === Velocity Component Config for PS ===
    v_par = v_tau * np.cos(pitch_rad)
    v_perp = v_tau * np.sin(pitch_rad)
    vx_initial = v_perp * np.cos(phi_rad)
    vy_initial = v_perp * np.sin(phi_rad)
    vz_initial = v_par
    if abs(vx_initial) < tol: vx_initial = npfloat(0.0)
    if abs(vy_initial) < tol: vy_initial = npfloat(0.0)
    if abs(vz_initial) < tol: vz_initial = npfloat(0.0)


    initial_pos_vel = np.array([x_initial, y_initial, z_initial, vx_initial, vy_initial, vz_initial], dtype=npfloat)


    # === Ensures that Total Time Elapsed is the Same ===
    steps_ps = int(round(norm_time / ps_step))
    norm_time = steps_ps * ps_step           # <-- adjust total time to be exact multiple
    t_eval_ps = np.linspace(0, norm_time, steps_ps + 1, dtype=npfloat)

    if USE_RK4:
        steps_rk4 = int(norm_time / rk4_step)
        t_eval_rk4 = np.linspace(0, norm_time, steps_rk4 + 1, dtype=npfloat)

    if USE_RK45:
        steps_rk45 = steps_rk4          # for plotting points
        t_eval_rk45 = np.float64(t_eval_rk4)      # for plots, it's doing it's own thing mostly

    # === Build parameter signature & check cache ===
    params = wr.get_run_params_hyperb(USE_RK45, USE_RK4, KE_particle, rtol_rk45, atol_rk45,
                       mass_si, q_e, B_0, delta,
                       x_initial, y_initial, z_initial,
                       pitch_deg, phi_deg,
                       norm_time, ps_step, rk4_step,
                       PS_order, tol, qoverm)
    cache_path = wr.h5_path_for(params, run_storage)

    if os.path.exists(cache_path) and READ_DATA:
        print(f"Found existing results: {os.path.basename(cache_path)} — loading.\n")
        cached = wr.load_results_h5_hyperb(cache_path)

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
        stem = os.path.splitext(os.path.basename(cache_path))[0]

    else:
        print("No matching file or 'Read Data' skipped. Running solvers...\n")
        # ====== Run RK45 ======
        if USE_RK45:
            start_time_rk45 = time.time()
            solution_rk45 = solve_ivp(
                hp.lorentz_force, (0, norm_time),
                initial_pos_vel,method='RK45',
                t_eval=t_eval_rk45, args=(gamma,qoverm),
                rtol= rtol_rk45,
                atol= atol_rk45)
            end_time_rk45 = time.time()

        # ====== Run RK4 ======
        if USE_RK4:
            start_time_rk4 = time.time()
            rk4_dt = npfloat(t_eval_rk4[1] - t_eval_rk4[0])
            solution_rk4 = ul.rk4_fixed_step(
                hp.lorentz_force, initial_pos_vel,
                rk4_dt, steps_rk4, args=(gamma,qoverm))
            end_time_rk4 = time.time()

        # ===== Run PS Method ====
        start_time_ps = time.time()
        solution_ps, orders_used = hp.ps_integrate(
            PS_order, steps_ps,
            initial_pos_vel, ps_step, gamma,
            qoverm, tol)
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
            wr.save_results_h5_hyperb(cache_path, params, results)
            print(f"Saved results → {os.path.basename(cache_path)}")
        stem = os.path.splitext(os.path.basename(cache_path))[0]

    # === Sanity Check ===
    print(f"Particle        : {KE_particle:.1e} eV {particle_type}")
    print(f"gyroradius      : {gyro_radius_si:.2f} km")
    print(f"delta           : {delta:.2f} km")
    if USE_RK45 and "rk45" in timing:
        print(f"Run Time RK45   : {timing['rk45']:.2f} s")
    if USE_RK4 and "rk4" in timing:
        print(f"Run Time RK4    : {timing['rk4']:.2f} s")
    if "ps" in timing:
        print(f"Run Time PS     : {timing['ps']:.2f} s")

    print(f"Norm Time       : {norm_time:.2e} s")
    print(f"Physical Time   : {physical_time:.2e} s")
    if orders_used is not None:
        print(f"PS Orders       : max={orders_used.max()}, mean={orders_used.mean():.1f}\n")

    # === Create run-specific output subfolders ===
    # data/hyperb/<config>/<stem>/figures/   ← plots
    # data/hyperb/<config>/<stem>/           ← summary, config copy
    run_folder = os.path.join(output_folder, stem)
    fig_folder = os.path.join(run_folder, "figures")
    os.makedirs(fig_folder, exist_ok=True)

    # --- Copy config YAML to run folder (with git hash) ---
    copy_config_to_output(cfg_path, run_folder)

    # --- Filename helper (shared stem for all plots) ---
    _base = f"{fig_folder}/{stem}"
    _field_label = "Hyperbolic B Field"
    _plot_kw = dict(particle_type=particle_type, field_label=_field_label, use_plot_titles=USE_PLOT_TITLES)

    # =====================================================
    # ============== Full 2D & 3D Trajectory Plots ========
    # =====================================================
    if USE_FULL_PLOT:
        _traj_kw = dict(
            solution_ps=solution_ps, orders_used=orders_used,
            solution_rk45=solution_rk45 if USE_RK45 else None,
            solution_rk4=solution_rk4 if USE_RK4 else None,
            use_rk45=USE_RK45, use_rk4=USE_RK4,
            **_plot_kw,
        )
        fplt.full_2d(f"{_base}_2D.png", **_traj_kw)
        fplt.full_3d(f"{_base}_3D.png", **_traj_kw)

    # =====================================================
    # ============== KE Relative Error Plot ===============
    # =====================================================
    time_factor = 1.0 / T_gyro  # convert normalized time to gyroperiods

    v_ps = solution_ps[3:6]
    E_ps = npfloat(0.5) * np.sum(v_ps**2, axis=0, dtype=npfloat)
    rel_drift_ps = np.abs(E_ps - E_ps[0]) / E_ps[0]
    final_ps = rel_drift_ps[-1]

    rel_drift_rk4 = None
    rel_drift_rk45 = None

    if USE_RK4:
        v_rk4 = solution_rk4[3:6]
        E_rk4 = npfloat(0.5) * np.sum(v_rk4**2, axis=0, dtype=npfloat)
        rel_drift_rk4 = np.abs(E_rk4 - E_rk4[0]) / E_rk4[0]

    if USE_RK45:
        v_rk45 = solution_rk45.y[3:6]
        E_rk45 = 0.5 * np.sum(v_rk45**2, axis=0)
        rel_drift_rk45 = np.abs(E_rk45 - E_rk45[0]) / E_rk45[0]

    fplt.ke_error(
        f"{_base}_KEerror.png",
        t_eval_ps=t_eval_ps, rel_drift_ps=rel_drift_ps, orders_used=orders_used,
        t_eval_rk4=t_eval_rk4 if USE_RK4 else None, rel_drift_rk4=rel_drift_rk4,
        t_eval_rk45=t_eval_rk45 if USE_RK45 else None, rel_drift_rk45=rel_drift_rk45,
        use_rk4=USE_RK4, use_rk45=USE_RK45, **_plot_kw,
    )

    # ==========================================
    # ================ Slicing  ================
    # ==========================================
    ps_x, ps_y, ps_z = ul.slice_solution_constb_hyperb(t_eval_ps, solution_ps, window_duration, norm_time, mode=slice_mode)[:3]

    if USE_RK45:
        rk45_x, rk45_y, rk45_z = ul.slice_solution_constb_hyperb(t_eval_rk45, solution_rk45.y, window_duration, norm_time, mode=slice_mode)[:3]
    if USE_RK4:
        rk4_x, rk4_y, rk4_z = ul.slice_solution_constb_hyperb(t_eval_rk4, solution_rk4, window_duration, norm_time, mode=slice_mode)[:3]

    # =====================================================
    # ================ 2D & 3D Trajectory Slices ==========
    # =====================================================
    fplt.slice_2d(
        f"{_base}_2Dslice.png",
        ps_x=ps_x, ps_y=ps_y, orders_used=orders_used,
        rk45_x=rk45_x if USE_RK45 else None, rk45_y=rk45_y if USE_RK45 else None,
        rk4_x=rk4_x if USE_RK4 else None, rk4_y=rk4_y if USE_RK4 else None,
        use_rk45=USE_RK45, use_rk4=USE_RK4,
        skip_rk4_slice=skip_rk4_slice,
        slice_ylim=slice_ylim, slice_ylim_top=slice_ylim_top, slice_equal_aspect=slice_equal_aspect,
        **_plot_kw,
    )

    if USE_FULL_PLOT:
        fplt.slice_3d(
            f"{_base}_3Dslice.png",
            ps_x=ps_x, ps_y=ps_y, ps_z=ps_z, orders_used=orders_used,
            rk45_x=rk45_x if USE_RK45 else None, rk45_y=rk45_y if USE_RK45 else None, rk45_z=rk45_z if USE_RK45 else None,
            rk4_x=rk4_x if USE_RK4 else None, rk4_y=rk4_y if USE_RK4 else None, rk4_z=rk4_z if USE_RK4 else None,
            use_rk45=USE_RK45, use_rk4=USE_RK4, **_plot_kw,
        )

    # ============================================================
    # ================ Multi-PS-order KE error ===================
    # ============================================================
    if USE_FULL_PLOT and not USE_FLOAT128:
        # --- Load external h5 data ---
        ext_data = None
        extb_data = None
        if USE_EXTERNAL_H5:
            external = wr.load_results_h5_hyperb(external_h5)
            ext_ps = external["ps"]
            t_ext, y_ext = ext_ps["t"], ext_ps["y"]
            vxe, vye, vze = y_ext[3].astype(np.float128), y_ext[4].astype(np.float128), y_ext[5].astype(np.float128)
            E_ext = 0.5 * (vxe**2 + vye**2 + vze**2)
            rel_drift_ext = (E_ext - E_ext[0]) / E_ext[0]
            ext_data = (t_ext, rel_drift_ext, PS_order_ext)

        if USE_EXTERNAL_H5b:
            externalb = wr.load_results_h5_hyperb(external_h5b)
            ext_psb = externalb["ps"]
            t_extb, y_extb = ext_psb["t"], ext_psb["y"]
            vxeb, vyeb, vzeb = y_extb[3].astype(np.float128), y_extb[4].astype(np.float128), y_extb[5].astype(np.float128)
            E_extb = 0.5 * (vxeb**2 + vyeb**2 + vzeb**2)
            rel_drift_extb = (E_extb - E_extb[0]) / E_extb[0]
            extb_data = (t_extb, rel_drift_extb, PS_order_extb)

        # --- Recompute PS at various orders ---
        _ps_orders = [5, 6, 7, 10, 15]
        _ps_colors = ["#0072B2", "#56B4E9", "#D55E00", "#000000", "#999999"]
        _ps_styles = ["--", ":", "-.", "--", "-"]
        ps_drifts = []
        for order, color, ls in zip(_ps_orders, _ps_colors, _ps_styles):
            sol, _ = hp.ps_integrate(order, steps_ps, initial_pos_vel, ps_step, gamma, qoverm, tol)
            vx, vy, vz = ea.extract_v(sol)
            drift = ea.compute_energy_drift(vx, vy, vz)
            ps_drifts.append((order, drift, color, ls))

        # Recompute RK drifts for the multi-PS plot
        if USE_RK4:
            vx_rk4 = np.array(solution_rk4[3], dtype=npfloat)
            vy_rk4 = np.array(solution_rk4[4], dtype=npfloat)
            vz_rk4 = np.array(solution_rk4[5], dtype=npfloat)
            rel_drift_rk4 = ea.compute_energy_drift(vx_rk4, vy_rk4, vz_rk4)
        if USE_RK45:
            vx_rk45 = np.array(solution_rk45.y[3], dtype=npfloat)
            vy_rk45 = np.array(solution_rk45.y[4], dtype=npfloat)
            vz_rk45 = np.array(solution_rk45.y[5], dtype=npfloat)
            rel_drift_rk45 = ea.compute_energy_drift(vx_rk45, vy_rk45, vz_rk45)

        # Add main PS (max order)
        ps_drifts.append((orders_used.max(), rel_drift_ps, "#009E73", ":"))

        fplt.ke_error_multi(
            f"{_base}_KEerror_many.png",
            t_eval_ps=t_eval_ps, orders_used=orders_used,
            ps_drifts=ps_drifts,
            t_eval_rk4=t_eval_rk4 if USE_RK4 else None, rel_drift_rk4=rel_drift_rk4,
            t_eval_rk45=t_eval_rk45 if USE_RK45 else None, rel_drift_rk45=rel_drift_rk45,
            use_rk4=USE_RK4, use_rk45=USE_RK45,
            ext_data=ext_data, extb_data=extb_data,
            energy_xlim_left=energy_xlim_left,
            **_plot_kw,
        )

    # ====================================
    # === Write Summary Output to File ===
    # ====================================

    output_filename = f"{run_folder}/{stem}_summary.txt"

    wr.summary_txt_hyperb(
        output_filename,
        stem=stem, WRITE_DATA=WRITE_DATA, READ_DATA=READ_DATA,
        particle_type=particle_type, KE_particle=KE_particle, mass_si=mass_si,
        pitch_deg=pitch_deg, phi_deg=phi_deg,
        tau_time=tau_time, v_tau=v_tau, gyro_radius_si=gyro_radius_si,
        x_initial_si=x_initial_si, y_initial_si=y_initial_si, z_initial_si=z_initial_si,
        vx_initial=vx_initial, vy_initial=vy_initial, vz_initial=vz_initial,
        delta=delta, B_0=B_0, gamma=gamma,
        npfloat_name=npfloat.__name__,
        norm_time=norm_time, physical_time=physical_time, gyroperiods=gyroperiods,
        ps_step=ps_step, rk4_step=rk4_step,
        steps_ps=steps_ps, steps_rk4=steps_rk4 if USE_RK4 else None,
        orders_used=orders_used,
        USE_RK4=USE_RK4, USE_RK45=USE_RK45,
        timing=timing,
        rel_drift_ps=rel_drift_ps,
        rel_drift_rk4=rel_drift_rk4,
        rel_drift_rk45=rel_drift_rk45,
    )

    print(f"\nRun Complete → {run_folder}")


if __name__ == "__main__":
    run = "demo"
    if len(sys.argv) > 1:
        run = sys.argv[1]
        print(f"Run mode set from command line: {run}\n")
    else:
        print(f"Using default run mode: {run}\n")

    _configs_dir = os.path.join(os.path.dirname(__file__), "configs", "hyperb")

    if run.endswith((".yml", ".yaml")) and os.path.isfile(run):
        _yaml_path = run
    elif os.path.isfile(os.path.join(_configs_dir, f"{run}.yml")):
        _yaml_path = os.path.join(_configs_dir, f"{run}.yml")
    else:
        raise FileNotFoundError(
            f"No YAML config found for '{run}'. "
            f"Expected configs/hyperb/{run}.yml or a direct path to a .yml file.\n"
            f"Available configs: {[f.replace('.yml','') for f in os.listdir(_configs_dir) if f.endswith('.yml') and f != 'base.yml']}"
        )

    print(f"Loading YAML config: {_yaml_path}\n")
    main(_yaml_path)
