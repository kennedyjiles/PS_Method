"""
constB.py — Driver for charged particle trajectory simulation in a
            uniform magnetic field using power series, RK4, RK45, and
            analytical solutions.

Usage:
    python constB.py                          # default config (demo)
    python constB.py demo                     # named config → configs/constB/demo.yml
    python constB.py paper                    # named config → configs/constB/paper.yml
    python constB.py configs/constB/my.yml    # direct path to a custom YAML config
"""

import numpy as np
import builtins
import os
import time
import sys
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib as mpl

from configs.config_loader import load_config, compute_derived_constB
from ps_method.constants import q_e, evtoj

# === Load YAML Config ===
run = "demo"
if len(sys.argv) > 1:
    run = sys.argv[1]
    print(f"Run mode set from command line: {run}\n")
else:
    print(f"Using default run mode: {run}\n")

_configs_dir = os.path.join(os.path.dirname(__file__), "configs", "constB")

if run.endswith((".yml", ".yaml")) and os.path.isfile(run):
    _yaml_path = run
elif os.path.isfile(os.path.join(_configs_dir, f"{run}.yml")):
    _yaml_path = os.path.join(_configs_dir, f"{run}.yml")
else:
    raise FileNotFoundError(
        f"No YAML config found for '{run}'. "
        f"Expected configs/constB/{run}.yml or a direct path to a .yml file.\n"
        f"Available configs: {[f.replace('.yml','') for f in os.listdir(_configs_dir) if f.endswith('.yml') and f != 'base.yml']}"
    )

print(f"Loading YAML config: {_yaml_path}\n")
cfg = load_config(_yaml_path)
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
from ps_method.constB_physics import PS_constB, analytical_constantB, lorentz_force_constB
from ps_method.universal import rk4_fixed_step, extract_v, compute_energy_drift, plt_config
from ps_method.field_plots import (
    plot_full_2d, plot_full_3d, plot_ke_error, plot_slice_2d, plot_slice_3d,
    plot_ke_error_multi, plot_trajectory_error, f64,
)
from ps_method.writers import get_run_params_constB as get_run_params, h5_path_for, save_results_h5_constB as save_results_h5, load_results_h5_constB as load_results_h5, write_summary_txt_constB

p = compute_derived_constB(cfg, npfloat=npfloat)

# === Unpack Config ===
READ_DATA       = p["READ_DATA"]
WRITE_DATA      = p["WRITE_DATA"]
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
plt_config(scale=1)
plt.ioff()

if USE_FLOAT128:
    mpl.rcParams['agg.path.chunksize'] = 100000
else:
    mpl.rcParams['agg.path.chunksize'] = 1000

particle_type = p["particle"].capitalize()

qoverm = npfloat(-1) if p["particle"].lower() in ("electron", "e") else npfloat(1)

# === Misc Normalizing  ===
B_0 = np.linalg.norm(Bfield_si)  # Magnitude of the field
Bfield = Bfield_si / B_0         # normalized B field
v_si = npfloat(np.sqrt(npfloat(2 * KE_particle * evtoj / mass)))
tau_time = mass / (abs(q_e) * B_0)
v_tau = v_si * tau_time
physical_time = norm_time * tau_time

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

gyro_radius_si = abs(v_si * np.sin(pitch_rad) * mass / (q_e * B_0))


initial_pos_vel = np.array([x_initial, y_initial, z_initial, vx_initial, vy_initial, vz_initial], dtype=npfloat)  
initial_pos_vel_ps = np.array([x_initial, y_initial, z_initial, vx_initial, vy_initial, vz_initial], dtype=npfloat)  

# === Ensures that Total Time Elapsed is the Same ===
"""
Note that here the norm time is being rounded so that the steps are to the nearest interger multiple 
of the PS step size. This ensures there aren't phase errors with plotting the trajectories. If PS and RK
have the same step size, everything is good. If you adjust their step sizes you may need to 
incorporate some additional interpolation. For the purposes of the study, I just wanted a one to
one comparison. This has no impact on the energy calculations though.
"""
steps_ps = int(round(norm_time / ps_step))
norm_time = steps_ps * ps_step           # <-- adjust total time to be exact multiple
t_eval_ps = np.linspace(0, norm_time, steps_ps + 1, dtype=npfloat)

if USE_RK4:
    steps_rk4 = int(round(norm_time / rk4_step))
    t_eval_rk4 = np.linspace(0, norm_time, steps_rk4 + 1, dtype=npfloat)

if USE_RK45:
    steps_rk45 = steps_ps   # just for plotting consistency
    t_eval_rk45 = np.float64(t_eval_ps)

phase_warning = False
if not np.isclose(ps_step, rk4_step, rtol=1e-12):
    phase_warning = True

if phase_warning:
    print(
        "⚠️  Warning: PS and RK4 step sizes or total times do not align exactly.\n"
        "    → Energy drift comparisons are fine.\n"
        "    → Trajectory and phase comparisons may show artificial offsets."
    )

if USE_ANALYTICAL: 
    start_time_analytical = time.time()
    solution_analytical = analytical_constantB(
        t_eval_ps,
        initial_pos_vel, qoverm)
    end_time_analytical = time.time()

# === Build parameter signature & check cache ===
params = get_run_params(USE_RK45, USE_RK4, KE_particle, rtol_rk45, atol_rk45,
                   mass, q_e, B_0,
                   x_initial, y_initial, z_initial,
                   pitch_deg, phi_deg,
                   norm_time, ps_step, rk4_step,
                   PS_order, tol, qoverm)
cache_path = h5_path_for(params, run_storage)
os.makedirs(run_storage, exist_ok=True)

if os.path.exists(cache_path) and READ_DATA:
    print(f"Found existing results: {os.path.basename(cache_path)} — loading.\n")
    cached = load_results_h5(cache_path)

    # Rehydrate what you need for plotting/analysis:
    solution_ps = cached["ps"]["y"] if cached["ps"] else None
    orders_used = cached["ps"]["orders"] if cached["ps"] else None
    t_eval_ps = cached["ps"]["t"] if cached["ps"] else None

    if USE_RK4 and cached["rk4"]:
        solution_rk4 = cached["rk4"]["y"]
        t_eval_rk4 = cached["rk4"]["t"]
    if USE_RK45 and cached["rk45"]:
        class _Obj: pass
        solution_rk45 = _Obj()
        solution_rk45.t = cached["rk45"]["t"]
        solution_rk45.y = cached["rk45"]["y"]   

    timing = cached.get("meta", {}).get("timing", {})
    stem = os.path.splitext(os.path.basename(cache_path))[0]


else:
    print("No matching file or 'Read Data' skipped. Running solvers...\n")
    # ====== Run RK45 ======
    if USE_RK45:
        start_time_rk45 = time.time()
        solution_rk45 = solve_ivp(
            lorentz_force_constB, (0, norm_time), 
            initial_pos_vel,method='RK45', 
            t_eval=t_eval_rk45, args=(Bfield,qoverm),
            rtol=rtol_rk45,
            atol=atol_rk45) 
        end_time_rk45 = time.time()

    # ====== Run RK4 ======
    if USE_RK4:
        start_time_rk4 = time.time()
        rk4_dt = npfloat(t_eval_rk4[1] - t_eval_rk4[0])
        solution_rk4 = rk4_fixed_step(
            lorentz_force_constB, initial_pos_vel,
            rk4_dt, steps_rk4, args=(Bfield,qoverm))
        end_time_rk4 = time.time()

    # ===== Run PS Method ====
    start_time_ps = time.time()
    solution_ps, orders_used=PS_constB(
        PS_order, steps_ps, initial_pos_vel_ps, 
        ps_step, Bfield, qoverm, tol)
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
        save_results_h5(cache_path, params, results)
        print(f"Saved results → {os.path.basename(cache_path)}")
    stem = os.path.splitext(os.path.basename(cache_path))[0]


# === Sanity Check ===
print(f"Particle        : {KE_particle:.1e} eV {particle_type}")
if USE_RK45 and "rk45" in timing:
    print(f"Run Time RK45   : {timing['rk45']:.2f} s")
if USE_RK4 and "rk4" in timing:
    print(f"Run Time RK4    : {timing['rk4']:.2f} s")
if "ps" in timing:
    print(f"Run Time PS     : {timing['ps']:.2f} s")
if USE_ANALYTICAL and "ana" in timing:
    print(f"Run Time Exact    : {timing['ana']:.2f} s")
print(f"Norm Time       : {norm_time:.2e} s")
print(f"Physical Time   : {physical_time:.2e} s")
if orders_used is not None:
    print(f"PS Orders       : max={orders_used.max()}, mean={orders_used.mean():.1f}\n")

# === Create run-specific output subfolders ===
# data/constB/<config>/<stem>/figures/   ← plots
# data/constB/<config>/<stem>/output/    ← text summaries
_run_name = f"{stem}_ConstB_{particle_type}_{KE_particle:.1e}eV_{ps_step}step_PS{orders_used.max()}_{gyroperiods:.1e}_{npfloat.__name__}"
run_folder = os.path.join(output_folder, _run_name)
fig_folder = os.path.join(run_folder, "figures")
out_folder = os.path.join(run_folder, "output")
os.makedirs(fig_folder, exist_ok=True)
os.makedirs(out_folder, exist_ok=True)

# --- Write config log to output folder ---
if _config_log:
    _config_log_path = os.path.join(out_folder, "config_log.txt")
    with open(_config_log_path, "w") as _f:
        _f.write("\n".join(_config_log))
    print(f"Config log written to {_config_log_path}\n")

# --- Filename helper (shared stem for all plots) ---
_base = f"{fig_folder}/{_run_name}"
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
    plot_full_2d(f"{_base}_2D.png", **_traj_kw)
    plot_full_3d(f"{_base}_3D.png", **_traj_kw)

# ================================================================
# ==================KE Error Plot Over time Only =================
# ================================================================
time_factor = 1.0 / T_gyro  # convert normalized time to gyroperiods

v_ps = solution_ps[3:6]
E_ps = npfloat(0.5) * np.sum(v_ps**2, axis=0, dtype=npfloat)
rel_drift_ps = (E_ps - E_ps[0]) / E_ps[0]
final_ps = rel_drift_ps[-1]

rel_drift_rk4 = None
rel_drift_rk45 = None

if USE_RK45:
    v_rk45 = solution_rk45.y[3:6]
    E_rk45 = 0.5 * np.sum(v_rk45**2, axis=0)
    rel_drift_rk45 = (E_rk45 - E_rk45[0]) / E_rk45[0]

if USE_RK4:
    v_rk4 = solution_rk4[3:6]
    E_rk4 = npfloat(0.5) * np.sum(v_rk4**2, axis=0, dtype=npfloat)
    rel_drift_rk4 = (E_rk4 - E_rk4[0]) / E_rk4[0]

plot_ke_error(
    f"{_base}_KEerror.png",
    t_eval_ps=t_eval_ps, rel_drift_ps=rel_drift_ps, orders_used=orders_used,
    t_eval_rk4=t_eval_rk4 if USE_RK4 else None, rel_drift_rk4=rel_drift_rk4,
    t_eval_rk45=t_eval_rk45 if USE_RK45 else None, rel_drift_rk45=rel_drift_rk45,
    use_rk4=USE_RK4, use_rk45=USE_RK45, **_plot_kw,
)

# ======================================
# ============= Slicing ================
# ======================================
window_duration = gyro_plot_slice * 2 * np.pi

if USE_RK4:
    _si = np.searchsorted(t_eval_rk4, norm_time - window_duration)
    rk4_x, rk4_y, rk4_z = solution_rk4[0][_si:], solution_rk4[1][_si:], solution_rk4[2][_si:]

if USE_RK45:
    _si = np.searchsorted(t_eval_rk45, norm_time - window_duration)
    rk45_x, rk45_y, rk45_z = solution_rk45.y[0][_si:], solution_rk45.y[1][_si:], solution_rk45.y[2][_si:]

if USE_ANALYTICAL:
    _si = np.searchsorted(t_eval_ps, norm_time - window_duration)
    ana_x, ana_y, ana_z = solution_analytical[0][_si:], solution_analytical[1][_si:], solution_analytical[2][_si:]

_si = np.searchsorted(t_eval_ps, norm_time - window_duration)
ps_x, ps_y, ps_z = solution_ps[0][_si:], solution_ps[1][_si:], solution_ps[2][_si:]

_slice_kw = dict(
    ps_x=ps_x, ps_y=ps_y, orders_used=orders_used,
    rk45_x=rk45_x if USE_RK45 else None, rk45_y=rk45_y if USE_RK45 else None,
    rk4_x=rk4_x if USE_RK4 else None, rk4_y=rk4_y if USE_RK4 else None,
    ana_x=ana_x if USE_ANALYTICAL else None, ana_y=ana_y if USE_ANALYTICAL else None,
    use_rk45=USE_RK45, use_rk4=USE_RK4, use_analytical=USE_ANALYTICAL,
    **_plot_kw,
)

if USE_FULL_PLOT:
    plot_slice_2d(f"{_base}_2Dslice.png", **_slice_kw)
    plot_slice_3d(
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
    # --- Load external h5 data ---
    ext_data = None
    extb_data = None
    if USE_EXTERNAL_H5:
        external = load_results_h5(external_h5)
        ext_ps = external["ps"]
        t_ext, y_ext = ext_ps["t"], ext_ps["y"]
        vxe, vye, vze = y_ext[3].astype(np.float128), y_ext[4].astype(np.float128), y_ext[5].astype(np.float128)
        E_ext = 0.5 * (vxe**2 + vye**2 + vze**2)
        rel_drift_ext = (E_ext - E_ext[0]) / E_ext[0]
        ext_data = (t_ext, rel_drift_ext, PS_order_ext)

    if USE_EXTERNAL_H5b:
        externalb = load_results_h5(external_h5b)
        ext_psb = externalb["ps"]
        t_extb, y_extb = ext_psb["t"], ext_psb["y"]
        vxeb, vyeb, vzeb = y_extb[3].astype(np.float128), y_extb[4].astype(np.float128), y_extb[5].astype(np.float128)
        E_extb = 0.5 * (vxeb**2 + vyeb**2 + vzeb**2)
        rel_drift_extb = (E_extb - E_extb[0]) / E_extb[0]
        extb_data = (t_extb, rel_drift_extb, PS_order_extb)

    # --- Recompute PS at various orders ---
    _ps_orders = [4, 5, 6, 7, 10]
    _ps_colors = ["crimson", "#0072B2", "#56B4E9", "#D55E00", "#999999"]
    _ps_styles = [":", "-.", ":", "--", "-."]
    ps_drifts = []
    for order, color, ls in zip(_ps_orders, _ps_colors, _ps_styles):
        sol, _ = PS_constB(order, steps_ps, initial_pos_vel_ps, ps_step, Bfield, qoverm, tol)
        vx, vy, vz = extract_v(sol)
        drift = compute_energy_drift(vx, vy, vz)
        ps_drifts.append((order, drift, color, ls))

    # Add the main PS (max order)
    if USE_RK4:
        vx_rk4 = np.array(solution_rk4[3], dtype=npfloat)
        vy_rk4 = np.array(solution_rk4[4], dtype=npfloat)
        vz_rk4 = np.array(solution_rk4[5], dtype=npfloat)
        rel_drift_rk4 = compute_energy_drift(vx_rk4, vy_rk4, vz_rk4)
    if USE_RK45:
        vx_rk45 = np.array(solution_rk45.y[3], dtype=npfloat)
        vy_rk45 = np.array(solution_rk45.y[4], dtype=npfloat)
        vz_rk45 = np.array(solution_rk45.y[5], dtype=npfloat)
        rel_drift_rk45 = compute_energy_drift(vx_rk45, vy_rk45, vz_rk45)

    ps_drifts.append((orders_used.max(), rel_drift_ps, "#009E73", ":"))

    plot_ke_error_multi(
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
    x_ana = solution_analytical[0]
    y_ana = solution_analytical[1]
    rel_err_ps = np.sqrt((solution_ps[0] - x_ana)**2 + (solution_ps[1] - y_ana)**2) / gyro_radius_si

    rel_err_rk4 = None
    if USE_RK4:
        rel_err_rk4 = np.sqrt((solution_rk4[0] - x_ana)**2 + (solution_rk4[1] - y_ana)**2) / gyro_radius_si

    rel_err_rk45 = None
    if USE_RK45:
        rel_err_rk45 = np.sqrt((solution_rk45.y[0] - x_ana)**2 + (solution_rk45.y[1] - y_ana)**2) / gyro_radius_si

    # External h5 trajectory error
    t_ext_traj = None
    rel_err_ext = None
    if USE_EXTERNAL_H5:
        external = load_results_h5(external_h5)
        ext_ps = external["ps"]
        t_ext_traj = ext_ps["t"]
        y_ext = ext_ps["y"]
        rel_err_ext = np.sqrt((y_ext[0] - x_ana)**2 + (y_ext[1] - y_ana)**2) / gyro_radius_si

    plot_trajectory_error(
        f"{_base}_TrajError.png",
        t_eval_ps=t_eval_ps, rel_err_ps=rel_err_ps, orders_used=orders_used,
        t_eval_rk4=t_eval_rk4 if USE_RK4 else None, rel_err_rk4=rel_err_rk4,
        t_eval_rk45=t_eval_rk45 if USE_RK45 else None, rel_err_rk45=rel_err_rk45,
        t_ext=t_ext_traj, rel_err_ext=rel_err_ext, ps_order_ext=PS_order_ext,
        use_rk4=USE_RK4, use_rk45=USE_RK45, use_external_h5=USE_EXTERNAL_H5,
        use_full_plot=USE_FULL_PLOT,
        field_label="a Constant Magnetic Field", **{k: v for k, v in _plot_kw.items() if k != "field_label"},
    )


# ============================================
# ======= Write Summary Output to File =======
# ============================================

output_filename = f"{out_folder}/{_run_name}_SimSummary.txt"

write_summary_txt_constB(
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
