"""
hyperB.py — Driver for charged particle trajectory simulation in a
            hyperbolic tangent magnetic field using power series, RK4,
            and RK45 solvers.

Usage:
    python hyperB.py                          # default config (demo)
    python hyperB.py demo                     # named config → configs/hyper/demo.yml
    python hyperB.py paper1                   # named config → configs/hyper/paper1.yml
    python hyperB.py configs/hyper/my.yml     # direct path to a custom YAML config
"""

import numpy as np
import builtins
import os
import time
import sys
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import LogLocator, LogFormatterSciNotation, NullFormatter, FuncFormatter

from configs.config_loader import load_config, compute_derived_hyper
from ps_method.constants import q_e, evtoj
from ps_method.hyper_physics import PS_hyperB, lorentz_force_hyperB
from ps_method.universal import rk4_fixed_step, extract_v, compute_energy_drift, plt_config, sparse_labels, data_to_fig, slice_solution
from ps_method.writers import get_run_params_hyper as get_run_params, h5_path_for, save_results_h5_hyper as save_results_h5, load_results_h5_hyper as load_results_h5, write_summary_txt_hyper

# === Load YAML Config ===
run = "demo"
if len(sys.argv) > 1:
    run = sys.argv[1]
    print(f"Run mode set from command line: {run}\n")
else:
    print(f"Using default run mode: {run}\n")

_configs_dir = os.path.join(os.path.dirname(__file__), "configs", "hyper")

if run.endswith((".yml", ".yaml")) and os.path.isfile(run):
    _yaml_path = run
elif os.path.isfile(os.path.join(_configs_dir, f"{run}.yml")):
    _yaml_path = os.path.join(_configs_dir, f"{run}.yml")
else:
    raise FileNotFoundError(
        f"No YAML config found for '{run}'. "
        f"Expected configs/hyper/{run}.yml or a direct path to a .yml file.\n"
        f"Available configs: {[f.replace('.yml','') for f in os.listdir(_configs_dir) if f.endswith('.yml') and f != 'base.yml']}"
    )

print(f"Loading YAML config: {_yaml_path}\n")
cfg = load_config(_yaml_path)

# --- Resolve float type BEFORE compute_derived (needs to be set for builtins) ---
USE_FLOAT128 = cfg.get("use_float128", False)
if USE_FLOAT128:
    npfloat = np.float128
else:
    npfloat = np.float64
builtins.npfloat = npfloat

p = compute_derived_hyper(cfg, npfloat=npfloat)

# === Unpack Config ===
READ_DATA       = p["READ_DATA"]
WRITE_DATA      = p["WRITE_DATA"]
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
ps_step     = p["ps_step"]
rk4_step    = p["rk4_step"]

PS_order    = p["PS_order"]
tol         = p["tol"]
rtol_rk45   = p["rtol_rk45"]
atol_rk45   = p["atol_rk45"]

# === Misc Odds and Ends ===
os.makedirs(run_storage, exist_ok=True)
os.makedirs(output_folder, exist_ok=True)
plt_config(scale=1)
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
initial_pos_vel_ps = np.array([x_initial, y_initial, z_initial, vx_initial, vy_initial, vz_initial], dtype=npfloat)  


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
params = get_run_params(USE_RK45, USE_RK4, KE_particle, rtol_rk45, atol_rk45,
                   mass_si, q_e, B_0, delta,
                   x_initial, y_initial, z_initial,
                   pitch_deg, phi_deg,
                   norm_time, ps_step, rk4_step,
                   PS_order, tol, qoverm)
cache_path = h5_path_for(params, run_storage)

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
        t_eval_rk45 = cached["rk45"]["t"]
   
    timing = cached.get("meta", {}).get("timing", {})
    stem = os.path.splitext(os.path.basename(cache_path))[0]

else:
    print("No matching file or 'Read Data' skipped. Running solvers...\n")
    # ====== Run RK45 ======
    if USE_RK45:
        start_time_rk45 = time.time()
        solution_rk45 = solve_ivp(
            lorentz_force_hyperB, (0, norm_time), 
            initial_pos_vel,method='RK45', 
            t_eval=t_eval_rk45, args=(gamma,qoverm),
            rtol= rtol_rk45,
            atol= atol_rk45) 
        end_time_rk45 = time.time()

    # ====== Run RK4 ======
    if USE_RK4:
        start_time_rk4 = time.time()
        rk4_dt = npfloat(t_eval_rk4[1] - t_eval_rk4[0])
        solution_rk4 = rk4_fixed_step(
            lorentz_force_hyperB, initial_pos_vel,
            rk4_dt, steps_rk4, args=(gamma,qoverm))
        end_time_rk4 = time.time()

    # ===== Run PS Method ====
    start_time_ps = time.time()
    solution_ps, orders_used = PS_hyperB(
        PS_order, steps_ps, 
        initial_pos_vel_ps, ps_step, gamma, 
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
        save_results_h5(cache_path, params, results)
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

# =====================================================
# ============== Full 2D Trajectory Plot ==============
# =====================================================
if USE_FULL_PLOT:
    fig, ax = plt.subplots(figsize=(10, 8))

    if USE_RK45:
        ax.plot(solution_rk45.y[0], solution_rk45.y[1], label="RK45", color='#E69F00', linestyle='--')
    if USE_RK4:
        ax.plot(solution_rk4[0], solution_rk4[1], label="RK4", color='#CC79A7', linestyle='-.')
    ax.plot(solution_ps[0], solution_ps[1], label=f"PS{orders_used.max()}", color='#009E73', linestyle=':')

    # === Labels and Legend ===
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    if USE_PLOT_TITLES: ax.set_title(f"2D {particle_type} Trajectory in Hyperbolic B Field")
    ax.legend(loc="upper right")
    ax.axis('equal')
    ax.grid(True)
    plt.tight_layout()

    # === Save and Close ===
    fig.canvas.draw()   
    fig.savefig( f"{output_folder}/{stem}_HyperB_{particle_type}_{KE_particle:.1e}eV_{ps_step}step_{delta}delta_PS{orders_used.max()}_{gyroperiods:.1e}_{npfloat.__name__}_2D.png", dpi=600, bbox_inches="tight")
    plt.close(fig)  

# =====================================================
# ============== Full 3D Trajectory Plot ==============
# =====================================================
if USE_FULL_PLOT:
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # === Plot Trajectories ===
    if USE_RK45:
        ax.plot(solution_rk45.y[0], solution_rk45.y[1], solution_rk45.y[2], label='RK45 ', color='#E69F00', linestyle='-.')
    if USE_RK4:
        ax.plot(solution_rk4[0], solution_rk4[1], solution_rk4[2], label='RK4 ', color='#CC79A7', linestyle=':')

    ax.plot(solution_ps[0], solution_ps[1], solution_ps[2], label=f"PS{orders_used.max()}", color='#009E73', linestyle=':')


    # === Labels and Legend ===
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    ax.set_zlabel(r"$z$")

    if USE_PLOT_TITLES: ax.set_title(f"3D {particle_type} Trajectory in Hyperbolic B Field")
    ax.legend(loc="upper right")
    plt.tight_layout()

    # === Save and Close ===
    fig.canvas.draw()   
    fig.savefig( f"{output_folder}/{stem}_HyperB_{particle_type}_{KE_particle:.1e}eV_{ps_step}step_{delta}delta_PS{orders_used.max()}_{gyroperiods:.1e}_{npfloat.__name__}_3D.png", dpi=600, bbox_inches="tight")
    plt.close(fig)  

# =====================================================
# ============== KE Relative Error Plot ===============
# =====================================================
time_factor = 1.0 / (2.0 * np.pi)  # to convert to gyroperiods if desired

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
    ratio_rk4_ps = rel_drift_rk4[-1]/final_ps
    order_mag_rk4 = int(np.floor(np.log10(abs(ratio_rk4_ps))))

if USE_RK45:
    v_rk45 = solution_rk45.y[3:6]
    E_rk45 = 0.5 * np.sum(v_rk45**2, axis=0)
    rel_drift_rk45 = np.abs(E_rk45 - E_rk45[0]) / E_rk45[0]
    ratio_rk45_ps = rel_drift_rk45[-1]/final_ps
    order_mag_rk45 = int(np.floor(np.log10(abs(ratio_rk45_ps))))

if USE_FULL_PLOT:
    # === Plot =====
    fig, ax = plt.subplots(figsize=(10, 5))

    if USE_RK45: line1, = ax.semilogy(t_eval_rk45 * time_factor, np.abs(rel_drift_rk45), color='#E69F00', linestyle='--')
    if USE_RK4: line2, = ax.semilogy(t_eval_rk4 * time_factor, np.abs(rel_drift_rk4), color='#CC79A7', linestyle='-.')
    line3, = ax.semilogy(t_eval_ps * time_factor,  np.abs(rel_drift_ps),  color='#009E73',  linestyle=':')

    # Getting log lines to work, mess with at your own risk
    ax.margins(x=0.01)
    ax.set_yscale('log') 
    ax.set_xscale('log') 
    ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=100))
    ax.yaxis.set_major_formatter(LogFormatterSciNotation(base=10.0))  
    ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=[]))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.xaxis.set_major_locator(LogLocator(base=10.0, numticks=100))
    ax.xaxis.set_major_formatter(LogFormatterSciNotation(base=10.0))  
    ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=[]))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.grid(True, which='major', linestyle='--', linewidth=0.7)
    ax.yaxis.set_major_formatter(FuncFormatter(sparse_labels))
    
    # ax.xaxis.set_major_formatter(FuncFormatter(sparse_labels))
    # ax.set_aspect('equal', adjustable='box')

    # Remove top and right borders
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # === Labels and Legend ===
    ax.set_xlabel(r"$\tau/T$")
    ax.set_ylabel(r"$|\Delta E|/E_0$")
    if USE_PLOT_TITLES: ax.set_title(f"{particle_type} Relative Kinetic Energy Error in Hyperbolic B Field")

    # building out labels for methods at endpoints
    fig.subplots_adjust(right=0.9)
    fig.canvas.draw()
    ax_pos = ax.get_position()  # Bbox in figure coords
    x_fig_label = ax_pos.x1

    endpoints = []
    if USE_RK45:
        endpoints.append((t_eval_rk45[-1], np.abs(rel_drift_rk45[-1]), f"RK45", line1.get_color()))
    if USE_RK4:
        endpoints.append((t_eval_rk4[-1], np.abs(rel_drift_rk4[-1]), f"RK4", line2.get_color()))
    endpoints.append((t_eval_ps[-1], np.abs(rel_drift_ps[-1]), f"PS{orders_used.max()}", line3.get_color()))

    labels = []
    for x, y, label, color in endpoints:
        _, fy = data_to_fig(x, y, ax, fig)
        # Clamp to axis bounds
        fy = min(max(fy, ax_pos.y0), ax_pos.y1)
        labels.append([fy, label, color])

    # Sort by vertical position
    labels.sort(key=lambda v: v[0])

    # Minimum vertical spacing in figure coords
    min_gap = 0.025  
    for i in range(1, len(labels)):
        if labels[i][0] - labels[i-1][0] < min_gap:
            labels[i][0] = labels[i-1][0] + min_gap

    # Clamp from the top back downward
    for i in range(len(labels)-2, -1, -1):
        if labels[i+1][0] - labels[i][0] < min_gap:
            labels[i][0] = labels[i+1][0] - min_gap

    # Draw adjusted labels
    for fy, label, color in labels:
        fig.text(x_fig_label, fy, label, color=color,
                va='center', ha='left', fontsize=11)

    # === Save and Close ===
    fig.canvas.draw()   
    fig.savefig( f"{output_folder}/{stem}_HyperB_{particle_type}_{KE_particle:.1e}eV_{ps_step}step_{delta}delta_PS{orders_used.max()}_{gyroperiods:.1e}_{npfloat.__name__}_KEerror.png", dpi=600, bbox_inches="tight")
    plt.close(fig)  

# ==========================================
# ================ Slicing  ================
# ==========================================
ps_x, ps_y, ps_z = slice_solution(t_eval_ps, solution_ps, window_duration, norm_time, mode=slice_mode)[:3]

if USE_RK45:
    rk45_x, rk45_y, rk45_z = slice_solution(t_eval_rk45, solution_rk45.y, window_duration, norm_time, mode=slice_mode)[:3]
if USE_RK4:
    rk4_x, rk4_y, rk4_z = slice_solution(t_eval_rk4, solution_rk4, window_duration, norm_time, mode=slice_mode)[:3]

# =====================================================
# ================ 2D Trajectory Slice ================
# =====================================================

# === Plot Last Few Orbits ===
fig, ax = plt.subplots(figsize=(10, 5))
if USE_RK45:
    ax.plot(rk45_x, rk45_y, label=f"RK45", color='#E69F00', linestyle='--')
if USE_RK4 and not skip_rk4_slice:
    ax.plot(rk4_x, rk4_y, label=f"RK4", color='#CC79A7', linestyle='-.')

ax.plot(ps_x, ps_y, label=f"PS{orders_used.max()}", color='#009E73', linestyle=':')

# === Labels and Legend ===
ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$y$")
if USE_PLOT_TITLES: ax.set_title(f"2D Trajectory of Final {particle_type} Orbits in Hyperbolic B Field")

# ===  Axis Limits ===
ax.ticklabel_format(style='plain', useOffset=False, axis='both')
ax.axis('equal')
if slice_ylim is not None:
    ax.set_ylim(slice_ylim[0], slice_ylim[1])
if slice_ylim_top is not None:
    ax.set_ylim(top=slice_ylim_top)
if slice_equal_aspect:
    ax.set_aspect('equal', adjustable='box')
ax.legend(loc="upper right")
ax.grid(True)

# ===  Save and Close ===
fig.canvas.draw()   
fig.savefig( f"{output_folder}/{stem}_HyperB_{particle_type}_{KE_particle:.1e}eV_{ps_step}step_{delta}delta_PS{orders_used.max()}_{gyroperiods:.1e}_{npfloat.__name__}_2Dslice.png", dpi=600, bbox_inches="tight")
plt.close(fig)  

# ======================================
# ============= Slice of 3D ============
# ======================================

if USE_FULL_PLOT:
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot each trajectory segment
    if USE_RK45:
        ax.plot(rk45_x, rk45_y, rk45_z, label=f"RK45", color='#E69F00', linestyle='--')
    if USE_RK4:
        ax.plot(rk4_x, rk4_y, rk4_z, label=f"RK4", color='#CC79A7', linestyle='-.')
    ax.plot(ps_x, ps_y, ps_z, label=f"PS{orders_used.max()}", color='#009E73', linestyle=':')

    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    ax.set_zlabel(r"$z$")
    if USE_PLOT_TITLES: ax.set_title(f'3D Trajectory of Final {particle_type} Orbits in Hyperbolic B Field')
    ax.legend(loc="upper right")

    # === Save and Close ===
    fig.canvas.draw()  
    fig.savefig( f"{output_folder}/{stem}_HyperB_{particle_type}_{KE_particle:.1e}eV_{ps_step}step_{delta}delta_PS{orders_used.max()}_{gyroperiods:.1e}_{npfloat.__name__}_3Dslice.png", dpi=600, bbox_inches="tight")
    plt.close(fig)



# ============================================================
# ================ Plotting multiple PS Orders ===============
# ============================================================
if not USE_FLOAT128:
    if USE_EXTERNAL_H5:
        external = load_results_h5(external_h5)
        ext_ps = external["ps"]
        t_ext  = ext_ps["t"]          
        y_ext  = ext_ps["y"]          

        vxe = y_ext[3].astype(np.float128)
        vye = y_ext[4].astype(np.float128)
        vze = y_ext[5].astype(np.float128)
        E_ext = 0.5 * (vxe**2 + vye**2 + vze**2)
        rel_drift_ext = (E_ext - E_ext[0]) / E_ext[0]

    if USE_EXTERNAL_H5b:
        externalb = load_results_h5(external_h5b)
        ext_psb = externalb["ps"]
        t_extb  = ext_psb["t"]          
        y_extb  = ext_psb["y"]          

        vxeb = y_extb[3].astype(np.float128)
        vyeb = y_extb[4].astype(np.float128)
        vzeb = y_extb[5].astype(np.float128)
        E_extb = 0.5 * (vxeb**2 + vyeb**2 + vzeb**2)
        rel_drift_extb = (E_extb - E_extb[0]) / E_extb[0]


    # === Compute PS solutions at various orders ===
    solution_ps_5, _ = PS_hyperB(5, steps_ps, initial_pos_vel_ps, ps_step, gamma, qoverm, tol)
    solution_ps_6, _ = PS_hyperB(6, steps_ps, initial_pos_vel_ps, ps_step, gamma, qoverm, tol)
    solution_ps_7, _ = PS_hyperB(7, steps_ps, initial_pos_vel_ps, ps_step, gamma, qoverm, tol)
    solution_ps_10, _ = PS_hyperB(10, steps_ps, initial_pos_vel_ps, ps_step, gamma, qoverm, tol)
    solution_ps_15, _ = PS_hyperB(15, steps_ps, initial_pos_vel_ps, ps_step, gamma, qoverm, tol)

    # === Compute drifts ===
    vx5, vy5, vz5 = extract_v(solution_ps_5)
    vx6, vy6, vz6 = extract_v(solution_ps_6)
    vx7, vy7, vz7 = extract_v(solution_ps_7)
    vx10, vy10, vz10 = extract_v(solution_ps_10)
    vx15, vy15, vz15 = extract_v(solution_ps_15)

    rel_drift_ps_5  = compute_energy_drift(vx5, vy5, vz5)
    rel_drift_ps_6  = compute_energy_drift(vx6, vy6, vz6)
    rel_drift_ps_7  = compute_energy_drift(vx7, vy7, vz7)
    rel_drift_ps_10 = compute_energy_drift(vx10, vy10, vz10)
    rel_drift_ps_15 = compute_energy_drift(vx15, vy15, vz15)

    # === RK4 and RK45 velocities (use already computed) ===
    if USE_RK4:
        vx_rk4 = np.array(solution_rk4[3], dtype=npfloat)
        vy_rk4 = np.array(solution_rk4[4], dtype=npfloat)
        vz_rk4 = np.array(solution_rk4[5], dtype=npfloat)
        rel_drift_rk4  = compute_energy_drift(vx_rk4, vy_rk4, vz_rk4)
        ratio_rk4_ps = rel_drift_rk4[-1]/final_ps
        order_mag_rk4 = int(np.floor(np.log10(abs(ratio_rk4_ps))))

    if USE_RK45:
        vx_rk45 = np.array(solution_rk45.y[3], dtype=npfloat)
        vy_rk45 = np.array(solution_rk45.y[4], dtype=npfloat)
        vz_rk45 = np.array(solution_rk45.y[5], dtype=npfloat)
        rel_drift_rk45 = compute_energy_drift(vx_rk45, vy_rk45, vz_rk45)
        ratio_rk45_ps = rel_drift_rk45[-1]/final_ps
        order_mag_rk45 = int(np.floor(np.log10(abs(ratio_rk45_ps))))

    # === Plot energy drift ===
    def f64(x): return np.array(x, dtype=np.float64)

    fig, ax = plt.subplots(figsize=(10, 5))
    if USE_RK45:
        lnrk45, = ax.semilogy(f64(t_eval_rk45[1:])*time_factor, np.abs(f64(rel_drift_rk45[1:])), label=f"RK45 ({order_mag_rk45})", linestyle='-',  color='#E69F00')   # orange
    if USE_RK4:
        lnrk4, = ax.semilogy(f64(t_eval_rk4[1:])*time_factor,  np.abs(f64(rel_drift_rk4[1:])),  label=f"RK4 ({order_mag_rk4})", linestyle='-.', color='#CC79A7')   # reddish purple

    lnps5, = ax.semilogy(f64(t_eval_ps[1:])*time_factor, np.abs(f64(rel_drift_ps_5[1:])),  label="PS5", linestyle='--', color='#0072B2')   # blue
    lnps6, = ax.semilogy(f64(t_eval_ps[1:])*time_factor, np.abs(f64(rel_drift_ps_6[1:])),  label="PS6", linestyle=':',  color='#56B4E9')   # sky blue
    lnps7, = ax.semilogy(f64(t_eval_ps[1:])*time_factor, np.abs(f64(rel_drift_ps_7[1:])),  label="PS7", linestyle='-.', color='#D55E00')   # vermillion
    lnps10, = ax.semilogy(f64(t_eval_ps[1:])*time_factor, np.abs(f64(rel_drift_ps_10[1:])), label="PS10", linestyle='--', color='#000000')   # black
    lnps15, = ax.semilogy(f64(t_eval_ps[1:])*time_factor, np.abs(f64(rel_drift_ps_15[1:])), label="PS15", linestyle='-',  color='#999999')   # gray
    lnps, = ax.semilogy(t_eval_ps[1:]*time_factor, np.abs(rel_drift_ps[1:]), label=f"PS{orders_used.max()}", linestyle=':', color='#009E73')    # bluish green
    if USE_EXTERNAL_H5:
        ln_ext, = ax.semilogy(f64(t_ext[1:]) * time_factor, np.abs(f64(rel_drift_ext[1:])), linestyle='-.', linewidth=1.2, color='black')
    if USE_EXTERNAL_H5b:
        ln_extb, = ax.semilogy(f64(t_extb[1:]) * time_factor, np.abs(f64(rel_drift_extb[1:])), linestyle='-', linewidth=1.2, color='#6A3D9A')


    # Getting log lines to work, mess with at your own risk
    ax.margins(x=0.01)
    ax.set_yscale('log') 
    ax.set_xscale('log') 
    if energy_xlim_left is not None:
        ax.set_xlim(left=energy_xlim_left)
    ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=100))
    ax.yaxis.set_major_formatter(LogFormatterSciNotation(base=10.0))  
    ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=[]))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.xaxis.set_major_locator(LogLocator(base=10.0, numticks=100))
    ax.xaxis.set_major_formatter(LogFormatterSciNotation(base=10.0))  
    ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=[]))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.grid(True, which='major', linestyle='--', linewidth=0.7)
    ax.yaxis.set_major_formatter(FuncFormatter(sparse_labels))
    # ax.xaxis.set_major_formatter(FuncFormatter(sparse_labels))
    # ax.set_aspect('equal', adjustable='box')

    # Remove top and right borders
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # === Labels and Legend ===
    ax.set_xlabel(r"$\tau/T$")
    ax.set_ylabel(r"$|\Delta E|/E_0$")
    if USE_PLOT_TITLES: ax.set_title(f"{particle_type} Relative Kinetic Energy Error in Hyperbolic B Field")

    # building out labels for methods
    fig.subplots_adjust(right=0.9)
    fig.canvas.draw()
    ax_pos = ax.get_position()  # Bbox in figure coords
    x_fig_label = ax_pos.x1 

    endpoints = []
    if USE_RK45:
        endpoints.append((t_eval_rk45[-1], np.abs(rel_drift_rk45[-1]), "RK45", lnrk45.get_color()))
    if USE_RK4:
        endpoints.append((t_eval_rk4[-1], np.abs(rel_drift_rk4[-1]), "RK4", lnrk4.get_color()))

    ps_endpoints = [(t_eval_ps[-1],  np.abs(rel_drift_ps_5[-1]),   f"PS5", lnps5.get_color()),
                    (t_eval_ps[-1],  np.abs(rel_drift_ps_6[-1]),   f"PS6", lnps6.get_color()),
                    (t_eval_ps[-1],  np.abs(rel_drift_ps_7[-1]),   f"PS7", lnps7.get_color()),
                    (t_eval_ps[-1],  np.abs(rel_drift_ps_10[-1]),   f"PS10", lnps10.get_color()),    
                    (t_eval_ps[-1],  np.abs(rel_drift_ps_15[-1]),   f"PS15", lnps15.get_color()),   
                    (t_eval_ps[-1],  np.abs(rel_drift_ps[-1]),   f"PS{orders_used.max()}", lnps.get_color())
                    ]
    endpoints.extend(ps_endpoints)

    if USE_EXTERNAL_H5:
        endpoints.append(
        (t_ext[-1], np.abs(rel_drift_ext[-1]), f"PS{PS_order_ext}*", ln_ext.get_color())
    )
    if USE_EXTERNAL_H5b:
        endpoints.append(
        (t_extb[-1], np.abs(rel_drift_extb[-1]), f"PS{PS_order_extb}*", ln_extb.get_color())
    )    

    labels = []
    for x, y, label, color in endpoints:
        _, fy = data_to_fig(x, y, ax, fig)
        fy = min(max(fy, ax_pos.y0), ax_pos.y1)
        labels.append([fy, label, color])

    # Sort by vertical position
    labels.sort(key=lambda v: v[0])

    # Minimum vertical spacing in figure coords
    min_gap = 0.025  
    for i in range(1, len(labels)):
        if labels[i][0] - labels[i-1][0] < min_gap:
            labels[i][0] = labels[i-1][0] + min_gap

    # Clamp from the top back downward
    for i in range(len(labels)-2, -1, -1):
        if labels[i+1][0] - labels[i][0] < min_gap:
            labels[i][0] = labels[i+1][0] - min_gap

    # Draw adjusted labels
    for fy, label, color in labels:
        fig.text(x_fig_label, fy, label, color=color,
                va='center', ha='left', fontsize=11)


    # ===  Save and Close ===
    fig.savefig( f"{output_folder}/{stem}_HyperB_{particle_type}_{KE_particle:.1e}eV_{ps_step}step_{delta}delta_PS{orders_used.max()}_{gyroperiods:.1e}_{npfloat.__name__}_KEerror_many.png", dpi=600, bbox_inches="tight")
    plt.close(fig)  

# ====================================
# === Write Summary Output to File ===
# ====================================

output_filename = f"{output_folder}/{stem}_HyperB_{particle_type}_{KE_particle:.1e}eV_{ps_step}step_{delta}delta_PS{orders_used.max()}_{gyroperiods:.1e}_{npfloat.__name__}_simulation_summary.txt"

write_summary_txt_hyper(
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

print(f"\nRun Complete → {output_folder}/{stem}")

