import numpy as np
import builtins
import test_particles.hyperB_testparticles as tp
builtins.npfloat = np.float128 if tp.USE_FLOAT128 else np.float64
from test_particles.hyperB_testparticles import *
import numpy as np
import os
import time
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import LogLocator, LogFormatterSciNotation, NullFormatter, FuncFormatter
from functions.functions_library_hyper import PS_hyperB, lorentz_force_hyperB
from functions.functions_library_universal import rk4_fixed_step, extract_v, compute_energy_drift, plt_config, sparse_labels, data_to_fig
from functions.functions_library_hyper import get_run_params, h5_path_for, save_results_h5, load_results_h5

run = "demo"   # options: "paper" or "demo"

globals().update(load_params(run))

# === Misc Odds and Ends ===
mpl.rcParams['agg.path.chunksize'] = 100000  
run_storage = "/Users/heatherjiles/Documents/Grad School/Parker-Sochacki Method/run_data"    # where trajectory files go
os.makedirs(run_storage, exist_ok=True)
plt_config(scale=1)    # Dr. W's Plotting SCrip
plt.ioff()              # Turn off interactive mode for plots

# for file naming
if mass_si == m_e:
    particle_type = "Electron"
elif mass_si == m_p:
    particle_type = "Proton"
else:
    particle_type = "Particle"

qoverm = npfloat(-1) if mass_si == m_e else npfloat(1)


# === Misc Normalization  ===
pitch_rad = np.radians(pitch_deg)
phi_rad = np.radians(phi_deg)

v_si = npfloat(np.sqrt(npfloat(2 * KE_particle * evtoj / mass_si)))/1000       # /1000 puts things in km
tau_time =  mass_si/(abs(q_e)*B_0)             # τ0 from paper, time normalization constant  

gyro_radius_si=abs(v_si * np.sin(pitch_rad) * mass_si/ (q_e * B_0))
r_normalization = delta
v_tau = v_si*tau_time/r_normalization    # velocity in dimensionless units
gamma= 1/(delta/r_normalization)         # if normalizing by delta this should be 1 

physical_time = norm_time * tau_time

x_initial = npfloat(x_initial_si/r_normalization)              
y_initial = npfloat(y_initial_si/r_normalization)  
z_initial = npfloat(z_initial_si/r_normalization)

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
steps_ps = int(norm_time / ps_step) # PS is normalized
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
    timing = cached.get("meta", {}).get("timing", {})
    stem = os.path.splitext(os.path.basename(cache_path))[0]

else:
    print("No matching file or 'Read Data' skipped. Running solvers...this will take a hot second...\n")
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
        solution_rk4 = rk4_fixed_step(
            lorentz_force_hyperB, initial_pos_vel, 
            t_eval_rk4, args=(gamma,qoverm))
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
            "particle_label": (
                f"{KE_particle:.1e} eV electron" if mass_si == m_e else
                f"{KE_particle:.1e} eV proton" if mass_si == m_p else
                "manual"
            ),
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
    ax.set_xlabel(r"$x/\delta$")
    ax.set_ylabel(r"$y/\delta$")
    if USE_PLOT_TITLES: ax.set_title(f"2D {particle_type} Trajectory in Hyperbolic B Field")
    ax.legend(loc="upper right")
    ax.axis('equal')
    ax.grid(True)
    plt.tight_layout()

    # === Save and Close ===
    fig.canvas.draw()   
    fig.savefig( f"{output_folder}/{stem}_HyperB_{particle_type}_{KE_particle:.1e}eV_{ps_step}step_{delta}delta_PS{orders_used.max()}_{norm_time:.1f}s_{npfloat.__name__}_2D.png", dpi=600, bbox_inches="tight")
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

    ax.plot(solution_ps[0], solution_ps[1], solution_ps[2], label='RK4 ', color='#009E73', linestyle=':')


    # === Labels and Legend ===
    ax.set_xlabel(r"$x/\delta$")
    ax.set_ylabel(r"$y/\delta$")
    ax.set_zlabel(r"$z/\delta$")

    if USE_PLOT_TITLES: ax.set_title(f"3D {particle_type} Trajectory in Hyperbolic B Field")
    ax.legend(loc="upper right")
    plt.tight_layout()

    # === Save and Close ===
    fig.canvas.draw()   
    fig.savefig( f"{output_folder}/{stem}_HyperB_{particle_type}_{KE_particle:.1e}eV_{ps_step}step_{delta}delta_PS{orders_used.max()}_{norm_time:.1f}s_{npfloat.__name__}_3D.png", dpi=600, bbox_inches="tight")
    plt.close(fig)  

# =====================================================
# ============== KE Relative Error Plot ===============
# =====================================================

v_ps = solution_ps[3:6]        
E_ps = npfloat(0.5) * np.sum(v_ps**2, axis=0, dtype=npfloat)
rel_drift_ps = np.abs(E_ps - E_ps[0]) / E_ps[0]

if USE_RK4:
    v_rk4 = solution_rk4[3:6]  
    E_rk4 = npfloat(0.5) * np.sum(v_rk4**2, axis=0, dtype=npfloat)
    rel_drift_rk4 = np.abs(E_rk4 - E_rk4[0]) / E_rk4[0]

if USE_RK45:
    v_rk45 = solution_rk45.y[3:6]
    E_rk45 = 0.5 * np.sum(v_rk45**2, axis=0)
    rel_drift_rk45 = np.abs(E_rk45 - E_rk45[0]) / E_rk45[0]

# === Plot =====
fig, ax = plt.subplots(figsize=(10, 5))

if USE_RK45: line1, = ax.semilogy(t_eval_rk45, np.abs(rel_drift_rk45), color='#E69F00', linestyle='--')
if USE_RK4: line2, = ax.semilogy(t_eval_rk4, np.abs(rel_drift_rk4), color='#CC79A7', linestyle='-.')
line3, = ax.semilogy(t_eval_ps,  np.abs(rel_drift_ps),  color='#009E73',  linestyle=':')

# === Formats ===
ax.margins(x=0.01)
ax.set_yscale('log') 
ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=100))
ax.yaxis.set_major_formatter(LogFormatterSciNotation(base=10.0))  # or LogFormatterMathtext()
ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=[]))
ax.yaxis.set_minor_formatter(NullFormatter())
ax.grid(False, which='both')
ax.grid(True, which='major', linestyle='--', linewidth=0.7)
ax.yaxis.set_major_formatter(FuncFormatter(sparse_labels))

# Remove top and right borders
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# === Labels and Legend ===
ax.set_xlabel(r"$t/\tau_0$")
ax.set_ylabel(r"$|\Delta E|/E_0$")
if USE_PLOT_TITLES: ax.set_title(f"{particle_type} Relative Kinetic Energy Error in Hyperbolic B Field")

# building out labels for methods at endpoints
fig.subplots_adjust(right=0.9)
fig.canvas.draw()
ax_pos = ax.get_position()  # Bbox in figure coords
x_fig_label = ax_pos.x1 + 0.01  # a small gap to the right of axes

endpoints = []
if USE_RK45:
    endpoints.append((t_eval_rk45[-1], np.abs(rel_drift_rk45[-1]), "RK45", line1.get_color()))
if USE_RK4:
    endpoints.append((t_eval_rk4[-1], np.abs(rel_drift_rk4[-1]), "RK4", line2.get_color()))
endpoints.append((t_eval_ps[-1], np.abs(rel_drift_ps[-1]), f"PS{orders_used.max()}", line3.get_color()))

# --- collision avoidance block replaces your old for-loop ---
labels = []
for x, y, label, color in endpoints:
    _, fy = data_to_fig(x, y, ax, fig)
    # Clamp to axis bounds
    fy = min(max(fy, ax_pos.y0), ax_pos.y1)
    labels.append([fy, label, color])

# Sort by vertical position
labels.sort(key=lambda v: v[0])

# Minimum vertical spacing in figure coords
min_gap = 0.02  
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
             va='center', ha='left', fontsize=10)

# === Save and Close ===
fig.canvas.draw()   
fig.savefig( f"{output_folder}/{stem}_HyperB_{particle_type}_{KE_particle:.1e}eV_{ps_step}step_{delta}delta_PS{orders_used.max()}_{norm_time:.1f}s_{npfloat.__name__}_KEerror.png", dpi=600, bbox_inches="tight")
plt.close(fig)  

# =====================================================
# ================ 2D Trajectory Slice ================
# =====================================================

# === Extract last some number of last steps from the simulation ===
window_duration = gyro_plot_slice * 2 * np.pi
start_t_ps   = norm_time - window_duration
start_idx_ps   = np.searchsorted(t_eval_ps, start_t_ps) # Find corresponding indices
ps_x, ps_y, ps_z = solution_ps[0][start_idx_ps:], solution_ps[1][start_idx_ps:], solution_ps[2][start_idx_ps:] # Slice solutions


if USE_RK4:
    start_t_rk4  = norm_time - window_duration
    start_idx_rk4  = np.searchsorted(t_eval_rk4, start_t_rk4)
    rk4_x, rk4_y, rk4_z = solution_rk4[0][start_idx_rk4:], solution_rk4[1][start_idx_rk4:], solution_rk4[2][start_idx_rk4:]

if USE_RK45:
    start_t_rk45  = norm_time - window_duration
    start_idx_rk45  = np.searchsorted(t_eval_rk45, start_t_rk45)
    rk45_x, rk45_y, rk45_z = solution_rk45.y[0][start_idx_rk45:], solution_rk45.y[1][start_idx_rk45:], solution_rk45.y[2][start_idx_rk45:]

# === Plot Last Few Orbits ===
fig, ax = plt.subplots(figsize=(10, 7))
if USE_RK45:
    ax.plot(rk45_x, rk45_y, label=f"RK45", color='#E69F00', linestyle='--')
if USE_RK4:
    ax.plot(rk4_x, rk4_y, label=f"RK4", color='#CC79A7', linestyle='-.')

ax.plot(ps_x, ps_y, label=f"PS{orders_used.max()}", color='#009E73', linestyle=':')

# === Labels and Legend ===
ax.set_xlabel(r"$x/\delta$")
ax.set_ylabel(r"$y/\delta$")
if USE_PLOT_TITLES: ax.set_title(f"2D Trajectory of Final {particle_type} Orbits in Hyperbolic B Field")

# ===  Axis Limits for Paper ===
# ax.set_ylim(-40, 40) # hard coded for now, adjust as needed for paper
# ax.set_aspect('equal', adjustable='box')
ax.axis('equal')
ax.legend(loc="upper right")
ax.grid(True)

# ===  Save and Close ===
fig.canvas.draw()   
fig.savefig( f"{output_folder}/{stem}_HyperB_{particle_type}_{KE_particle:.1e}eV_{ps_step}step_{delta}delta_PS{orders_used.max()}_{norm_time:.1f}s_{npfloat.__name__}_2Dslice.png", dpi=600, bbox_inches="tight")
plt.close(fig)  

# ======================================
# ============= Slice of 3D ============
# ======================================
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot each trajectory segment
if USE_RK45:
    ax.plot(rk45_x, rk45_y, rk45_z, label=f"RK45", color='#E69F00', linestyle='--')
if USE_RK4:
    ax.plot(rk4_x, rk4_y, rk4_z, label=f"RK4", color='#CC79A7', linestyle='-.')
ax.plot(ps_x, ps_y, ps_z, label=f"PS{orders_used.max()}", color='#009E73', linestyle=':')

ax.set_xlabel(r"$x/\delta$")
ax.set_ylabel(r"$y/\delta$")
ax.set_zlabel(r"$z/\delta$")
if USE_PLOT_TITLES: ax.set_title(f'3D Trajectory of Final {particle_type} Orbits in Hyperbolic B Field')
ax.legend(loc="upper right")

# === Save and Close ===
fig.canvas.draw()  
fig.savefig( f"{output_folder}/{stem}_HyperB_{particle_type}_{KE_particle:.1e}eV_{ps_step}step_{delta}delta_PS{orders_used.max()}_{norm_time:.1f}s_{npfloat.__name__}_3Dslice.png", dpi=600, bbox_inches="tight")
plt.close(fig)



# ============================================================
# ================ Plotting multiple PS Orders ===============
# ============================================================
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

if USE_RK45:
    vx_rk45 = np.array(solution_rk45.y[3], dtype=npfloat)
    vy_rk45 = np.array(solution_rk45.y[4], dtype=npfloat)
    vz_rk45 = np.array(solution_rk45.y[5], dtype=npfloat)
    rel_drift_rk45 = compute_energy_drift(vx_rk45, vy_rk45, vz_rk45)

# === Plot energy drift ===
def f64(x): return np.array(x, dtype=np.float64)

fig, ax = plt.subplots(figsize=(10, 5))
if USE_RK45:
    lnrk45, = ax.semilogy(f64(t_eval_rk45[1:]), np.abs(f64(rel_drift_rk45[1:])), label="RK45", linestyle='-',  color='#E69F00')   # orange
if USE_RK4:
    lnrk4, = ax.semilogy(f64(t_eval_rk4[1:]),  np.abs(f64(rel_drift_rk4[1:])),  label="RK4", linestyle='-.', color='#CC79A7')   # reddish purple

lnps5, = ax.semilogy(f64(t_eval_ps[1:]), np.abs(f64(rel_drift_ps_5[1:])),  label="PS5", linestyle='--', color='#0072B2')   # blue
lnps6, = ax.semilogy(f64(t_eval_ps[1:]), np.abs(f64(rel_drift_ps_6[1:])),  label="PS6", linestyle=':',  color='#56B4E9')   # sky blue
lnps7, = ax.semilogy(f64(t_eval_ps[1:]), np.abs(f64(rel_drift_ps_7[1:])),  label="PS7", linestyle='-.', color='#D55E00')   # vermillion
lnps10, = ax.semilogy(f64(t_eval_ps[1:]), np.abs(f64(rel_drift_ps_10[1:])), label="PS10", linestyle='--', color='#000000')   # black
lnps15, = ax.semilogy(f64(t_eval_ps[1:]), np.abs(f64(rel_drift_ps_15[1:])), label="PS15", linestyle='-',  color='#999999')   # gray
lnps, = ax.semilogy(t_eval_ps[1:], np.abs(rel_drift_ps[1:]), label=f"PS{orders_used.max()}", linestyle=':', color='#009E73')    # bluish green

ax.margins(x=0.01)
ax.set_yscale('log') 
ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=100))
ax.yaxis.set_major_formatter(LogFormatterSciNotation(base=10.0))  # or LogFormatterMathtext()
ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=[]))
ax.yaxis.set_minor_formatter(NullFormatter())
ax.grid(False, which='both')
ax.grid(True, which='major', linestyle='--', linewidth=0.7)
ax.yaxis.set_major_formatter(FuncFormatter(sparse_labels))

# Remove top and right borders
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# === Labels and Legend ===
ax.set_xlabel(r"$t/\tau_0$")
ax.set_ylabel(r"$|\Delta E|/E_0$")
if USE_PLOT_TITLES: ax.set_title(f"{particle_type} Relative Kinetic Energy Error in Hyperbolic B Field")

# building out labels for methods
fig.subplots_adjust(right=0.9)
fig.canvas.draw()
ax_pos = ax.get_position()  # Bbox in figure coords
x_fig_label = ax_pos.x1 + 0.01  # a small gap to the right of axes

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

labels = []
for x, y, label, color in endpoints:
    _, fy = data_to_fig(x, y, ax, fig)
    fy = min(max(fy, ax_pos.y0), ax_pos.y1)
    labels.append([fy, label, color])

# Sort by vertical position
labels.sort(key=lambda v: v[0])

# Minimum vertical spacing in figure coords
min_gap = 0.02  
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
             va='center', ha='left', fontsize=10)


# ===  Save and Close ===
fig.savefig( f"{output_folder}/{stem}_HyperB_{particle_type}_{KE_particle:.1e}eV_{ps_step}step_{delta}delta_PS{orders_used.max()}_{norm_time:.1f}s_{npfloat.__name__}_KEerror_many.png", dpi=600, bbox_inches="tight")
plt.close(fig)  

# ====================================
# === Write Summary Output to File ===
# ====================================

finalnum = max(1, int(steps_ps * 0.01)) # Number of steps to average over, last 1%

def summarize_error(label, err, f):
    mean_val = np.mean(err[-finalnum:])
    max_val  = np.max(np.abs(err[-finalnum:]))
    rms_val  = np.sqrt(np.mean(err[-finalnum:]**2))
    f.write(f"  {label:<8}: mean = {mean_val:.2e}, max = {max_val:.2e}, rms = {rms_val:.2e}\n")

output_filename = f"{output_folder}/{stem}_HyperB_{particle_type}_{KE_particle:.1e}eV_{ps_step}step_{delta}delta_PS{orders_used.max()}_{norm_time:.1f}s_{npfloat.__name__}_simulation_summary.txt"

with open(output_filename, "w") as f:
    if WRITE_DATA or READ_DATA: f.write(f"Run Data: {stem}.hd5\n\n")
    f.write("=== Simulation Summary ===\n")
    f.write(f"Initial Conditions:\n")
    f.write(f"  particle      = {particle_type}\n")
    f.write(f"  mass          = {mass_si} kg\n")
    f.write(f"  Energy        = {KE_particle} eV\n")
    f.write(f"  pitch_deg     = {pitch_deg}\n")
    f.write(f"  phi_deg       = {phi_deg}\n")
    f.write(f"  tau           = {tau_time} s\n")
    f.write(f"  v_tau         = {v_tau}\n")
    f.write(f"  x_initial     = {x_initial} km\n")
    f.write(f"  y_initial     = {y_initial} km\n")
    f.write(f"  z_initial     = {z_initial} km\n")
    f.write(f"  vx_initial    = {vx_initial}\n")
    f.write(f"  vy_initial    = {vy_initial}\n")
    f.write(f"  vz_initial    = {vz_initial}\n")
    f.write(f"  delta         = {delta} km\n")
    f.write(f"  Initial Bfield= {B_0} T\n")
    f.write(f"  float type    = {npfloat.__name__}\n\n")

    
    f.write("=== Timing Summary ===\n")
    if USE_RK45:
        f.write(f"  Run Time RK45 = {timing['rk45']:.2f} s\n")    
    if USE_RK4:
        f.write(f"  Run Time RK4  = {timing['rk4']:.2f} s\n")
    f.write(f"  Run Time PS   = {timing['ps']:.2f} s\n")
    f.write(f"  PS Orders     = max={orders_used.max()}, mean={orders_used.mean():.1f}\n")
    f.write(f"  norm time     = {norm_time}\n")
    f.write(f"  physical time = {physical_time:.2e} s\n")
    if USE_RK4:
        f.write(f"  rk4 step size = {rk4_step}\n")
    f.write(f"  ps step size  = {ps_step}\n")
    if USE_RK4:
        f.write(f"  rk4 steps     = {steps_rk4}\n")
    f.write(f"  ps steps      = {steps_ps}\n\n")

    f.write(f"=== |ΔE|/E0 (relative, last {finalnum} steps)===\n")
    if USE_RK45:
        summarize_error("RK45", rel_drift_rk45, f)
    if USE_RK4:
        summarize_error("RK4",  rel_drift_rk4, f)
    summarize_error("PS",   rel_drift_ps,  f)


