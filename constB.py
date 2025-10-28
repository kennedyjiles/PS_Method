import numpy as np
import builtins
import test_particles.constB_testparticles as tp
builtins.npfloat = np.float128 if tp.USE_FLOAT128 else np.float64
from test_particles.constB_testparticles import *
import time
import sys
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import LogLocator, LogFormatterSciNotation, NullFormatter, FuncFormatter
from functions.functions_library_constB import PS_constantB_adaptive, analytical_constantB, lorentz_force_constB, vector_error
from functions.functions_library_universal import rk4_fixed_step, extract_v, compute_energy_drift, plt_config, sparse_labels, interp_to_grid, interp_to_grid, data_to_fig

run = "demo"   # options: "paper" or "demo"

# Allow command-line override
if len(sys.argv) > 1:
    run = sys.argv[1]
    print(f"Run mode set from command line: {run}\n")
else:
    print(f"Using default run mode: {run}\n")

globals().update(load_params(run))

# === Misc Odds and Ends ===
mpl.rcParams['agg.path.chunksize'] = 100  
plt_config(scale=1) 
plt.ioff()              # Turns off interactive mode for figures

if mass == m_e:
    particle_type = "Electron"
elif mass == m_p:
    particle_type = "Proton"
else:
    particle_type = "Particle"

qoverm = npfloat(-1) if mass == m_e else npfloat(1)

# === Misc Normalizing  ===
B_0 = np.linalg.norm(Bfield_si)  # Magnitude of the field
Bfield = Bfield_si/B_0           # normalized B field
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

# ====== Run PS ======
start_time_ps = time.time()
solution_ps, orders_used=PS_constantB_adaptive(
    PS_order, steps_ps, initial_pos_vel_ps, 
    ps_step, Bfield, qoverm, tol)
end_time_ps = time.time()

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
    solution_rk4 = rk4_fixed_step(
        lorentz_force_constB, initial_pos_vel, 
        t_eval_rk4, args=(Bfield,qoverm))
    end_time_rk4 = time.time()

# ====== Run Analytical ======
if USE_ANALYTICAL: 
    start_time_analytical = time.time()
    solution_analytical = analytical_constantB(
        t_eval_ps, initial_pos_vel, Bfield, qoverm)
    end_time_analytical = time.time()

# === Sanity Summary ===
print(f"Particle            : {KE_particle} eV {particle_type}")
print(f"PS Run Time         : {end_time_ps - start_time_ps:.2f} s")
if USE_RK4: 
    print(f"RK4 Run Time        : {end_time_rk4 - start_time_rk4:.2f} s")
if USE_RK45: 
    print(f"RK45 Run Time       : {end_time_rk45 - start_time_rk45:.2f} s")
if USE_ANALYTICAL: 
    print(f"Analytical Run Time : {end_time_analytical - start_time_analytical:.2f} s")
print(f"Normalized Time     : {norm_time:.2e} ")
print(f"PS Orders           : max={orders_used.max()}, mean={orders_used.mean():.1f}\n")

# ===================================================================
# ================Full 2D Plots of All Trajectories==================
# ===================================================================
if USE_FULL_PLOT:
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plotting the 2D trajectory
    if USE_ANALYTICAL: 
        ax.plot(solution_analytical[0], solution_analytical[1], color='black', linestyle='-', linewidth=0.3, label="Analytical")
    if USE_RK45:
        ax.plot(solution_rk45.y[0], solution_rk45.y[1], color='#E69F00', linestyle='--', label="RK45")
    if USE_RK4:
        ax.plot(solution_rk4[0], solution_rk4[1], color='#CC79A7', linestyle='--', linewidth=0.75, label="RK4")
    ax.plot(solution_ps[0], solution_ps[1], color='#009E73', linestyle=':', label=f"PS{orders_used.max()}")

    # === Labels and Legend ===
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    if USE_PLOT_TITLES: ax.set_title(f'2D {particle_type} Trajectory in Constant B Field')
    ax.legend(loc="upper right")
    ax.axis('equal')
    ax.grid(True)

    fig.canvas.draw()   
    fig.savefig( f"{output_folder}/ConstB_{particle_type}_{KE_particle:.1e}eV_{ps_step}step_PS{orders_used.max()}_{norm_time}s_{npfloat.__name__}_2D.png", dpi=600, bbox_inches="tight")
    plt.close(fig)      

# ===================================================================
# ================Full 3D Plots of All Trajectories==================
# ===================================================================
if USE_FULL_PLOT:
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # # === Plot Trajectories ===
    if USE_ANALYTICAL:
        ax.plot(solution_analytical[0], solution_analytical[1], solution_analytical[2], color='black', linestyle='-', linewidth=0.3, label="Analytical")
    if USE_RK45:
        ax.plot(solution_rk45.y[0], solution_rk45.y[1], solution_rk45.y[2], label='RK45', color='#E69F00', linestyle='--')
    if USE_RK4:
        ax.plot(solution_rk4[0], solution_rk4[1], solution_rk4[2], label='RK4', color='#CC79A7', linestyle='-.')

    ax.plot(solution_ps[0], solution_ps[1], solution_ps[2], label=f"PS{orders_used.max()}", color='#009E73', linestyle=':')

    # === Labels and Legend ===
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_zlabel('z (m)')
    if USE_PLOT_TITLES: ax.set_title(f'3D {particle_type} Trajectory in Constant B Field')
    ax.legend(loc="upper right")

    # === Save and Close ===
    fig.canvas.draw()   
    plt.savefig( f"{output_folder}/ConstB_{particle_type}_{KE_particle:.1e}eV_{ps_step}step_PS{orders_used.max()}_{norm_time}s_{npfloat.__name__}_3D.png", dpi=600, bbox_inches="tight")
    plt.close(fig)      

# ================================================================
# ==================KE Error Plot Over time Only =================
# ================================================================

v_ps = solution_ps[3:6]  
E_ps = npfloat(0.5) * np.sum(v_ps**2, axis=0, dtype=npfloat)
rel_drift_ps = (E_ps - E_ps[0]) / E_ps[0]


if USE_FULL_PLOT:
    if USE_RK45:
        v_rk45 = solution_rk45.y[3:6]  
        E_rk45 = 0.5 * np.sum(v_rk45**2, axis=0)
        rel_drift_rk45 = (E_rk45 - E_rk45[0]) / E_rk45[0]

    if USE_RK4:
        v_rk4 = solution_rk4[3:6]      
        E_rk4 = npfloat(0.5) * np.sum(v_rk4**2, axis=0, dtype=npfloat)
        rel_drift_rk4 = (E_rk4 - E_rk4[0]) / E_rk4[0]

    # === Plot ===
    fig, ax = plt.subplots(figsize=(10, 5))

    if USE_RK45: line1, = ax.semilogy(t_eval_rk45, np.abs(rel_drift_rk45), color='#E69F00', linestyle='--')
    if USE_RK4: line2, = ax.semilogy(t_eval_rk4, np.abs(rel_drift_rk4), color='#CC79A7', linestyle='-.')
    line3, = ax.semilogy(t_eval_ps,  np.abs(rel_drift_ps),  color='#009E73',  linestyle=':')

    ax.margins(x=0.01)
    ax.set_yscale('log')
    ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=100))
    ax.yaxis.set_major_formatter(LogFormatterSciNotation(base=10.0))
    ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=[]))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.grid(True, which='major', linestyle='--', linewidth=0.7)
    ax.yaxis.set_major_formatter(FuncFormatter(sparse_labels))

    # Remove top and right borders
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.set_xlabel(r"t/$\tau_0$")
    ax.set_ylabel(r"$|\Delta E|/E_0$")
    if USE_PLOT_TITLES: ax.set_title(f'{particle_type} Relative Kinetic Energy Error in Constant B Field')


    # make room on the right (no tight_layout; it can fight placements)
    fig.subplots_adjust(right=0.9)  # leaves ~10% margin for labels
    fig.canvas.draw()

    ax_pos = ax.get_position()  # Bbox in figure coords
    x_fig_label = ax_pos.x1 + 0.01  # a small gap to the right of axes

    # making method endpoint labels
    endpoints = []
    if USE_RK45:
        endpoints.append((t_eval_rk45[-1], np.abs(rel_drift_rk45[-1]), "RK45", line1.get_color()))
    if USE_RK4:
        endpoints.append((t_eval_rk4[-1], np.abs(rel_drift_rk4[-1]), "RK4", line2.get_color()))
    endpoints.append((t_eval_ps[-1], np.abs(rel_drift_ps[-1]), f"PS{orders_used.max()}", line3.get_color()))

    for x, y, label, color in endpoints:
        _, fy = data_to_fig(x, y, ax, fig)
        fy = min(max(fy, ax_pos.y0), ax_pos.y1)
        fig.text(x_fig_label, fy, label, color=color, va='center', ha='left', fontsize=10)


    # === Save and Close ===
    fig.canvas.draw()   
    plt.savefig( f"{output_folder}/ConstB_{particle_type}_{KE_particle:.1e}eV_{ps_step}step_PS{orders_used.max()}_{norm_time}s_{npfloat.__name__}_KEerror.png", dpi=600, bbox_inches="tight")
    plt.close(fig)      
            

# ======================================
# ============= Slice of 2D ============
# ======================================
# === Extract number of last steps from the simulation ===
window_duration = gyro_plot_slice * 2 * np.pi

if USE_RK4:
    start_t_rk4  = norm_time - window_duration
    start_idx_rk4  = np.searchsorted(t_eval_rk4, start_t_rk4)
    rk4_x, rk4_y, rk4_z = solution_rk4[0][start_idx_rk4:], solution_rk4[1][start_idx_rk4:], solution_rk4[2][start_idx_rk4:]

if USE_RK45:
    start_t_rk45  = norm_time - window_duration
    start_idx_rk45  = np.searchsorted(t_eval_rk45, start_t_rk45)
    rk45_x, rk45_y, rk45_z = solution_rk45.y[0][start_idx_rk45:], solution_rk45.y[1][start_idx_rk45:], solution_rk45.y[2][start_idx_rk45:]

if USE_ANALYTICAL:
    start_t_ana  = norm_time - window_duration
    start_idx_ana = np.searchsorted(t_eval_ps, start_t_ana )
    ana_x, ana_y, ana_z =solution_analytical[0][start_idx_ana:], solution_analytical[1][start_idx_ana:], solution_analytical[2][start_idx_ana:]

start_t_ps   = norm_time - window_duration
start_idx_ps   = np.searchsorted(t_eval_ps, start_t_ps)
ps_x, ps_y, ps_z = solution_ps[0][start_idx_ps:], solution_ps[1][start_idx_ps:], solution_ps[2][start_idx_ps:]

if USE_FULL_PLOT:
    # === Plot Last Few Cycles ===
    fig, ax = plt.subplots(figsize=(10, 8))
    if USE_ANALYTICAL:
        ax.plot(ana_x, ana_y, label=f"Analytical", color='Black', linestyle='-',linewidth=0.4)
    if USE_RK45:
        ax.plot(rk45_x, rk45_y, label=f"RK45", color='#E69F00', linestyle='--')
    if USE_RK4:
        ax.plot(rk4_x, rk4_y, label=f"RK4", color='#CC79A7', linestyle='-.')
    ax.plot(ps_x, ps_y, label=f"PS{orders_used.max()}", color='#009E73', linestyle=':')

    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    if USE_PLOT_TITLES: ax.set_title(f'2D Trajectory of Final {particle_type} Orbits in Constant B Field')
    ax.legend(loc="upper right")
    ax.axis('equal')
    ax.grid(True)
    plt.tight_layout()

    # === Save and Close ===
    fig.canvas.draw()  
    fig.savefig( f"{output_folder}/ConstB_{particle_type}_{KE_particle:.1e}eV_{ps_step}step_PS{orders_used.max()}_{norm_time}s_{npfloat.__name__}_2Dslice.png", dpi=600, bbox_inches="tight")
    plt.close(fig)

# ======================================
# ============= Slice of 3D ============
# ======================================
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot each trajectory segment
if USE_ANALYTICAL:
    ax.plot(ana_x, ana_y, ana_z, label=f"Analytical", color='Black', linestyle='-',linewidth=0.4)
if USE_RK45:
    ax.plot(rk45_x, rk45_y, rk45_z, label=f"RK45", color='#E69F00', linestyle='--')
if USE_RK4:
    ax.plot(rk4_x, rk4_y, rk4_z, label=f"RK4", color='#CC79A7', linestyle='-.')
ax.plot(ps_x, ps_y, ps_z, label=f"PS{orders_used.max()}", color='#009E73', linestyle=':')

ax.set_xlabel("x (m)", labelpad = 15)
ax.set_ylabel("y (m)", labelpad = 15)
ax.set_zlabel("z (m)", labelpad = 20)
ax.tick_params(axis='z', pad = 10)
if USE_PLOT_TITLES: ax.set_title(f'3D Trajectory of Final {particle_type} Orbits in Constant B Field')
ax.legend(loc="upper right")

# === Save and Close ===
fig.canvas.draw()  
plt.savefig( f"{output_folder}/ConstB_{particle_type}_{KE_particle:.1e}eV_{ps_step}step_PS{orders_used.max()}_{norm_time}s_{npfloat.__name__}_3Dslice.png", dpi=600, bbox_inches="tight")
plt.close(fig)

# ===============================================================
# =====This plots KE Error over time for many different PS Orders
# ===============================================================

# === Recompute PS solutions at various orders ===
solution_ps_4, _ = PS_constantB_adaptive(4, steps_ps, initial_pos_vel_ps, ps_step, Bfield, qoverm, tol)
solution_ps_5, _ = PS_constantB_adaptive(5, steps_ps, initial_pos_vel_ps, ps_step, Bfield, qoverm, tol)
solution_ps_6, _ = PS_constantB_adaptive(6, steps_ps, initial_pos_vel_ps, ps_step, Bfield, qoverm, tol)
solution_ps_7, _ = PS_constantB_adaptive(7, steps_ps, initial_pos_vel_ps, ps_step, Bfield, qoverm, tol)
solution_ps_10, _ = PS_constantB_adaptive(10, steps_ps, initial_pos_vel_ps, ps_step, Bfield, qoverm, tol)

# === Compute drifts ===
vx4, vy4, vz4 = extract_v(solution_ps_4)
vx5, vy5, vz5 = extract_v(solution_ps_5)
vx6, vy6, vz6 = extract_v(solution_ps_6)
vx7, vy7, vz7 = extract_v(solution_ps_7)
vx10, vy10, vz10 = extract_v(solution_ps_10)

rel_drift_ps_4  = compute_energy_drift(vx4, vy4, vz4)
rel_drift_ps_5  = compute_energy_drift(vx5, vy5, vz5)
rel_drift_ps_6  = compute_energy_drift(vx6, vy6, vz6)
rel_drift_ps_7  = compute_energy_drift(vx7, vy7, vz7)
rel_drift_ps_10 = compute_energy_drift(vx10, vy10, vz10)

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


# plot
f64 = lambda a: np.asarray(a, dtype=np.float64)
fig, ax = plt.subplots(figsize=(10, 5))
if USE_RK45:
    lnrk45, = ax.semilogy(f64(t_eval_rk45[1:]), np.abs(f64(rel_drift_rk45[1:])), linestyle='--', color='#E69F00')
if USE_RK4:
    lnrk4,  = ax.semilogy(f64(t_eval_rk4[1:]),  np.abs(f64(rel_drift_rk4[1:])),  linestyle='-.', color='#CC79A7')
lnps4,  = ax.semilogy(f64(t_eval_ps[1:]),  np.abs(f64(rel_drift_ps_4[1:])),  linestyle=':',  color='crimson')
lnps5,  = ax.semilogy(f64(t_eval_ps[1:]),  np.abs(f64(rel_drift_ps_5[1:])),  linestyle='-.', color='#0072B2')
lnps6,  = ax.semilogy(f64(t_eval_ps[1:]),  np.abs(f64(rel_drift_ps_6[1:])),  linestyle=':',  color='#56B4E9')
lnps7,  = ax.semilogy(f64(t_eval_ps[1:]),  np.abs(f64(rel_drift_ps_7[1:])),  linestyle='--', color='#D55E00')
lnps10, = ax.semilogy(f64(t_eval_ps[1:]),  np.abs(f64(rel_drift_ps_10[1:])), linestyle='-.', color='#999999')
lnps,   = ax.semilogy(f64(t_eval_ps[1:]),  np.abs(f64(rel_drift_ps[1:])),     linestyle=':',  color='#009E73')

# axis formatting
ax.margins(x=0.01)
ax.set_yscale('log')
ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=100))
ax.yaxis.set_major_formatter(LogFormatterSciNotation(base=10.0))
ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=[]))
ax.yaxis.set_minor_formatter(NullFormatter())
ax.grid(True, which='major', linestyle='--', linewidth=0.7)
ax.yaxis.set_major_formatter(FuncFormatter(sparse_labels))

# Remove top and right borders
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.set_xlabel(r"$t/\tau_0$")
ax.set_ylabel(r"$|\Delta E|/E_0$")
if USE_PLOT_TITLES: ax.set_title(f'{particle_type} Relative Kinetic Energy Error in Constant B Field')

# room for labels + apply collision-free labels (no legend)
fig.subplots_adjust(right=0.9)
fig.canvas.draw()
ax_pos = ax.get_position()  # Bbox in figure coords

# method labels at end points
x_fig_label = ax_pos.x1  # a small gap to the right of axes
endpoints = []
if USE_RK45:
    endpoints.append(
        (t_eval_rk45[-1], np.abs(rel_drift_rk45[-1]), "RK45", lnrk45.get_color())
    )
if USE_RK4:
    endpoints.append(
        (t_eval_rk4[-1], np.abs(rel_drift_rk4[-1]), "RK4", lnrk4.get_color())
    )

ps_endpoints = [
    (t_eval_ps[-1], np.abs(rel_drift_ps_4[-1]),  "PS4",  lnps4.get_color()),
    (t_eval_ps[-1], np.abs(rel_drift_ps_5[-1]),  "PS5",  lnps5.get_color()),
    (t_eval_ps[-1], np.abs(rel_drift_ps_6[-1]),  "PS6",  lnps6.get_color()),
    (t_eval_ps[-1], np.abs(rel_drift_ps_7[-1]),  "PS7",  lnps7.get_color()),
    (t_eval_ps[-1], np.abs(rel_drift_ps_10[-1]), "PS10", lnps10.get_color()),
    (t_eval_ps[-1], np.abs(rel_drift_ps[-1]),    f"PS{orders_used.max()}", lnps.get_color()),
]
endpoints.extend(ps_endpoints)

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
fig.savefig( f"{output_folder}/ConstB_{particle_type}_{KE_particle:.1e}eV_{ps_step}step_PS{orders_used.max()}_{norm_time}s_{npfloat.__name__}_KEerror_manyPS.png", dpi=600, bbox_inches="tight")
plt.close(fig)


# =================================================================
# ================Plotting trajectory error over time =============
# =================================================================
if USE_ANALYTICAL:
    t_common = t_eval_ps
    eps = 1e-15

    solution_analytical_ps = analytical_constantB(
        t_eval_ps, initial_pos_vel, Bfield, qoverm)
    abs_err_ps, rel_err_ps = vector_error(solution_ps, solution_analytical_ps)

    if USE_RK4:
        solution_analytical_rk4 = analytical_constantB(
            t_eval_rk4, initial_pos_vel, Bfield, qoverm)
        abs_err_rk4, rel_err_rk4 = vector_error(solution_rk4, solution_analytical_rk4)
        rel_err_rk4_interp  = interp_to_grid(t_eval_rk4,  rel_err_rk4,  t_common)

    if USE_RK45:
        solution_analytical_rk45 = analytical_constantB(
            t_eval_rk45, initial_pos_vel, Bfield, qoverm)
        abs_err_rk45, rel_err_rk45 = vector_error(solution_rk45, solution_analytical_rk45, is_rk45=True)
        rel_err_rk45_interp = interp_to_grid(t_eval_rk45, rel_err_rk45, t_common)

    # choose a reference grid (e.g. PS)
    t_common = t_eval_ps
    rel_err_ps_interp   = rel_err_ps  # already on t_common

    # ---- Plot ----
    fig, ax = plt.subplots(figsize=(10, 5))

    if USE_RK45:
        linerk45, = ax.semilogy(t_common, rel_err_rk45_interp, label='RK45', linestyle='--', color='#E69F00')
    if USE_RK4:
        linerk4, = ax.semilogy(t_common, rel_err_rk4_interp,  label='RK4',  linestyle='-.', color='#CC79A7')
    lineps, = ax.semilogy(t_common, rel_err_ps_interp,   label=f"PS{orders_used.max()}", linestyle=':', color='#009E73')


    ax.margins(x=0.01)

    ax.set_yscale('log')
    ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=100))  
    ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=[]))       
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(FuncFormatter(sparse_labels))       

    ax.grid(False, which='both')
    ax.grid(True, which='major', linestyle='--', linewidth=0.7)

    # Remove top and right borders
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.set_xlabel(r't/$\tau_0$')
    ax.set_ylabel(r'$|\Delta \mathbf{r}|/r$')
    if USE_PLOT_TITLES: ax.set_title(f'{particle_type} Relative Trajector Energy Error in Constant B Field')

    fig.subplots_adjust(right=0.9)  
    fig.canvas.draw()

    # method endpoint labels
    ax_pos = ax.get_position()  
    x_fig_label = ax_pos.x1  
    endpoints = []

    if USE_RK45:
        endpoints.append((t_eval_rk45[-1], np.abs(rel_err_rk45_interp[-1]), "RK45", linerk45.get_color()))
    if USE_RK4:
        endpoints.append((t_eval_rk4[-1], np.abs(rel_err_rk4_interp[-1]), "RK4", linerk4.get_color()))

    endpoints.append((t_eval_ps[-1], np.abs(rel_err_ps_interp[-1]), f"PS{orders_used.max()}", lineps.get_color()))

    for x, y, label, color in endpoints:
        _, fy = data_to_fig(x, y, ax, fig)
        fy = min(max(fy, ax_pos.y0), ax_pos.y1)
        fig.text(x_fig_label, fy, label, color=color, va='center', ha='left', fontsize=10)


    labels = []
    for x, y, label, color in endpoints:
        _, fy = data_to_fig(x, y, ax, fig)
        fy = min(max(fy, ax_pos.y0), ax_pos.y1)
        labels.append([fy, label, color])

    labels.sort(key=lambda v: v[0])

    min_gap = 0.02  
    for i in range(1, len(labels)):
        if labels[i][0] - labels[i-1][0] < min_gap:
            labels[i][0] = labels[i-1][0] + min_gap

    for i in range(len(labels)-2, -1, -1):
        if labels[i+1][0] - labels[i][0] < min_gap:
            labels[i][0] = labels[i+1][0] - min_gap

    for fy, label, color in labels:
        fig.text(x_fig_label, fy, label, color=color,
                va='center', ha='left', fontsize=10)


    # === Save and Close ===
    fig.canvas.draw()  
    fig.savefig( f"{output_folder}/ConstB_{particle_type}_{KE_particle:.1e}eV_{ps_step}step_PS{orders_used.max()}_{norm_time}s_{npfloat.__name__}_TrajError.png", dpi=600, bbox_inches="tight")
    plt.close(fig)

# ============================================
# ======= Write Summary Output to File =======
# ============================================

output_filename = f"{output_folder}/ConstB_{particle_type}_{KE_particle:.1e}eV_{ps_step}step_PS{orders_used.max()}_{norm_time}s_{npfloat.__name__}_SimSummary.txt"

with open(output_filename, "w") as f:
    f.write("=== Simulation Summary ===\n")
    f.write(f"Initial Conditions:\n")
    f.write(f"  Particle      = {particle_type}\n")
    f.write(f"  Energy        = {KE_particle} eV\n")
    f.write(f"  mass          = {mass} kg\n")
    f.write(f"  pitch_deg     = {pitch_deg}\n")
    f.write(f"  phi_deg       = {phi_deg}\n")
    f.write(f"  tau_time      = {tau_time}\n")
    f.write(f"  v_tau         = {v_tau}\n")
    f.write(f"  x_initial     = {x_initial} m\n")
    f.write(f"  y_initial     = {y_initial} m\n")
    f.write(f"  z_initial     = {z_initial} m\n")
    f.write(f"  vx_initial    = {vx_initial} m/s\n")
    f.write(f"  vy_initial    = {vy_initial} m/s\n")
    f.write(f"  vz_initial    = {vz_initial} m/s\n\n")
    f.write(f"  Bfield        = {Bfield} T\n")
    f.write(f"  float type    = {npfloat.__name__}\n\n")

    f.write(f"Timing Summary:\n")
    f.write(f"  PS       = {end_time_ps - start_time_ps:.6f} s\n")
    if USE_RK4:
        f.write(f"  RK4      = {end_time_rk4 - start_time_rk4:.6f} s\n")
    if USE_RK45:
        f.write(f"  RK45     = {end_time_rk45 - start_time_rk45:.6f} s\n")
    if USE_ANALYTICAL:
        f.write(f"  Analytical     = {end_time_analytical - start_time_analytical:.6f} s\n")    
    f.write(f"  norm_time     = {norm_time}\n")
    f.write(f"  Phsyical time = {physical_time} s\n")
    f.write(f"  rk4_step      = {rk4_step}\n")
    f.write(f"  ps_step       = {ps_step}\n")
    f.write(f"  PS Orders     = max={orders_used.max()}, mean={orders_used.mean():.1f}\n")
