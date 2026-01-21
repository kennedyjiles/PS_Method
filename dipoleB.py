import numpy as np
import builtins
import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
import test_particles.dipoleB_testparticles as tp
builtins.npfloat = np.float128 if tp.USE_FLOAT128 else np.float64
from test_particles.dipoleB_testparticles import *
import pandas as pd 
from datetime import datetime
import os, time, sys, tracemalloc, logging, h5py, gc
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from matplotlib.ticker import LogLocator, LogFormatterSciNotation, NullFormatter, FuncFormatter
from functions.functions_library_universal_chunk import rk4_fixed_step, plt_config, sparse_labels, data_to_fig
from functions.functions_library_dipole import PS_dipoleB, lorentz_force_dipole, compute_mu_ps, compute_mu_rk, vector_potential_dipole, rkgl4_hamiltonian, hamiltonian_rhs, summarize, slice_solution, append_results_h5, compute_energy_ps_chunked
from functions.functions_library_dipole import mirror_times_from_PS, bounce_summary, drift_period_from_PS, get_run_params, h5_path_for, save_results_h5, load_results_h5, summarize_error, run_ps_streaming_with_decimation
from logger_util import setup_logger
logger = setup_logger("dipole_logger", "dipole_chunk.log", level=logging.DEBUG)


DEBUG = False

if DEBUG: tracemalloc.start()

run = "demo"   # key options: "demo", "paper1", "paper2", "paper3", unless a new input is made. Demo mode is a quick test run. Paper modes can take upwards of half an hour. 

# Allow command-line override
if len(sys.argv) > 1:
    run = sys.argv[1]
    print(f"Run mode set from command line: {run}\n")
else:
    print(f"Using default run mode: {run}\n")

globals().update(load_params(run)) # fix later and turn into dictionary 

# === Misc Odds and Ends ===  
plt_config(scale=1)                   # config file for setting plot sizes and fonts (from Dr. W)
os.makedirs(run_storage, exist_ok=True)
os.makedirs(output_folder, exist_ok=True)

plt.ioff()              # turn off interactive mode for plots
if USE_FLOAT128: USE_RKG = False

# for file/plot naming
if mass_si == m_e: particle_type = "Electron"
elif mass_si == m_p: particle_type = "Proton"
else: particle_type = "Particle"

qoverm = npfloat(-1) if mass_si == m_e else npfloat(1)

# === Misc Conversions  ===
KE_joules = KE_particle * evtoj                     # converting KE from eV to Joules
gamma = 1.0 + KE_joules / (mass_si * spdlight**2)   # Lorentz factor
mass = gamma * mass_si                              # Relativistic mass used for magnetic moment calculations
v_si = spdlight * np.sqrt(1.0 - 1.0 / gamma**2)     # m/s
tau_time = gamma * mass_si / (abs(q_e) * abs(B_0))  # this is tau0 from paper 
v_tau = v_si * tau_time / RE                        # dimensionless velocity
physical_time = norm_time * abs(tau_time)           # actual physical time, t; normalized time =t/tau_time
window_duration = window_time/tau_time              # converting to dimensionless time

tol = npfloat(tol) * tau_time                       # convert tolerance to normalized units    

# === Velocity Config based on INput Angles ===
pitch_rad = npfloat(np.radians(pitch_deg))              # degrees to radians, pitch 
phi_rad = npfloat(np.radians(phi_deg))                  # degrees to radians, phi 
v_par = npfloat(v_tau) * npfloat(np.cos(pitch_rad))     # parallel velocity component
v_perp = npfloat(v_tau) * npfloat(np.sin(pitch_rad))    # perpendicular velocity component 

vx_initial = npfloat(v_perp * np.cos(phi_rad))          
vy_initial = npfloat(v_perp * np.sin(phi_rad))
vz_initial = npfloat(v_par)

# ===  cleaning small trig values to zero === 
if abs(vx_initial) < (1.0 * np.finfo(npfloat).eps): vx_initial = npfloat(0.0) 
if abs(vy_initial) < (1.0 * np.finfo(npfloat).eps): vy_initial = npfloat(0.0)
if abs(vz_initial) < (1.0 * np.finfo(npfloat).eps): vz_initial = npfloat(0.0)

gyro_radius_si = (gamma * mass_si * v_si * np.sin(pitch_rad) / (np.abs(q_e) * (B_0 / x_initial**3)))
gyro_radius_REi=float(gyro_radius_si/RE)

# --- these should be identical, kept seperate in case I decide to scale one method at a later point---
initial_pos_vel = np.array([x_initial, y_initial, z_initial, vx_initial, vy_initial, vz_initial], dtype=npfloat)  
initial_pos_vel_ps = np.array([x_initial, y_initial, z_initial, vx_initial, vy_initial, vz_initial], dtype=npfloat) 

if DEBUG: 
    logger.info("Starting chunked dipole run.")
    logger.debug(f"Initial velocity: {vx_initial}, {vy_initial}, {vz_initial}")
    logger.debug(f"Initial position: {x_initial}, {y_initial}, {z_initial}")
    logger.debug(f"Initial gyroradius: {gyro_radius_REi}")

# --- Initial invariants for E0 and mu0 for h5 file ---
vx0, vy0, vz0 = initial_pos_vel_ps[3:6]
E0_ps = npfloat(0.5) * (vx0*vx0 + vy0*vy0 + vz0*vz0)
y0_ps = np.zeros((17, 1), dtype=npfloat)
y0_ps[0:6, 0] = initial_pos_vel_ps
x0, y0, z0 = initial_pos_vel_ps[0:3]
r2 = x0*x0 + y0*y0 + z0*z0
r5inv = r2**(-2.5)
y0_ps[14, 0] = -3 * x0 * z0 * r5inv
y0_ps[15, 0] = -3 * y0 * z0 * r5inv
y0_ps[16, 0] = -(3*z0*z0 - r2) * r5inv
mu0_ps = compute_mu_ps(y0_ps, mass)[0]

# === Build parameter tracer & check cache ===
"""
this is scanning the files already stored to see if we already have the data,
beware that these files can be GB size for dipole
"""
params = get_run_params(USE_RK45, USE_RK4, USE_RKG,    # parameters it is scanning
                   mass_si, q_e, B_0, gamma,
                   x_initial, y_initial, z_initial,
                   pitch_deg, phi_deg,
                   norm_time, ps_step, rk4_step, rkg_step,
                   PS_order, tol, qoverm, rtol_rk45, atol_rk45)
cache_path = h5_path_for(params, run_storage)

if os.path.exists(cache_path) and READ_DATA:
    print(f"Found existing results: {os.path.basename(cache_path)} — loading.\n")

    with h5py.File(cache_path, "r") as cached:

        meta_group = cached.get("meta", None)
        timing = {}
        if meta_group is not None:
            timing = dict(meta_group.attrs)
            norm_time = meta_group.attrs.get("norm_time")
            physical_time = meta_group.attrs.get("physical_time")
            particle_label = meta_group.attrs.get("particle_label")
            percent_c = meta_group.attrs.get("percent_c")
        stem = os.path.splitext(os.path.basename(cache_path))[0]
        timing = {}

        meta_group = cached.get("meta", None)
        if meta_group is not None:
            timing_keys = {
                "ps": "timing_ps",
                "rk4": "timing_rk4",
                "rk45": "timing_rk45",
                "rkg": "timing_rkg"
            }

            for short_key, attr_key in timing_keys.items():
                if attr_key in meta_group.attrs:
                    timing[short_key] = meta_group.attrs[attr_key]


        if USE_PS and "ps" in cached:
            ps_group = cached["ps"]

            # Attributes
            E0_ps = ps_group.attrs.get("E0")
            mu0_ps = ps_group.attrs.get("mu0")
            PS_decimate = ps_group.attrs.get("decimate", 1)
            ps_step = ps_group.attrs.get("dt")
            steps_ps = ps_group.attrs.get("steps")
            is_streaming = bool(ps_group.attrs.get("streaming", False))
            PS_CHUNKING = is_streaming
            ps_order_label = ps_group.attrs.get("max_ps", None)

            # Datasets- DON'T LOAD BIG ASS FILES, slice up later
            if PS_CHUNKING:
                solution_ps = None
                orders_used = None
            else:
                # Load full dataset into memory
                solution_ps = ps_group["y"][()]
                
                # Try to load orders safely
                if "orders" in ps_group:
                    orders_used = ps_group["orders"][()]
                else:
                    orders_used = None  # or set to np.full(...) if needed


        if USE_RK4 and "rk4" in cached:
            rk4_group = cached["rk4"]
            rk4_step = rk4_group.attrs.get("dt")
            steps_rk4 = rk4_group.attrs.get("steps")
            solution_rk4 = rk4_group["y"][()] 

        if USE_RK45 and "rk45" in cached:
            rk45_group = cached["rk45"]
            class _Obj: pass
            solution_rk45 = _Obj()
            solution_rk45.t = rk45_group["t"][()] 
            solution_rk45.y = rk45_group["y"][()] 
            solution_rk45.sol = None  # placeholder

        if USE_RKG and "rkg" in cached:
            rkg_group = cached["rkg"]
            rkg_step = rkg_group.attrs.get("dt")
            steps_rkg = rkg_group.attrs.get("steps")
            solution_rkg = rkg_group["y"][()] 
else:
    print("No matching file or 'Read Data' skipped. Running solvers...\n")

    # Common grid size (used by RK45, and PS if enabled)
    steps_ps = int(norm_time / ps_step)
    # ====== Run PS ======
    max_ps = None
    if USE_PS:
        start_time_ps = time.time()

        if PS_CHUNKING:
            max_ps, elapsed_ps = run_ps_streaming_with_decimation(
                initial_pos_vel_ps=initial_pos_vel_ps,
                steps_ps=steps_ps,
                ps_step=ps_step,
                PS_order=PS_order,
                tol=tol,
                qoverm=qoverm,
                E0_ps=E0_ps,
                mu0_ps=mu0_ps,
                cache_path=cache_path,
                write_data=True,
                chunk_steps=PS_chunk_steps,
                decimate=PS_decimate,
            )
            end_time_ps = start_time_ps + elapsed_ps
            solution_ps = None
            orders_used = None
        else:
            solution_ps, orders_used = PS_dipoleB(
                PS_order, steps_ps, initial_pos_vel_ps, tol, qoverm, ps_step
            )
            end_time_ps = time.time()

    # ====== Run RK45 ======
    if USE_RK45:
        start_time_rk45 = time.time()
        t_common = ps_step * np.arange(steps_ps, dtype=npfloat)
        solution_rk45 = solve_ivp(
            lorentz_force_dipole,
            (0.0, norm_time),
            initial_pos_vel,
            method="RK45",
            args=(qoverm,),
            t_eval=t_common,
            rtol=rtol_rk45,
            atol=atol_rk45,)
        end_time_rk45 = time.time()

    # ====== Run RK4 ======
    if USE_RK4:
        steps_rk4 = int(norm_time / rk4_step)
        start_time_rk4 = time.time()
        solution_rk4 = rk4_fixed_step(
            lorentz_force_dipole,
            initial_pos_vel,
            rk4_step,
            steps_rk4,
            args=(qoverm,),)
        end_time_rk4 = time.time()

    # ====== Run RKG ======
    if USE_RKG:
        # === Symplectic Implementations =====
        r0 = np.array([x_initial, y_initial, z_initial], dtype=npfloat)   # already normalized RE units
        v_tau_vec = np.array([vx_initial, vy_initial, vz_initial], dtype=npfloat)

        A0 = vector_potential_dipole(r0)
        p0 = v_tau_vec + A0
        y0 = np.concatenate((r0, p0))   # for Hamiltonian in RKG
        # y0 = np.concatenate((r0, v_tau_vec))  # for Lorentz force in RKG, used as a sanity check

        steps_rkg = int(norm_time / rkg_step)
        steps_rkg = max(1, steps_rkg)

        start_time_rkg = time.time()
        solution_rkg = rkgl4_hamiltonian(
            hamiltonian_rhs,
            y0,
            rkg_step,
            steps_rkg,
            args=(qoverm,),
        )
        end_time_rkg = time.time()

    # Preparing a results dictionary for saving so future heather doesn't have to keep waiting
    results = {
        "ps": None,
        "rk4": None,
        "rk45": None,
        "rkg": None,
        "meta": {
            "timing": {},
            "physical_time": float(physical_time),
            "norm_time": float(norm_time),
            "percent_c": float(v_si/spdlight),
            "particle_label": (
                f"{KE_particle:.1e} eV electron" if mass_si == m_e else
                f"{KE_particle:.1e} eV proton" if mass_si == m_p else
                "manual"
            ),
        }
    }

    if USE_PS:
        results["meta"]["timing"]["ps"] = end_time_ps - start_time_ps
        if not PS_CHUNKING:
            results["ps"] = {
                "y": solution_ps,
                "orders": orders_used,
                "dt": float(ps_step),
                "steps": int(steps_ps),
                "t0": 0.0,
                "decimate": int(PS_decimate),
                "E0": float(E0_ps),
                "mu0": float(mu0_ps),
                "max_ps": int(orders_used.max()),
            }
        else:
            results["ps"] = {
                "dt": float(ps_step),
                "steps": int(steps_ps),
                "t0": 0.0,
                "decimate": int(PS_decimate),
                "E0": float(E0_ps),
                "mu0": float(mu0_ps),
                "max_ps": int(max_ps) if max_ps is not None else None,
                "streaming": True,
            }

    if USE_RK4:
        results["rk4"] = {"y": solution_rk4, "dt": float(rk4_step), "steps": int(steps_rk4), "t0": 0.0}
        results["meta"]["timing"]["rk4"] = end_time_rk4 - start_time_rk4

    if USE_RK45:
        results["rk45"] = {"t": solution_rk45.t, "y": solution_rk45.y}
        results["meta"]["timing"]["rk45"] = end_time_rk45 - start_time_rk45

    if USE_RKG:
        results["rkg"] = {"y": solution_rkg, "dt": float(rkg_step), "steps": int(steps_rkg), "t0": 0.0}
        results["meta"]["timing"]["rkg"] = end_time_rkg - start_time_rkg

    # ====== Save ======
    if WRITE_DATA:
        if USE_PS and PS_CHUNKING:
            append_results_h5(cache_path, results, params)
            print(f"Updated streamed file → {os.path.basename(cache_path)}")
        else:
            save_results_h5(cache_path, params, results)
            print(f"Saved results → {os.path.basename(cache_path)}")

    timing = results["meta"]["timing"]
    stem = os.path.splitext(os.path.basename(cache_path))[0]

if DEBUG: 
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    logger.info(f"Peak memory usage for load/write h5: {peak / 1024**2:.2f} MB\n")

# ===============================
# Build RK45 solution on PS grid 
# ===============================
"""
this is building RK45 time base for points we want on PS grid. Not meant for long runs
as this can be a memory hog but rk45 is not great on long runs anyways
"""
if USE_RK45 and not USE_PS:
    raise RuntimeError("RK45 requires USE_PS=True in this workflow it builds a grid to match PS.")

if USE_RK45:
    t_common = ps_step * np.arange(steps_ps, dtype=npfloat)

    if hasattr(solution_rk45, "sol") and solution_rk45.sol is not None:
        y_rk45_common = solution_rk45.sol(t_common)
    else:
        t_src = solution_rk45.t
        y_src = solution_rk45.y

        y_rk45_common = np.empty((y_src.shape[0], len(t_common)), dtype=y_src.dtype)
        for i in range(y_src.shape[0]):
            y_rk45_common[i] = np.interp(t_common, t_src, y_src[i])


# =====================================================
# ============= Data Set Access for Stream ============
# =====================================================
tracemalloc.start()

ps_order_label = None

if USE_PS:
    if not PS_CHUNKING:
        n_ps = steps_ps
        stride = max(1, n_ps // MAX_PLOT_POINTS)
        ps_order_label = int(orders_used.max())
    else:
        with h5py.File(cache_path, "r") as ps_h5:
            ps_grp = ps_h5["ps"]
            n_ps = steps_ps
            stride = max(1, n_ps // MAX_PLOT_POINTS)
            ps_order_label = int(ps_grp.attrs["max_ps"])


if USE_PS and USE_FULL_PLOT:
    # PS in RAM
    if not PS_CHUNKING:
        n_ps = solution_ps.shape[1]
        stride = max(1, n_ps // MAX_PLOT_POINTS)
        ps_order_label = int(orders_used.max())

        x_ps_plot = solution_ps[0, ::stride]
        y_ps_plot = solution_ps[1, ::stride]
        z_ps_plot = solution_ps[2, ::stride]

    # PS pulling from HDF5
    else:
        with h5py.File(cache_path, "r") as ps_h5:
            ps_grp = ps_h5["ps"]
            ps_y_h5 = ps_grp["y"]

            x_ps_plot = ps_y_h5[0, ::stride]
            y_ps_plot = ps_y_h5[1, ::stride]
            z_ps_plot = ps_y_h5[2, ::stride]

            ps_order_label = int(ps_grp.attrs["max_ps"])
current, peak = tracemalloc.get_traced_memory()
tracemalloc.stop()
logger.debug(f"Data access for plottings: {peak / 1024**2:.2f} MB\n")

# === paper specific adjustments for aeshetic purposes ====
if run == "paper1": USE_RK4 = False 
if run == "paper2": USE_RKG = False

# === Timing Summary ===
print(f"Particle        : {KE_particle:.1e} eV {particle_type}")
if USE_RK45 and "rk45" in timing:
    print(f"Run Time RK45   : {timing['rk45']:.2f} s")
if USE_RK4 and "rk4" in timing:
    print(f"Run Time RK4    : {timing['rk4']:.2f} s")
if USE_RKG and "rkg" in timing:
    print(f"Run Time RKG    : {timing['rkg']:.2f} s")
if USE_PS and "ps" in timing:
    print(f"Run Time PS     : {timing['ps']:.2f} s")

print(f"Norm Time       : {norm_time:.2e} ")
print(f"Physical Time   : {physical_time:.2e} s")
if USE_PS and not PS_CHUNKING:
    print(f"PS Orders       : max={orders_used.max()}, mean={orders_used.mean():.1f}")
else: print(f"PS Orders       : max={ps_order_label}")
print(f"% of c          : {v_si/spdlight:.8f}")

if DEBUG: 
    logger.debug(f"Norm Time: {norm_time:.2e} MB")
    logger.debug(f"Physical Time   : {physical_time:.2e} s")
    logger.debug(f"ps_step: {ps_step}, norm_time: {norm_time}, steps_ps: {steps_ps}")
    # logger.debug(f"t_common[0]: {t_common[0]}, t_common[-1]: {t_common[-1]}")



# =====================================================
# ============== Full 2D Trajectory Plot ==============
# =====================================================
plotbounds = x_initial + 1.1 

if USE_FULL_PLOT:
# === Plot Trajectories ===
    fig, ax = plt.subplots(figsize=(10, 8))

    if USE_RK45:
        ax.plot(solution_rk45.y[0], solution_rk45.y[1], label='RK45', color='#E69F00', linestyle='--')
    if USE_RK4:
        ax.plot(solution_rk4[0], solution_rk4[1], label='RK4', alpha=0.8, color='#CC79A7', linestyle='-.')
    if USE_RKG:
        ax.plot(solution_rkg[:, 0], solution_rkg[:, 1], label='RKG', alpha=0.8, color='#CC0000', linestyle='-.')
    if USE_PS:
        ax.plot( x_ps_plot, y_ps_plot, label=f"PS{ps_order_label}", alpha=0.7, color="#009E73", linestyle=":")

    # === Formatting ===
    ax.set_xlabel(r"X")
    ax.set_ylabel(r"Y")
    ax.ticklabel_format(style='plain', useOffset=False, axis='both')
    if USE_PLOT_TITLES: ax.set_title(f"2D {particle_type} Trajectory in Dipole B Field")

    ax.legend(loc="upper right")
    ax.set_xlim(-plotbounds, plotbounds)
    ax.set_ylim(-plotbounds, plotbounds)
    ax.set_aspect('equal', adjustable='box')
    # ax.axis('equal')
    ax.grid(True)

    # === Save and Close ===
    fig.canvas.draw()   
    if USE_PS:
        fig.savefig( f"{output_folder}/{stem}_DipoleB_chunk_{particle_type}_{KE_particle:.1e}eV_{ps_step}step_PS{ps_order_label}_pitch{pitch_deg}_phi{phi_deg}_{norm_time:.2e}s_{npfloat.__name__}_2D.png", dpi=600, bbox_inches="tight")
    else:
        fig.savefig( f"{output_folder}/{stem}_DipoleB_chunk_{particle_type}_{KE_particle:.1e}eV_pitch{pitch_deg}_phi{phi_deg}_{norm_time:.2e}s_{npfloat.__name__}_2D.png", dpi=600, bbox_inches="tight")
    plt.close(fig)  

# =====================================================
# ============== Full 3D Trajectory Plot ==============
# =====================================================
if USE_FULL_PLOT:
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # === Plot Trajectories ===
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

    # === Labels and Legend ===
    ax.set_xlabel(r'X')
    ax.set_ylabel(r'Y')
    ax.set_zlabel(r'Z')
    if USE_PLOT_TITLES: ax.set_title(f"3D {particle_type} Trajectory in Dipole B Field")
    ax.legend(loc="upper right")

    # === Save and Close ===
    fig.canvas.draw()   
    if USE_PS:
        fig.savefig( f"{output_folder}/{stem}_DipoleB_chunk_{particle_type}_{KE_particle:.1e}eV_{ps_step}step_PS{ps_order_label}_pitch{pitch_deg}_phi{phi_deg}_{norm_time:.2e}s_{npfloat.__name__}_3D.png", dpi=600, bbox_inches="tight")
    else:
        fig.savefig( f"{output_folder}/{stem}_DipoleB_chunk_{particle_type}_{KE_particle:.1e}eV_pitch{pitch_deg}_phi{phi_deg}_{norm_time:.2e}s_{npfloat.__name__}_3D.png", dpi=600, bbox_inches="tight")
    plt.close(fig) 

# ==========================================
# ================ Window ==================
# ==========================================
if DEBUG: tracemalloc.start()

if slice_mode == "first":
    t_start = 0.0
    t_end   = min(norm_time, window_duration)
elif slice_mode == "last":
    t_end   = norm_time
    t_start = max(0.0, norm_time - window_duration)
else:
    raise ValueError("slice_mode must be 'first' or 'last'")

# ==========================================
# =========== PS Window Load ===============
# ==========================================

if USE_PS:

    # --- map physical time → PS indices ---
    i0_phys = int(np.floor(t_start / ps_step))
    i1_phys = int(np.floor(t_end   / ps_step))
    i0_phys = max(0, i0_phys)
    i1_phys = min(i1_phys, steps_ps)

    if i1_phys < i0_phys:
        raise RuntimeError("Empty PS slice window")

    ps_store_stride = PS_decimate if (PS_CHUNKING and PS_decimate > 1) else 1

    # --- map physical → stored indices ---
    j0 = int(np.ceil(i0_phys / ps_store_stride))
    j1 = int(np.floor(i1_phys / ps_store_stride))

    if j1 < j0:
        raise RuntimeError("Empty PS stored slice window")

    # ======================================
    # Load ONLY the window
    # ======================================

    if not PS_CHUNKING:
        # PS in RAM
        y_win = solution_ps[:, j0:j1+1]
        ps_order_label = int(np.max(orders_used))

    else:
        # PS streaming from HDF5
        with h5py.File(cache_path, "r") as ps_h5:
            ps_grp = ps_h5["ps"]
            ps_y   = ps_grp["y"]

            n_store = ps_y.shape[1]
            j0 = max(0, min(j0, n_store - 1))
            j1 = max(0, min(j1, n_store - 1))


            if j1 < j0:
                raise RuntimeError("Empty PS stored slice")

            y_win = ps_y[:, j0:j1+1]
            ps_order_label = int(ps_grp.attrs["max_ps"])

    plot_stride = max(1, y_win.shape[1] // MAX_PLOT_POINTS)

    ps_x_slice = y_win[0, ::plot_stride]
    ps_y_slice = y_win[1, ::plot_stride]
    ps_z_slice = y_win[2, ::plot_stride]

if USE_RK4:
    t_rk4 = rk4_step * np.arange(solution_rk4.shape[1], dtype=npfloat)
    rk4_x_slice, rk4_y_slice, rk4_z_slice = slice_solution(
        t_rk4,solution_rk4, window_duration, norm_time, mode=slice_mode )[:3]

if USE_RKG:
    t_rkg = rkg_step * np.arange(solution_rkg.shape[0], dtype=npfloat)
    rkg_x_slice, rkg_y_slice, rkg_z_slice = slice_solution(
        t_rkg, solution_rkg.T, window_duration, norm_time, mode=slice_mode )[:3]

if USE_RK45:
    t_rk45 = ps_step * np.arange(y_rk45_common.shape[1], dtype=npfloat)
    rk45_x_slice, rk45_y_slice, rk45_z_slice = slice_solution( 
        t_rk45, y_rk45_common, window_duration, norm_time, mode=slice_mode )[:3]


if DEBUG: 
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    logger.info(f"Peak memory usage for slice analysis: {peak / 1024**2:.2f} MB")
    x_end_time = j1 * ps_store_stride * ps_step
    logger.debug(f"[Window] mode={slice_mode}, t_start={t_start:.6e}, t_end={t_end:.6e}")
    logger.debug(f"[PS] x_start={y_win[0,0]:.6e}, x_end={y_win[0,-1]:.6e}")

# =====================================================
# ================ 2D Trajectory Slice ================
# =====================================================
if USE_FULL_PLOT:
    # === Plot Last Few Cycles ===
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
    if USE_PLOT_TITLES: ax.set_title(f"2D Trajectory of Slice {particle_type} Orbits in Dipole B Field")
    # ax.set_xlim(-plotbounds, plotbounds)
    # ax.set_ylim(-plotbounds, plotbounds)
    # ax.set_aspect('equal', adjustable='box')
    ax.axis('equal')
    ax.legend(loc="upper right")
    ax.grid(True)


    # === Save and Close ===
    fig.canvas.draw()   
    if USE_PS:
        fig.savefig( f"{output_folder}/{stem}_DipoleB_chunk_chunk_{particle_type}_{KE_particle:.1e}eV_{ps_step}step_PS{ps_order_label}_pitch{pitch_deg}_phi{phi_deg}_{norm_time:.2e}s_{npfloat.__name__}_2Dslice.png", dpi=600, bbox_inches="tight")
    else:
        fig.savefig( f"{output_folder}/{stem}_DipoleB_chunk_chunk_{particle_type}_{KE_particle:.1e}eV_pitch{pitch_deg}_phi{phi_deg}_{norm_time:.2e}s_{npfloat.__name__}_2Dslice.png", dpi=600, bbox_inches="tight")
    plt.close(fig) 


# =====================================================
# ================ 3D Trajectory Slice ================
# =====================================================
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
if USE_PLOT_TITLES: ax.set_title(f'3D Trajectory Slice of {particle_type} Orbits in Dipole B Field')
ax.legend(loc="upper right")

# === Save and Close ===
fig.canvas.draw()   
if USE_PS:
    fig.savefig( f"{output_folder}/{stem}_DipoleB_chunk_{particle_type}_{KE_particle:.1e}eV_{ps_step}step_PS{ps_order_label}_pitch{pitch_deg}_phi{phi_deg}_{norm_time:.2e}s_{npfloat.__name__}_3Dslice.png", dpi=600, bbox_inches="tight")
else:
    fig.savefig( f"{output_folder}/{stem}_DipoleB_chunk_{particle_type}_{KE_particle:.1e}eV_pitch{pitch_deg}_phi{phi_deg}_{norm_time:.2e}s_{npfloat.__name__}_3Dslice.png", dpi=600, bbox_inches="tight")
plt.close(fig)  

# =====================================================
# ============== KE Relative Error Plot ===============
# =====================================================

if DEBUG:
    tracemalloc.start()

equatorial_r3 = x_initial**3
time_factor = 1.0 / (2.0 * np.pi * equatorial_r3)  # to convert plots to gyroperiods
fig, ax = plt.subplots(figsize=(10, 5))

energy_stride = max(1, n_ps // MAX_PLOT_POINTS)
# energy_stride=1 

if USE_EXTERNAL_H5_ps:
    external = load_results_h5(external_h5_ps)
    ext_ps = external["ps"]

    y_ext = ext_ps["y"]
    if "t" in ext_ps and ext_ps["t"] is not None:
        t_ext = np.asarray(ext_ps["t"])

    elif "dt" in ext_ps and "steps" in ext_ps:
        t_ext = ext_ps["dt"] * np.arange(ext_ps["steps"] + 1, dtype=npfloat)
        PS_order_ext = ext_ps["max_ps"]
    else:
        raise ValueError(
            "External PS H5 file has no time information "
            "(no 't', no 'time', no 'dt/steps').")

    vxe = y_ext[3].astype(np.float64)
    vye = y_ext[4].astype(np.float64)
    vze = y_ext[5].astype(np.float64)

    E_ext = 0.5 * (vxe**2 + vye**2 + vze**2)
    rel_drift_ext = (E_ext - E_ext[0]) / E_ext[0]


if USE_EXTERNAL_H5_rk4:
    external_rk4 = load_results_h5(external_h5_rk4)
    ext_rk4 = external_rk4["rk4"]
    y_rk4_ext = ext_rk4["y"]   

    if "t" in ext_rk4 and ext_rk4["t"] is not None:
        t_eval_rk4_ext = ext_rk4["t"]

    elif "dt" in ext_rk4 and "steps" in ext_rk4:
        t_eval_rk4_ext = ext_rk4["dt"] * np.arange(ext_rk4["steps"] + 1, dtype=npfloat)
    else:
        raise ValueError(
            "External RK4 H5 file has no time information "
            "(no 't', no 'dt/steps').")

    # ensure shape consistency 
    if y_rk4_ext.shape[0] != 6:
        y_rk4_ext = y_rk4_ext.T

    # velocity, energy, drift ----
    v_rk4_ext = y_rk4_ext[3:6]
    E_rk4_ext = 0.5 * np.sum(v_rk4_ext**2, axis=0)
    rel_drift_rk4_ext = np.abs(E_rk4_ext - E_rk4_ext[0]) / E_rk4_ext[0]


if USE_EXTERNAL_H5_rk45:
    externalb = load_results_h5(external_h5_rk45)
    ext_rk45 = externalb["rk45"]
    t_eval_rk45_ext = ext_rk45["t"]
    y_rk45_ext = ext_rk45["y"]   

    # ensure shape consistency 
    if y_rk45_ext.shape[0] != 6:
        y_rk45_ext = y_rk45_ext.T

    # velocity, energy, drift 
    v_rk45_ext = y_rk45_ext[3:6]
    E_rk45_ext = 0.5 * np.sum(v_rk45_ext**2, axis=0)
    rel_drift_rk45_ext = np.abs(E_rk45_ext - E_rk45_ext[0]) / E_rk45_ext[0]


if USE_EXTERNAL_H5_rkg:
    external_rkg = load_results_h5(external_h5_rkg)

    ext_rkg = external_rkg["rkg"]
    y_ext_rkg = ext_rkg["y"]   

    
    if "t" in ext_rkg and ext_rkg["t"] is not None:
        t_ext_rkg = ext_rkg["t"]
    elif "dt" in ext_rkg and "steps" in ext_rkg:
        t_ext_rkg = ext_rkg["dt"] * np.arange(ext_rkg["steps"] + 1, dtype=npfloat)
    else:
        raise ValueError(
            "External RKG H5 file has no time information "
            "(no 't', no 'dt/steps').")

    #  ensure shape consistency 
    if y_ext_rkg.shape[0] == 6:
        y_ext_rkg = y_ext_rkg.T

    # Split canonical variables
    r_rkg_ext = y_ext_rkg[:, 0:3]
    p_rkg_ext = y_ext_rkg[:, 3:6]

    # Recompute vector potential
    A_rkg_ext = np.zeros_like(r_rkg_ext)
    for i in range(len(r_rkg_ext)):
        A_rkg_ext[i] = vector_potential_dipole(r_rkg_ext[i])

    # velocity, energy, drift 
    v_rkg_ext = p_rkg_ext - A_rkg_ext
    E_rkg_ext = npfloat(0.5) * np.sum(v_rkg_ext**2, axis=1, dtype=npfloat)
    rel_drift_ext_rkg = np.abs(E_rkg_ext - E_rkg_ext[0]) / E_rkg_ext[0]

if USE_PS:
    if not PS_CHUNKING:
        v_ps = solution_ps[3:6]             
        E_ps = 0.5 * np.sum(v_ps * v_ps, axis=0)
        E_ps_0 = E_ps[0]
        rel_drift_ps = np.abs(E_ps - E_ps_0) / E_ps_0
        t_ps_store = ps_step * np.arange(rel_drift_ps.size, dtype=npfloat)
        t_ps_plot = t_ps_store[::stride]
        rel_drift_ps = rel_drift_ps[::stride]
    else:
        with h5py.File(cache_path, "r") as ps_h5:
            ps_y_h5 = ps_h5["ps"]["y"]

            t_ps_plot, rel_drift_ps = compute_energy_ps_chunked(
                ps_y_h5=ps_y_h5,
                E0_ps=E0_ps,
                dt_ps_store=ps_step * (PS_decimate if PS_CHUNKING and PS_decimate > 1 else 1),
                chunk_cols=MAX_PLOT_POINTS,
                stride=energy_stride,
                return_plot_data=True,
            )


# === If using Hamiltonian in RKG ==
if USE_RKG:
    r_rkg = solution_rkg[:, 0:3]
    p_rkg = solution_rkg[:, 3:6]
    A_rkg = np.zeros_like(r_rkg)
    for i in range(len(r_rkg)):
        A_rkg[i] = vector_potential_dipole(r_rkg[i])
    v_rkg = p_rkg - A_rkg
    E_rkg = npfloat(0.5) * np.sum(v_rkg**2, axis=1, dtype=npfloat)
    E_rkg_0 = E_rkg[0]
    rel_drift_rkg = np.abs(E_rkg - E_rkg_0) / E_rkg_0

# === If using Lorentz force in RKG (not part of paper) ==
# r_rkg = solution_rkg[:, 0:3]
# v_rkg = solution_rkg[:, 3:6]  

if USE_RK45:
    v_rk45 = y_rk45_common[3:6]   
    E_rk45 = 0.5 * np.sum(v_rk45**2, axis=0)
    E_rk45_0 = E_rk45[0]
    rel_drift_rk45 = np.abs(E_rk45 - E_rk45_0) / E_rk45_0

if USE_RK4:
    v_rk4 = solution_rk4[3:6]  
    E_rk4 = npfloat(0.5) * np.sum(v_rk4**2, axis=0, dtype=npfloat)
    E_rk4_0 = E_rk4[0]
    rel_drift_rk4 = np.abs(E_rk4 - E_rk4_0) / E_rk4_0

if USE_EXTERNAL_H5_ps:
    ln_ext, = ax.semilogy((t_ext[1:]) * time_factor, np.abs(rel_drift_ext[1:]), alpha=0.8, color='#009E73', linestyle=':')
if USE_EXTERNAL_H5_rk4:
    ln_extrk4, = ax.semilogy((t_eval_rk4_ext[1:]) * time_factor, np.abs(rel_drift_rk4_ext[1:]), alpha=0.8, color='#CC79A7', linestyle='-.')
if USE_EXTERNAL_H5_rk45:
    ln_extb, = ax.semilogy((t_eval_rk45_ext[1:]) * time_factor, np.abs(rel_drift_rk45_ext[1:]), alpha=0.8, color='#E69F00', linestyle='--')
if USE_EXTERNAL_H5_rkg:
    ln_extc, = ax.semilogy((t_ext_rkg[1:]) * time_factor, np.abs(rel_drift_ext_rkg[1:]), alpha=0.8, color='#CC0000', linestyle='-.')

if USE_PS:
    lnps, = ax.semilogy(t_ps_plot[1:] * time_factor, np.abs(rel_drift_ps[1:]), label=f"PS{ps_order_label}", alpha=0.8, color="#009E73", linestyle=":")
if USE_RK4:
    t_rk4 = rk4_step * np.arange(len(rel_drift_rk4), dtype=npfloat)
    lnrk4, = ax.semilogy(t_rk4[1:] * time_factor, np.abs(rel_drift_rk4[1:]), label='RK4', alpha=0.8, color='#CC79A7', linestyle='-.')
if USE_RKG:
    t_rkg = rkg_step * np.arange(len(rel_drift_rkg), dtype=npfloat)
    lnrkg, = ax.semilogy(t_rkg[1:] * time_factor, np.abs(rel_drift_rkg[1:]), label='RKG', alpha=0.8, color='#CC0000', linestyle='-.')
if USE_RK45:
    t_rk45 = ps_step * np.arange(len(rel_drift_rk45), dtype=npfloat)
    lnrk45, = ax.semilogy(t_rk45[1:] * time_factor, np.abs(rel_drift_rk45[1:]), label='RK45', color='#E69F00', linestyle='--')


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

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.set_xlabel(r"$\tau/T$")
ax.set_ylabel(r"$|\Delta E|/E_0$")

if USE_PLOT_TITLES: ax.set_title(f"{particle_type} Relative Kinetic Energy Error in Dipole B Field")

fig.subplots_adjust(right=0.9)
fig.canvas.draw()
ax_pos = ax.get_position()  # Bbox in figure coords
x_fig_label = ax_pos.x1   # a small gap to the right of axes


# Getting labels for end of graphs to work in log plotting, dear lord don't touch this 
endpoints = []
if USE_PS:
    endpoints.append(( norm_time * time_factor, np.abs(rel_drift_ps[-1]), f"PS{ps_order_label}", lnps.get_color()))
if USE_RK4:
    endpoints.append(( norm_time * time_factor, np.abs(rel_drift_rk4[-1]), "RK4", lnrk4.get_color()))
if USE_RKG:
    endpoints.append(( norm_time * time_factor, np.abs(rel_drift_rkg[-1]), "RKG", lnrkg.get_color()))
if USE_RK45:
    endpoints.append(( norm_time * time_factor, np.abs(rel_drift_rk45[-1]), "RK45", lnrk45.get_color()))


if USE_EXTERNAL_H5_ps:
    endpoints.append((t_ext[-1]*time_factor, np.abs(rel_drift_ext[-1]), f"PS{PS_order_ext}", ln_ext.get_color()))
if USE_EXTERNAL_H5_rk4:
    endpoints.append((t_eval_rk4_ext[-1]*time_factor, np.abs(rel_drift_rk4_ext[-1]), f"RK4", ln_extrk4.get_color()))
if USE_EXTERNAL_H5_rk45:
    endpoints.append((t_eval_rk45_ext[-1]*time_factor, np.abs(rel_drift_rk45_ext[-1]), f"RK45", ln_extb.get_color()))    
if USE_EXTERNAL_H5_rkg:
    endpoints.append((t_ext_rkg[-1]*time_factor, np.abs(rel_drift_ext_rkg[-1]), f"RKG", ln_extc.get_color()))

xmin, xmax = ax.get_xlim()
ax.set_xlim(xmin, xmax * 1.05)

last_fy = None
min_gap = 0.025  # min gap if labels are close as fraction of axes height, don't go less than 0.025
endpoints_sorted = sorted(endpoints, key=lambda e: e[1])

for x, y, label, color in endpoints_sorted:
    if ax.get_yscale() == "log" and y <= 0:
        continue

    # Convert data -> figure y
    _, fy = data_to_fig(x, y, ax, fig)

    fy_adj = fy
    if last_fy is not None and fy_adj - last_fy < min_gap:
        fy_adj = last_fy + min_gap

    # Convert figure y-shift to point offset
    dy_pts = (fy_adj - fy) * fig.get_figheight() * 72

    ax.annotate(
        label,
        xy=(x, y),
        xytext=(5, dy_pts),  
        textcoords="offset points",
        ha="left",
        va="center",
        fontsize=11,
        color=color,
        clip_on=False,
        zorder=10,
    )

    last_fy = fy_adj

# === Save and Close ===
fig.canvas.draw()   
if USE_PS:
    fig.savefig( f"{output_folder}/{stem}_DipoleB_chunk_{particle_type}_{KE_particle:.1e}eV_{ps_step}step_PS{ps_order_label}_pitch{pitch_deg}_phi{phi_deg}_{norm_time:.2e}s_{npfloat.__name__}_KEerror.png", dpi=600, bbox_inches="tight")
else:
    fig.savefig( f"{output_folder}/{stem}_DipoleB_chunk_{particle_type}_{KE_particle:.1e}eV_pitch{pitch_deg}_phi{phi_deg}_{norm_time:.2e}s_{npfloat.__name__}_KEerror.png", dpi=600, bbox_inches="tight")
plt.close(fig)  

if DEBUG:
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    midpoint_ps = int(round(len(rel_drift_ps) / 2))
    logger.info(f"Peak memory usage for KE analysis: {peak / 1024**2:.2f} MB")
    logger.info(f"energy stride: {energy_stride}")
    logger.debug(f"[PS] E rel drift initial ={rel_drift_ps[0]:.2e}, E rel drift mid ={rel_drift_ps[midpoint_ps]:.2e}, E rel drift final ={rel_drift_ps[-1]:.2e}")
    
    if USE_RK4: midpoint_rk4 = int(round(len(rel_drift_rk4) / 2))
    if USE_RKG: midpoint_rkg = int(round(len(rel_drift_rkg) / 2))
    if USE_RK45: midpoint_rk45 = int(round(len(rel_drift_rk45) / 2))
    
    if USE_RKG: logger.debug(f"[RKG] E rel drift initial ={rel_drift_rkg[0]:.2e}, E rel drift mid ={rel_drift_rkg[midpoint_rkg]:.2e}, E rel drift final ={rel_drift_rkg[-1]:.2e}")
    if USE_RK4: logger.debug(f"[RK4] E rel drift initial ={rel_drift_rk4[0]:.2e}, E rel drift mid ={rel_drift_rk4[midpoint_rk4]:.2e}, E rel drift final ={rel_drift_rk4[-1]:.2e}")
    if USE_RK45: logger.debug(f"[RK45] E rel drift initial ={rel_drift_rk45[0]:.2e}, E rel drift mid ={rel_drift_rk45[midpoint_rk45]:.2e}, E rel drift final ={rel_drift_rk45[-1]:.2e}")

# ============================================================
# ================ Magnetic Moment Deviations ================
# ============================================================
"Defining Magnetic Moment at initiation"
if DEBUG: tracemalloc.start()

if USE_RK4:
    window_steps_rk4 = N_GYRO * N_STEPS_PER_GYRO_rk4
    mu0_rk4 = compute_mu_rk(solution_rk4[:, 0:1].T, mass)[0]

    if gyro_window == "last":
        i1_rk4 = steps_rk4
        i0_rk4 = max(0, i1_rk4 - window_steps_rk4)
    elif gyro_window == "first":
        i0_rk4 = 0
        i1_rk4 = min(window_steps_rk4, steps_rk4)
    elif gyro_window == "all":
        i0_rk4 = 0
        i1_rk4 = steps_rk4
    else:
        raise ValueError("gyro_window must be 'first', 'last', or 'all'")

    mu_rk4 = compute_mu_rk(solution_rk4[:, i0_rk4:i1_rk4].T, mass)
    mudrift_rk4 = np.abs(mu_rk4 - mu0_rk4) / mu0_rk4
    t_rk4_plot = (i0_rk4 + np.arange(mudrift_rk4.size, dtype=npfloat)) * rk4_step * time_factor

if USE_RKG:
    window_steps_rkg = N_GYRO * N_STEPS_PER_GYRO_rkg
    r0 = solution_rkg[0, 0:3]
    p0 = solution_rkg[0, 3:6]
    A0 = vector_potential_dipole(r0)
    v0 = p0 - A0
    state0 = np.hstack((r0, v0))[None, :]
    mu0_rkg = compute_mu_rk(state0, mass)[0]


    if gyro_window == "last":
        i1_rkg = steps_rkg
        i0_rkg = max(0, i1_rkg - window_steps_rkg)
    elif gyro_window == "first":
        i0_rkg = 0
        i1_rkg = min(window_steps_rkg, steps_rkg)
    elif gyro_window == "all":
        i0_rkg = 0
        i1_rkg = steps_rkg
    else:
        raise ValueError("gyro_window must be 'first', 'last', or 'all'")

    r_rkg = solution_rkg[i0_rkg:i1_rkg, 0:3]
    p_rkg = solution_rkg[i0_rkg:i1_rkg, 3:6]
    A_rkg = np.zeros_like(r_rkg)
    for i in range(len(r_rkg)):
        A_rkg[i] = vector_potential_dipole(r_rkg[i])
    v_rkg = p_rkg - A_rkg
    state_rkg = np.hstack((r_rkg, v_rkg))

    mu_rkg = compute_mu_rk(state_rkg, mass)
    mudrift_rkg = np.abs(mu_rkg - mu0_rkg) / mu0_rkg
    t_rkg_plot = (i0_rkg + np.arange(mudrift_rkg.size, dtype=npfloat)) * rkg_step * time_factor


if USE_RK45:
    window_steps_ps = N_GYRO * N_STEPS_PER_GYRO_ps
    y0 = y_rk45_common[:, 0:1]   
    mu0_rk45 = compute_mu_rk(y0.T, mass)[0]


    if gyro_window == "last":
        i1_rk45 = steps_ps
        i0_rk45 = max(0, i1_rk45 - window_steps_ps)
    elif gyro_window == "first":
        i0_rk45 = 0
        i1_rk45 = min(window_steps_ps, steps_ps)
    elif gyro_window == "all":
        i0_rk45 = 0
        i1_rk45 = steps_ps
    else:
        raise ValueError("gyro_window must be 'first', 'last', or 'all'")

    mu_rk45 = compute_mu_rk(y_rk45_common[:, i0_rk45:i1_rk45].T, mass)
    mudrift_rk45 = np.abs(mu_rk45 - mu0_rk45) / mu0_rk45
    t_rk45_plot = (i0_rk45 + np.arange(mudrift_rk45.size, dtype=npfloat)) * ps_step * time_factor

if USE_PS and not PS_CHUNKING:
    window_steps_ps = N_GYRO * N_STEPS_PER_GYRO_ps

    if gyro_window == "last":
        i1_phys = steps_ps
        i0_phys = max(0, i1_phys - window_steps_ps)
    elif gyro_window == "first":
        i0_phys = 0
        i1_phys = min(window_steps_ps, steps_ps)
    elif gyro_window == "all":
        i0_phys = 0
        i1_phys = steps_ps
    else:
        raise ValueError("gyro_window must be 'first', 'last', or 'all'")

    mu_ps = compute_mu_ps(solution_ps[:, i0_phys:i1_phys], mass)
    mudrift_ps = np.abs(mu_ps - mu0_ps) / mu0_ps
    t_ps_store = np.arange(i0_phys, i1_phys, dtype=npfloat) * ps_step
    moment_stride = max(1, round(len(mu_ps) // MAX_PLOT_POINTS))
    t_ps_plot = t_ps_store[::moment_stride] * time_factor
    mudrift_ps_plot = mudrift_ps[::moment_stride]

elif USE_PS and PS_CHUNKING:
    window_steps_ps = N_GYRO * N_STEPS_PER_GYRO_ps

    if gyro_window == "last":
        i1_phys = steps_ps
        i0_phys = max(0, i1_phys - window_steps_ps)
    elif gyro_window == "first":
        i0_phys = 0
        i1_phys = min(window_steps_ps, steps_ps)
    elif gyro_window == "all":
        i0_phys = 0
        i1_phys = steps_ps
    else:
        raise ValueError("gyro_window must be 'first', 'last', or 'all'")

    ps_store_stride = PS_decimate if (PS_decimate > 1) else 1
    j0 = int(np.ceil(i0_phys / ps_store_stride))
    j1 = int(np.floor(i1_phys / ps_store_stride))

    with h5py.File(cache_path, "r") as ps_h5:
        ps_grp = ps_h5["ps"]
        ps_y = ps_grp["y"]
        ps_order_label = int(ps_grp.attrs["max_ps"])
        n_store = ps_y.shape[1]

        j0 = max(0, min(j0, n_store - 1))
        j1 = max(0, min(j1, n_store - 1))

        if j1 < j0:
            raise RuntimeError("Empty PS μ window (chunked)")

        y_ps_win = ps_y[:, j0:j1]
        mu_ps = compute_mu_ps(y_ps_win, mass)
        mudrift_ps = np.abs(mu_ps - mu0_ps) / mu0_ps

        dt_ps_store = ps_step * ps_store_stride
        t_ps_store = np.arange(j0, j1, dtype=npfloat) * dt_ps_store
        moment_stride = max(1, round(len(mu_ps) // MAX_PLOT_POINTS))
        t_ps_plot = t_ps_store[::moment_stride] * time_factor
        mudrift_ps_plot = mudrift_ps[::moment_stride]

# Plotting

fig, ax = plt.subplots(figsize=(10, 5))

if USE_RK45:
    lnrk45, = ax.semilogy(t_rk45_plot, mudrift_rk45, label="RK45", color="#E69F00", linestyle="--")
if USE_RK4:
    lnrk4, = ax.semilogy(t_rk4_plot, mudrift_rk4, label="RK4", alpha=0.3, color="#CC79A7", linestyle="-.")
if USE_RKG:
    lnrkg, = ax.semilogy(t_rkg_plot, mudrift_rkg, label="RKG", alpha=0.3, color="#CC0000", linestyle="-.")
if USE_PS:
    lnps, = ax.semilogy(t_ps_plot, mudrift_ps_plot, label=f"PS{ps_order_label}", alpha=0.7, color="#009E73", linestyle=":")

ax.margins(x=0.01)
ax.set_yscale("log")
ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=100))
ax.yaxis.set_major_formatter(LogFormatterSciNotation(base=10.0))
ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=[]))
ax.yaxis.set_minor_formatter(NullFormatter())
ax.grid(True, which="major", linestyle="--", linewidth=0.7)
# ax.set_ylim( 5e-7, 5e0)
ax.get_xaxis().get_major_formatter().set_useOffset(False)

# # for top slices of mu 
# ax.set_ylim(1e-1, 2e-1)
# ax.set_yscale('linear')

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.set_xlabel(r"$\tau/T$")
ax.set_ylabel(r"$|\Delta \mu|/\mu_\emptyset$")

if USE_PLOT_TITLES: ax.set_title(f"{particle_type} Magnetic Moment Variations in Dipole B Field")

fig.subplots_adjust(right=0.9)
fig.canvas.draw()
ax_pos = ax.get_position()
x_fig_label = ax_pos.x1

# Getting labels for end of graphs to work in log plotting, dear lord don't touch this 
endpoints = []
if USE_RK45:
    endpoints.append((t_rk45_plot[-1], float(np.abs(mudrift_rk45[-1])), "RK45", lnrk45.get_color()))
if USE_RK4:
    endpoints.append((t_rk4_plot[-1],  float(np.abs(mudrift_rk4[-1])),  "RK4",  lnrk4.get_color()))
if USE_RKG:
    endpoints.append((t_rkg_plot[-1],  float(np.abs(mudrift_rkg[-1])),  "RKG",  lnrkg.get_color()))
if USE_PS:
    endpoints.append((t_ps_plot[-1],   float(np.abs(mudrift_ps[-1])),   f"PS{ps_order_label}", lnps.get_color()))

labels = []
for x, y, label, color in endpoints:
    _, fy = data_to_fig(x, y, ax, fig)
    fy = min(max(fy, ax_pos.y0), ax_pos.y1)
    labels.append([fy, label, color])

labels.sort(key=lambda v: v[0])

min_gap = 0.025
for i in range(1, len(labels)):
    if labels[i][0] - labels[i-1][0] < min_gap:
        labels[i][0] = labels[i-1][0] + min_gap

for i in range(len(labels)-2, -1, -1):
    if labels[i+1][0] - labels[i][0] < min_gap:
        labels[i][0] = labels[i+1][0] - min_gap

for fy, label, color in labels:
    fig.text(x_fig_label, fy, label, color=color, va="center", ha="left", fontsize=11)

# === Save and Close ===
if USE_PS:
    fig.savefig( f"{output_folder}/{stem}_DipoleB_chunk_{particle_type}_{KE_particle:.1e}eV_{ps_step}step_PS{ps_order_label}_pitch{pitch_deg}_phi{phi_deg}_{norm_time:.2e}s_{npfloat.__name__}_mu.png", dpi=600, bbox_inches="tight")
else:
    fig.savefig( f"{output_folder}/{stem}_DipoleB_chunk_{particle_type}_{KE_particle:.1e}eV_pitch{pitch_deg}_phi{phi_deg}_{norm_time:.2e}s_{npfloat.__name__}_mu.png", dpi=600, bbox_inches="tight")
plt.close(fig)  


if DEBUG:
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    logger.info(f"Peak memory usage for moment analysis: {peak / 1024**2:.2f} MB")
    if USE_PS: mumidpoint_ps = int(round(len(mudrift_ps) / 2))
    if USE_RK4: mumidpoint_rk4 = int(round(len(mudrift_rk4) / 2))
    if USE_RKG: mumidpoint_rkg = int(round(len(mudrift_rkg) / 2))
    if USE_RK45: mumidpoint_rk45 = int(round(len(mudrift_rk45) / 2))
    
    if USE_PS: logger.debug(f"mu midpoint: {mumidpoint_ps}")
    if USE_RK4: logger.debug(f"mu midpoint: {mumidpoint_rk4}")
    if USE_RKG: logger.debug(f"mu midpoint: {mumidpoint_rkg}")
    if USE_RK45: logger.debug(f"mu midpoint: {mumidpoint_rk45}")

    logger.info(f"moment stride: {stride}")
    logger.debug(f"[PS] mu rel drift initial ={mudrift_ps[0]:.2e}, mu rel drift mid ={mudrift_ps[mumidpoint_ps]:.2e}, mu rel drift final ={mudrift_ps[-1]:.2e}")
    if USE_RKG: logger.debug(f"[RKG] mu rel drift initial ={mudrift_rkg[0]:.2e}, mu rel drift mid ={mudrift_rkg[mumidpoint_rkg]:.2e}, mu rel drift final ={mudrift_rkg[-1]:.2e}")
    if USE_RK4: logger.debug(f"[RK4] mu rel drift initial ={mudrift_rk4[0]:.2e}, mu rel drift mid ={mudrift_rk4[mumidpoint_rk4]:.2e}, mu rel drift final ={mudrift_rk4[-1]:.2e}")
    if USE_RK45: 
        logger.debug(f"[RK45] mu rel drift initial ={mudrift_rk45[0]:.2e}, mu rel drift mid ={mudrift_rk45[mumidpoint_rk45]:.2e}, mu rel drift final ={mudrift_rk45[-1]:.2e}")
        logger.debug(f"[RK45 Slice] t_start={t_rk45[0]:.3e}, t_end={t_rk45[-1]:.3e}, len={len(t_rk45)}")


# ===================================================
# ================ Mirror and Drift  ================
# ===================================================
# only for PS method currently and not when we've chunked data

if USE_PS and not PS_CHUNKING:

    v_eps = npfloat(1e-14) * v_tau
    user_min_gap = max(3, int(0.5 * T_gyro / ps_step))

    ps_analysis = solution_ps
    dt_ps_eff = ps_step

    idxs, crossings_tau = mirror_times_from_PS(ps_analysis, dt_ps_eff, interp=True, min_gap=user_min_gap, s_eps=v_eps)

    bounce_stats = bounce_summary(crossings_tau, time_scale_sec=tau_time)

    if bounce_stats["full_mean_s"] is not None:
        print("Mirror crossings:", bounce_stats["n_crossings"])
        print(f"Full bounce period (mean): {bounce_stats['full_mean_s']:.6g} s")
        print("Bounce frequency [Hz]:", bounce_stats["bounce_frequency_hz"])
    else:
        print("No mirror motion detected (no full-bounce interval).")

    #Drift (use mirrors if we have them; else raw)
    has_mirrors = (crossings_tau is not None) and (len(crossings_tau) >= 2)

    drift_stats = drift_period_from_PS(
        final_coeff_matrix=ps_analysis,
        dt_tau=dt_ps_eff,
        mirror_times_tau=crossings_tau if has_mirrors else None,
        sample="mirrors" if has_mirrors else "raw",
        time_scale_sec=tau_time,
        min_phase_rad=user_min_phase,
        return_details=False
    )
    # 3) Report (prefer crossings-mean, fallback to slope-fit)
    T_drift_s   = drift_stats["period_s_mean"] or drift_stats["period_s_fit"]
    T_drift_tau = drift_stats["period_tau_mean"] or drift_stats["period_tau_fit"]
    direction   = drift_stats["direction"]

    if T_drift_s is None:
        print("Drift period: not enough azimuthal motion to estimate (yet).")
    else:
        print(f"Drift period ≈ {T_drift_s:.6g} s  (≈ {T_drift_tau:.6g} (normalized time), direction {'east' if direction>0 else 'west'})")

else:
    print("⚠️  Mirror and drift analysis skipped (PS chunked mode).")

# ====================================
# === Write Summary Output to File ===
# ====================================
if DEBUG: tracemalloc.start()

if gyroperiods < 1e6:
    TAIL_FRAC = 0.01        # last 1%
else:
    TAIL_FRAC = 0.0001     # last 0.01%

tail_start = (1.0 - TAIL_FRAC) * npfloat(norm_time)

MAX_TAIL_STEPS = 500_000   # hard safety cap
tail_masks = {}

def make_tail_mask(n_points, step_size, label=None):
    j0 = int(tail_start / step_size)
    j0 = max(0, min(j0, n_points - 1))

    # Clamp tail size
    if n_points - j0 > MAX_TAIL_STEPS:
        j0 = n_points - MAX_TAIL_STEPS

    mask = np.zeros(n_points, dtype=bool)
    mask[j0:] = True

    if not np.any(mask):
        NMIN = min(1000, n_points)
        mask[-NMIN:] = True
        j0 = n_points - NMIN

    return mask, j0

# ============================================================
# Build tail masks for last fraction of invariants
# ============================================================

if USE_PS:
    step_ps = ps_store_stride * ps_step
    tail_masks["PS"], j0_ps = make_tail_mask(rel_drift_ps.size, step_ps, "PS")

if USE_RK45:
    tail_masks["RK45"], j0_rk45 = make_tail_mask(len(rel_drift_rk45), ps_step, "RK45")

if USE_RK4:
    tail_masks["RK4"], j0_rk4 = make_tail_mask(len(rel_drift_rk4), rk4_step, "RK4")

if USE_RKG:
    tail_masks["RKG"], j0_rkg = make_tail_mask(len(rel_drift_rkg), rkg_step, "RKG")

# ============================================================
# Write summary file
# ============================================================

output_filename = (
    f"{output_folder}/{stem}_DipoleB_chunk_{particle_type}_"
    f"{KE_particle:.1e}eV_{ps_step}step_PS{ps_order_label}_"
    f"pitch{pitch_deg}_phi{phi_deg}_{norm_time:.2e}s_"
    f"{npfloat.__name__}_simulation_summary.txt"
    if USE_PS else
    f"{output_folder}/{stem}_DipoleB_chunk_{particle_type}_"
    f"{KE_particle:.1e}eV_pitch{pitch_deg}_phi{phi_deg}_"
    f"{norm_time:.2e}s_{npfloat.__name__}_simulation_summary.txt"
)

with open(output_filename, "w") as f:

    if WRITE_DATA or READ_DATA:
        f.write(f"Run Data: {stem}.hd5\n\n")

    f.write("=== Simulation Summary ===\n")
    f.write(f"particle = {particle_type}\n")
    f.write(f"energy   = {KE_particle} eV\n")
    f.write(f"pitch    = {pitch_deg} deg\n")
    f.write(f"phi      = {phi_deg} deg\n")
    f.write(f"ps step  = {ps_step}\n")
    f.write(f"norm_time = {norm_time}\n\n")

    f.write("=== Timing Summary ===\n")
    for k in ("rk45", "rk4", "rkg", "ps"):
        if k in timing:
            f.write(f"  Run Time {k.upper()} = {timing[k]:.2f} s\n")

    f.write("\n=== |delta E|/E0 (tail average) ===\n")
    if USE_RK45:
        summarize_error("RK45", rel_drift_rk45[j0_rk45:], f)
    if USE_RK4:
        summarize_error("RK4", rel_drift_rk4[j0_rk4:], f)
    if USE_RKG:
        summarize_error("RKG", rel_drift_rkg[j0_rkg:], f)
    if USE_PS:
        summarize_error("PS", rel_drift_ps[j0_ps:], f)

    f.write("\n=== |delta mu|/mu0 (tail average) ===\n")

    # === mu drift for RK45 ===
    if USE_RK45:
        y_tail = y_rk45_common[:, j0_rk45:] 
        mu_tail = compute_mu_rk(y_tail.T, mass)
        summarize_error("RK45", np.abs(mu_tail - mu0_rk45) / mu0_rk45, f)
        del y_tail, mu_tail
        gc.collect()

    # === mu drift for RK4 ===
    if USE_RK4:
        y_tail = solution_rk4[:, j0_rk4:]  
        mu_tail = compute_mu_rk(y_tail.T, mass)
        summarize_error("RK4", np.abs(mu_tail - mu0_rk4) / mu0_rk4, f)
        del y_tail, mu_tail
        gc.collect()

    # === mu drift for RKG ===
    if USE_RKG:
        r_tail = solution_rkg[j0_rkg:, 0:3]
        p_tail = solution_rkg[j0_rkg:, 3:6]

        A_tail = np.empty_like(r_tail)
        for i in range(len(r_tail)):
            A_tail[i] = vector_potential_dipole(r_tail[i])

        v_tail = p_tail - A_tail
        state_tail = np.hstack((r_tail, v_tail))

        mu_tail = compute_mu_rk(state_tail, mass)
        summarize_error("RKG", np.abs(mu_tail - mu0_rkg) / mu0_rkg, f)
        del r_tail, p_tail, A_tail, v_tail, state_tail, mu_tail
        gc.collect()

    # === mu drift for PS ===
    if USE_PS:

        # ---------- RAM ----------
        if not PS_CHUNKING:
            y_tail = solution_ps[:, j0_ps:]
            mu_tail = compute_mu_ps(y_tail, mass)
            summarize_error("PS", np.abs(mu_tail - mu0_ps) / mu0_ps, f)

            del y_tail, mu_tail
            gc.collect()

        # ---------- Chunking ----------
        else:
            step_ps = ps_store_stride * ps_step

            with h5py.File(cache_path, "r") as ps_h5:
                ps_y = ps_h5["ps"]["y"]
                n_store = ps_y.shape[1]

                j0 = int(tail_start / step_ps)
                j0 = max(0, min(j0, n_store - 1))

                if n_store - j0 > MAX_TAIL_STEPS:
                    j0 = n_store - MAX_TAIL_STEPS

                y_tail = ps_y[:, j0:]

            mu_tail = compute_mu_ps(y_tail, mass)
            summarize_error("PS", np.abs(mu_tail - mu0_ps) / mu0_ps, f)

            del y_tail, mu_tail
            gc.collect()

if DEBUG:
    if USE_RK4:logger.debug(f"  rk4 step size = {rk4_step}")
    if USE_RKG: logger.debug(f"  rkg step size = {rkg_step}")
    if USE_RK4: logger.debug(f"  rk4 steps     = {steps_rk4}")
    if USE_RKG: logger.debug(f"  rkg steps     = {steps_rkg}")
    if USE_PS: logger.debug(f"  ps steps      = {steps_ps}")

# === Shared metadata ===
run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
x_0, y_0, z_0 = x_initial, y_initial, z_initial

# === Collect rows (one per method) ===
def make_record(method, e_drift, mu_drift):
    if method == "PS":
        steps, dt = steps_ps, ps_step

    elif method == "RK45":
        # RK45 is evaluated on the PS grid by design
        steps, dt = steps_ps, ps_step

    elif method == "RK4":
        steps, dt = steps_rk4, rk4_step

    elif method == "RKG":
        steps, dt = steps_rkg, rkg_step

    else:
        raise ValueError(f"Unknown method: {method}")

    e = summarize(e_drift)
    mu = summarize(mu_drift)

    return {
        "run_id": stem,
        "particle": particle_type,
        "energy_keV": KE_particle,
        "x": x_0,
        "y": y_0,
        "z": z_0,
        "pitch_deg": pitch_deg,
        "phi_deg": phi_deg,
        "steps": steps,
        "dt": dt,
        "method": method,
        "energy_mean_err": e["mean"],
        "energy_max_err": e["max"],
        "energy_rms_err": e["rms"],
        "mu_mean_err": mu["mean"],
        "mu_max_err": mu["max"],
        "mu_rms_err": mu["rms"],
    }


records = []

methods = []
if USE_RK4:
    methods.append(("RK4",  rel_drift_rk4,  mudrift_rk4))
if USE_RK45:
    methods.append(("RK45", rel_drift_rk45, mudrift_rk45))
if USE_RKG:
    methods.append(("RKG",  rel_drift_rkg,  mudrift_rkg))
if USE_PS:
    methods.append(("PS",   rel_drift_ps,   mudrift_ps))

for method, e_drift, mu_drift in methods:
    records.append(make_record(method, e_drift, mu_drift))

# === Write to master log ===
df = pd.DataFrame(records)
csv_path = f"{output_folder}/master_simulation_log.csv"

if os.path.exists(csv_path):
    df.to_csv(csv_path, mode='a', header=False, index=False)
else:
    df.to_csv(csv_path, index=False)

print(f"\nRun Complete → {output_folder}/{stem}")

if DEBUG:
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    logger.info(f"Peak memory usage for summary write up: {peak / 1024**2:.2f} MB")
