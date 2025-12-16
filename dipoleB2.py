import numpy as np
import builtins
import test_particles.dipoleB_testparticles as tp
builtins.npfloat = np.float128 if tp.USE_FLOAT128 else np.float64
from test_particles.dipoleB_testparticles import *
import pandas as pd 
from datetime import datetime
import os
import time
import sys
import json
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from matplotlib.ticker import LogLocator, LogFormatterSciNotation, NullFormatter, FuncFormatter
from functions.functions_library_universal import rk4_fixed_step, plt_config, sparse_labels, data_to_fig, slice_solution
from functions.functions_library_dipole import PS_dipoleB, lorentz_force_dipole, compute_mu_ps, compute_mu_rk, vector_potential_dipole, rkgl4_hamiltonian, hamiltonian_rhs
from functions.functions_library_dipole import mirror_times_from_PS, bounce_summary, drift_period_from_PS, get_run_params, h5_path_for, save_results_h5, load_results_h5, run_ps_streaming_with_decimation
import h5py  # for streaming / chunked HDF5 writes

# === PS streaming / decimation options ===
# You can override these in dipoleB_testparticles.py if you want.
# PS_STREAMING = getattr(tp, "PS_STREAMING", False)
# PS_CHUNK_STEPS = getattr(tp, "PS_CHUNK_STEPS", int(1e6))   # number of PS steps per chunk
# PS_DECIMATE   = getattr(tp, "PS_DECIMATE", 1)              # keep every PS_DECIMATE-th step (1 = no decimation)

PS_STREAMING = True   # for this file, we want streaming mode
PS_CHUNK_STEPS = int(1e5)
PS_DECIMATE = 500

run = "demo"   # options: "demo", "paper1", "paper2", or "paper3". Demo mode is a quick test run. Paper modes can take upwards of half an hour. 

# Allow command-line override
if len(sys.argv) > 1:
    run = sys.argv[1]
    print(f"Run mode set from command line: {run}\n")
else:
    print(f"Using default run mode: {run}\n")

globals().update(load_params(run))

# === Misc Odds and Ends ===  
plt_config(scale=1)                   # config file for setting plot sizes and fonts (from Dr. W)
os.makedirs(run_storage, exist_ok=True)
plt.ioff()              # Turn off interactive mode for plots
if USE_FLOAT128: USE_RKG=False

# for file/plot naming
if mass_si == m_e:
    particle_type = "Electron"
elif mass_si == m_p:
    particle_type = "Proton"
else:
    particle_type = "Particle"

qoverm = npfloat(-1) if mass_si == m_e else npfloat(1)


# === Misc Conversions  ===
KE_joules = KE_particle * evtoj                     # converting KE from eV to Joules
gamma = 1.0 + KE_joules / (mass_si * spdlight**2)   # Lorentz factor
mass = gamma * mass_si                              # Relativistic mass used for magnetic moment calculations
v_si = spdlight * np.sqrt(1.0 - 1.0 / gamma**2)     # m/s
tau_time = gamma * mass_si / (abs(q_e) * abs(B_0))  # this is tau0 from paper 
v_tau = v_si * tau_time / RE                        # dimensionless velocity
physical_time = norm_time * abs(tau_time)           # actual physical time, t; normalized time =t/tau_time


# === Velocity Config based on INput Angles ===
pitch_rad = npfloat(np.radians(pitch_deg))              # degrees to radians, pitch 
phi_rad = npfloat(np.radians(phi_deg))                  # degrees to radians, phi 
v_par = npfloat(v_tau) * npfloat(np.cos(pitch_rad))     # parallel velocity component
v_perp = npfloat(v_tau) * npfloat(np.sin(pitch_rad))    # perpendicular velocity component 
vx_initial = npfloat(v_perp * np.cos(phi_rad))          
vy_initial = npfloat(v_perp * np.sin(phi_rad))
vz_initial = npfloat(v_par)
if abs(vx_initial) < (1.0 * np.finfo(npfloat).eps): vx_initial = npfloat(0.0) 
if abs(vy_initial) < (1.0 * np.finfo(npfloat).eps): vy_initial = npfloat(0.0)
if abs(vz_initial) < (1.0 * np.finfo(npfloat).eps): vz_initial = npfloat(0.0)
gyro_radius_si=abs(v_si * np.sin(pitch_rad) * mass_si/ (q_e * B_0))


# these should be identical, kept seperate in case I decide to scale one method at a later point
initial_pos_vel = np.array([x_initial, y_initial, z_initial, vx_initial, vy_initial, vz_initial], dtype=npfloat)  
initial_pos_vel_ps = np.array([x_initial, y_initial, z_initial, vx_initial, vy_initial, vz_initial], dtype=npfloat) 

if USE_RKG:
    # === Symplectic Implementations =====
    r0 = np.array([x_initial, y_initial, z_initial])                # already normalized RE units
    v_tau_vec = np.array([vx_initial, vy_initial, vz_initial], dtype=npfloat)
    A0 = vector_potential_dipole(r0)
    p0 = v_tau_vec + A0
    y0 = np.concatenate((r0, p0))           # for Hamiltonian in RKG
    # y0 = np.concatenate((r0, v_tau_vec))  # for Lorentz force in RKG, used as a sanity check


# === Ensures that Total Time Elapsed is the Same ===
# steps_ps = int(norm_time / (ps_step)) 
# t_eval_ps = np.linspace(0, norm_time, steps_ps + 1, dtype=npfloat)

# === Ensures that Total Time Elapsed is the Same ===
steps_ps = int(norm_time / (ps_step))

if not PS_STREAMING:
    # Original behavior: build full time array in memory
    t_eval_ps = np.linspace(0, norm_time, steps_ps + 1, dtype=npfloat)
else:
    # In streaming mode we do NOT pre-build the full time array.
    t_eval_ps = None   # will be built in a decimated form


if USE_RK4:
    steps_rk4 = int(norm_time / (rk4_step))
    t_eval_rk4 = np.linspace(0, norm_time, steps_rk4 + 1, dtype=npfloat)
if USE_RK45:
    t_eval_rk45 = t_eval_ps   # no steps so we just set equal to ps
if USE_RKG:
    steps_rkg = int(norm_time / (rkg_step))
    t_eval_rkg = np.linspace(0, norm_time, steps_rkg + 1, dtype=npfloat)

def run_ps_streaming_with_decimation(
    initial_pos_vel_ps,
    steps_ps,
    ps_step,
    PS_order,
    tol,
    qoverm,
    cache_path,
    write_data=True,
    chunk_steps=10**6,
    decimate=1,
):
    """
    Streaming PS driver:
    - advances PS in chunks using PS_dipoleB
    - optionally decimates output for storage/plotting
    - writes chunked data to HDF5
    - returns decimated in-memory arrays (t, y, orders)
    - returns runtime for timing
    """

    import time
    start_time_ps = time.time()

    # PS_dipoleB always returns 17-component rows
    n_state = 17

    # cur_state must ALWAYS be the 6 physical variables only
    cur_state = initial_pos_vel_ps.copy()   # shape (6,)
    remaining = steps_ps
    global_index = 0

    # In-memory decimated storage
    t_list = []
    y_list = []
    orders_list = []

    # Create the HDF5 file (or overwrite)
    if write_data:
        f = h5py.File(cache_path, "w")

        # meta group
        meta_grp = f.create_group("meta")

        # ps group
        ps_grp = f.create_group("ps")

        dset_t = ps_grp.create_dataset(
            "t",
            shape=(0,),
            maxshape=(None,),
            dtype=npfloat,
        )
        dset_y = ps_grp.create_dataset(
            "y",
            shape=(n_state, 0),
            maxshape=(n_state, None),
            dtype=npfloat,
            chunks=(n_state, min(chunk_steps, steps_ps + 1)),
        )
        dset_orders = ps_grp.create_dataset(
            "orders",
            shape=(0,),
            maxshape=(None,),
            dtype=np.int16,
        )
    else:
        f = None
        dset_t = dset_y = dset_orders = None

    try:
        while remaining > 0:
            this_chunk = min(chunk_steps, remaining)

            # 🔹 Compute chunk
            sol_chunk, orders_chunk = PS_dipoleB(
                PS_order, this_chunk, cur_state, tol, qoverm, ps_step
            )
            # sol_chunk: (17, this_chunk+1)
            # orders_chunk: (this_chunk+1,)

            # Global indices for this chunk
            idx_chunk = np.arange(global_index, global_index + this_chunk + 1, dtype=np.int64)

            # 🔹 Avoid duplicating boundary between chunks
            if global_index == 0:
                sol_eff = sol_chunk
                orders_eff = orders_chunk
                idx_eff = idx_chunk
            else:
                sol_eff = sol_chunk[:, 1:]
                orders_eff = orders_chunk[1:]
                idx_eff = idx_chunk[1:]

            # 🔹 Decimation
            if decimate <= 1:
                keep = np.ones_like(idx_eff, dtype=bool)
            else:
                keep = (idx_eff % decimate == 0) | (idx_eff == steps_ps)

            idx_keep = idx_eff[keep]
            sol_keep = sol_eff[:, keep]         # KEEP ALL 17 ROWS
            orders_keep = orders_eff[keep]

            # 🔹 Append to in-memory lists
            if idx_keep.size > 0:
                t_list.append(idx_keep.astype(npfloat) * npfloat(ps_step))
                y_list.append(sol_keep)
                orders_list.append(orders_keep.astype(np.int16))

                # 🔹 Stream to HDF5
                if write_data:
                    old_len = dset_t.shape[0]
                    new_len = old_len + idx_keep.size

                    dset_t.resize((new_len,))
                    dset_y.resize((n_state, new_len))
                    dset_orders.resize((new_len,))

                    dset_t[old_len:new_len] = idx_keep.astype(npfloat) * npfloat(ps_step)
                    dset_y[:, old_len:new_len] = sol_keep
                    dset_orders[old_len:new_len] = orders_keep.astype(np.int16)

            # 🔹 Prepare next-chunk initial state
            cur_state = sol_chunk[0:6, -1].copy()   # ONLY pass (x,y,z,vx,vy,vz)

            # Step forward
            global_index += this_chunk
            remaining -= this_chunk

        # Concatenate decimated arrays
        if t_list:
            t_dec = np.concatenate(t_list)
            y_dec = np.hstack(y_list)
            orders_dec = np.concatenate(orders_list)
        else:
            t_dec = np.zeros(1, dtype=npfloat)
            y_dec = initial_pos_vel_ps.reshape(n_state, 1)
            orders_dec = np.zeros(1, dtype=np.int16)

        end_time_ps = time.time()
        elapsed_ps = end_time_ps - start_time_ps

        return t_dec, y_dec, orders_dec, elapsed_ps

    finally:
        if f is not None:
            f.close()


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
                   PS_order, tol, qoverm)
cache_path = h5_path_for(params, run_storage)

# === Build parameter tracer & check cache ===
"""
this is scanning the files already stored to see if we already have the data,
beware that these files can be GB size for dipole
"""
params = get_run_params(
    USE_RK45, USE_RK4, USE_RKG,
    mass_si, q_e, B_0, gamma,
    x_initial, y_initial, z_initial,
    pitch_deg, phi_deg,
    norm_time, ps_step, rk4_step, rkg_step,
    PS_order, tol, qoverm
)
cache_path = h5_path_for(params, run_storage)

if os.path.exists(cache_path) and READ_DATA:
    # ==========================
    # ===== Load from cache =====
    # ==========================
    print(f"Found existing results: {os.path.basename(cache_path)} — loading.\n")
    cached = load_results_h5(cache_path)

    solution_ps = cached["ps"]["y"] if cached["ps"] else None
    orders_used = cached["ps"]["orders"] if cached["ps"] else None
    t_eval_ps   = cached["ps"]["t"] if cached["ps"] else None

    if USE_RK4 and cached["rk4"]:
        solution_rk4 = cached["rk4"]["y"]
        t_eval_rk4   = cached["rk4"]["t"]
    if USE_RK45 and cached["rk45"]:
        class _Obj: pass
        solution_rk45 = _Obj()
        solution_rk45.t = cached["rk45"]["t"]
        solution_rk45.y = cached["rk45"]["y"]
    if USE_RKG and cached["rkg"]:
        solution_rkg = cached["rkg"]["y"]
        t_eval_rkg   = cached["rkg"]["t"]

    timing = cached.get("meta", {}).get("timing", {})
    stem   = os.path.splitext(os.path.basename(cache_path))[0]

else:
    # ==========================
    # ===== Run all solvers ====
    # ==========================
    print("No matching file or 'Read Data' skipped. Running solvers...\n")

    timing = {}

    # ====== Run RK45 (optional) ======
    if USE_RK45:
        print("Running RK45...\n")
        start_time_rk45 = time.time()
        solution_rk45 = solve_ivp(
            lorentz_force_dipole, (0, norm_time),
            initial_pos_vel, method='RK45',
            t_eval=t_eval_rk45, args=(qoverm,),
            rtol=rtol_rk45, atol=atol_rk45
        )
        end_time_rk45 = time.time()
        timing["rk45"] = end_time_rk45 - start_time_rk45

    # ====== Run RK4 (optional) ======
    if USE_RK4:
        print("Running RK4...\n")
        start_time_rk4 = time.time()
        solution_rk4 = rk4_fixed_step(
            lorentz_force_dipole, initial_pos_vel, t_eval_rk4, args=(qoverm,))
        end_time_rk4 = time.time()
        timing["rk4"] = end_time_rk4 - start_time_rk4

    # ====== Run RKG (optional) ======
    if USE_RKG:
        print("Running RKG...\n")
        start_time_rkg = time.time()
        solution_rkg = rkgl4_hamiltonian(hamiltonian_rhs, y0, t_eval_rkg, args=(qoverm,))
        end_time_rkg = time.time()
        timing["rkg"] = end_time_rkg - start_time_rkg

    # ====== Run PS (streaming or standard) ======
    if PS_STREAMING:
        print("Running PS in streaming mode...\n")

        t_eval_ps, solution_ps, orders_used, ps_runtime = \
            run_ps_streaming_with_decimation(
                initial_pos_vel_ps=initial_pos_vel_ps,
                steps_ps=steps_ps,
                ps_step=ps_step,
                PS_order=PS_order,
                tol=tol,
                qoverm=qoverm,
                cache_path=cache_path,
                write_data=WRITE_DATA,
                chunk_steps=PS_CHUNK_STEPS,
                decimate=PS_DECIMATE,
            )

        timing["ps"] = ps_runtime

    else:
        print("Running PS in standard mode...\n")
        start_time_ps = time.time()
        solution_ps, orders_used = PS_dipoleB(
            PS_order, steps_ps, initial_pos_vel_ps, tol, qoverm, ps_step
        )
        end_time_ps = time.time()
        timing["ps"] = end_time_ps - start_time_ps

        if t_eval_ps is None:
            t_eval_ps = np.linspace(0, norm_time, steps_ps + 1, dtype=npfloat)

    # ===============================
    # ====== Saving Results  =========
    # ===============================
    if WRITE_DATA and (not PS_STREAMING):
        # Original save system for non-streaming mode
        results = {
            "ps": {"t": t_eval_ps, "y": solution_ps, "orders": orders_used},
            "rk4": {"t": t_eval_rk4, "y": solution_rk4} if USE_RK4 else None,
            "rk45": {"t": solution_rk45.t, "y": solution_rk45.y} if USE_RK45 else None,
            "rkg": {"t": t_eval_rkg, "y": solution_rkg} if USE_RKG else None,
            "meta": {
                "timing": timing,
                "physical_time": float(physical_time),
                "norm_time": float(norm_time),
                "percent_c": float(v_si/spdlight),
                "particle_label": (
                    f"{KE_particle:.1e} eV electron" if mass_si == m_e else
                    f"{KE_particle:.1e} eV proton" if mass_si == m_p else "manual"
                ),
            }
        }

        save_results_h5(cache_path, params, results)
        print(f"Saved results → {os.path.basename(cache_path)}")

    if WRITE_DATA and PS_STREAMING:
        # Streaming: PS trajectory already written inside run_ps_streaming_with_decimation
        with h5py.File(cache_path, "a") as f:
            import json
            f.attrs["params_json"] = json.dumps(params)

            meta_grp = f["meta"]
            meta_grp.attrs["timing_ps"] = timing["ps"]
            meta_grp.attrs["physical_time"] = float(physical_time)
            meta_grp.attrs["norm_time"] = float(norm_time)
            meta_grp.attrs["percent_c"] = float(v_si/spdlight)
            meta_grp.attrs["particle_label"] = (
                f"{KE_particle:.1e} eV electron"
                if mass_si == m_e else
                f"{KE_particle:.1e} eV proton"
                if mass_si == m_p else
                "manual"
            )

    stem = os.path.splitext(os.path.basename(cache_path))[0]

# === Timing Summary ===


print(f"Particle        : {KE_particle:.1e} eV {particle_type}")
if USE_RK45 and "rk45" in timing:
    print(f"Run Time RK45   : {timing['rk45']:.2f} s")
if USE_RK4 and "rk4" in timing:
    print(f"Run Time RK4    : {timing['rk4']:.2f} s")
if USE_RKG and "rkg" in timing:
    print(f"Run Time RKG    : {timing['rkg']:.2f} s")    
if "ps" in timing:
    print(f"Run Time PS     : {timing['ps']:.2f} s")

print(f"Norm Time       : {norm_time:.2e} ")
print(f"Physical Time   : {physical_time:.2e} s")
if orders_used is not None:
    print(f"PS Orders       : max={orders_used.max()}, mean={orders_used.mean():.1f}")
print(f"% of c          : {v_si/spdlight:.8f}")


# =====================================================
# ============== Full 2D Trajectory Plot ==============
# =====================================================
plotbounds=x_initial+ 1.1 

if USE_FULL_PLOT:
# === Plot Trajectories ===
    fig, ax = plt.subplots(figsize=(10, 8))

    if USE_RK45:
        ax.plot(solution_rk45.y[0], solution_rk45.y[1], label='RK45', color='#E69F00', linestyle='--')
    if USE_RK4:
        ax.plot(solution_rk4[0], solution_rk4[1], label='RK4', alpha=0.8, color='#CC79A7', linestyle='-.')
    if USE_RKG:
        ax.plot(solution_rkg[:, 0], solution_rkg[:, 1], label='RKG', alpha=0.8, color='#CC0000', linestyle='-.')
    ax.plot(solution_ps[0], solution_ps[1], label=f"PS{orders_used.max()}", alpha=0.7, color='#009E73', linestyle=':')

    # === Formatting ===
    ax.set_xlabel(r"x ($R_E$)")
    ax.set_ylabel(r"y ($R_E$)")
    if USE_PLOT_TITLES: ax.set_title(f"2D {particle_type} Trajectory in Dipole B Field")

    ax.legend(loc="upper right")
    ax.set_xlim(-plotbounds, plotbounds)
    ax.set_ylim(-plotbounds, plotbounds)
    ax.set_aspect('equal', adjustable='box')
    # ax.axis('equal')
    ax.grid(True)

    # === Save and Close ===
    fig.canvas.draw()   
    fig.savefig( f"{output_folder}/{stem}_DipoleB_{particle_type}_{KE_particle:.1e}eV_{ps_step}step_PS{orders_used.max()}_pitch{pitch_deg}_phi{phi_deg}_{norm_time:.2e}s_{npfloat.__name__}_2D.png", dpi=600, bbox_inches="tight")
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
    ax.plot(solution_ps[0], solution_ps[1], solution_ps[2], label=f"PS{orders_used.max()}", alpha=0.6, color='#009E73', linestyle=':')


    ax.set_xlim(-plotbounds, plotbounds)
    ax.set_ylim(-plotbounds, plotbounds)
    ax.set_zlim(-plotbounds, plotbounds)

    # === Labels and Legend ===
    ax.set_xlabel(r'x ($R_E$)')
    ax.set_ylabel(r'y ($R_E$)')
    ax.set_zlabel(r'z ($R_E$)')
    if USE_PLOT_TITLES: ax.set_title(f"3D {particle_type} Trajectory in Dipole B Field")
    ax.legend(loc="upper right")
    # plt.tight_layout()

    # === Save and Close ===
    fig.canvas.draw()   
    fig.savefig( f"{output_folder}/{stem}_DipoleB_{particle_type}_{KE_particle:.1e}eV_{ps_step}step_PS{orders_used.max()}_pitch{pitch_deg}_phi{phi_deg}_{norm_time:.2e}s_{npfloat.__name__}_3D.png", dpi=600, bbox_inches="tight")
    plt.close(fig) 


# ==========================================
# ================ Slicing  ================
# ==========================================
ps_x, ps_y, ps_z = slice_solution(t_eval_ps, solution_ps, window_duration, norm_time, mode=slice_mode)[:3]

if USE_RK45:
    rk45_x, rk45_y, rk45_z = slice_solution(t_eval_rk45, solution_rk45.y, window_duration, norm_time, mode=slice_mode)[:3]
if USE_RK4:
    rk4_x, rk4_y, rk4_z = slice_solution(t_eval_rk4, solution_rk4, window_duration, norm_time, mode=slice_mode)[:3]
if USE_RKG:
    rkg_x, rkg_y, rkg_z = slice_solution(t_eval_rkg, solution_rkg, window_duration, norm_time, mode=slice_mode)[:3]

# =====================================================
# ================ 2D Trajectory Slice ================
# =====================================================

if USE_FULL_PLOT:
    # === Plot Last Few Cycles ===
    fig, ax = plt.subplots(figsize=(10, 7))
    if USE_RK45:
        ax.plot(rk45_x, rk45_y, label='RK45', color='#E69F00', linestyle='--')
    if USE_RK4:
        ax.plot(rk4_x, rk4_y, label='RK4', alpha=0.8, color='#CC79A7', linestyle='-.')
    if USE_RKG:
        ax.plot(rkg_x, rkg_y, label='RKG', alpha=0.8, color='#CC0000', linestyle='-.')
    ax.plot(ps_x, ps_y, label=f"PS{orders_used.max()}", alpha=0.8, color='#009E73', linestyle=':')

    ax.set_xlabel(r"x ($R_E$)")
    ax.set_ylabel(r"y ($R_E$)")
    if USE_PLOT_TITLES: ax.set_title(f"2D Trajectory of Slice {particle_type} Orbits in Dipole B Field")
    # ax.set_xlim(-plotbounds, plotbounds)
    # ax.set_ylim(-plotbounds, plotbounds)
    # ax.set_aspect('equal', adjustable='box')
    ax.axis('equal')
    ax.legend(loc="upper right")
    ax.grid(True)


    # === Save and Close ===
    fig.canvas.draw()   
    fig.savefig( f"{output_folder}/{stem}_DipoleB_{particle_type}_{KE_particle:.1e}eV_{ps_step}step_PS{orders_used.max()}_pitch{pitch_deg}_phi{phi_deg}_{norm_time:.2e}s_{npfloat.__name__}_2Dslice.png", dpi=600, bbox_inches="tight")
    plt.close(fig) 


# =====================================================
# ================ 3D Trajectory Slice ================
# =====================================================
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot each trajectory segment
if USE_RK45:
    ax.plot(rk45_x, rk45_y, rk45_z, label='RK45', color='#E69F00', linestyle='--')
if USE_RK4:
    ax.plot(rk4_x, rk4_y, rk4_z, label='RK4', alpha=0.8, color='#CC79A7', linestyle='-.')
if USE_RKG:
    ax.plot(rkg_x, rkg_y, rkg_z, label='RKG', alpha=0.8, color='#CC0000', linestyle='-.')
ax.plot(ps_x, ps_y, ps_z, label=f"PS{orders_used.max()}", alpha=0.8, color='#009E73', linestyle=':')

ax.set_xlim(-plotbounds, plotbounds)
ax.set_ylim(-plotbounds, plotbounds)
ax.set_zlim(-plotbounds, plotbounds)
ax.legend(loc="upper right")
ax.grid(True)


ax.set_xlabel('x ($R_E$)')
ax.set_ylabel('y ($R_E$)')
ax.set_zlabel('z ($R_E$)')
if USE_PLOT_TITLES: ax.set_title(f'3D Trajectory of Slice {particle_type} Orbits in Dipole B Field')
ax.legend(loc="upper right")
# plt.tight_layout()

# === Save and Close ===
fig.canvas.draw()   
fig.savefig( f"{output_folder}/{stem}_DipoleB_{particle_type}_{KE_particle:.1e}eV_{ps_step}step_PS{orders_used.max()}_pitch{pitch_deg}_phi{phi_deg}_{norm_time:.2e}s_{npfloat.__name__}_3Dslice.png", dpi=600, bbox_inches="tight")
plt.close(fig)  

# =====================================================
# ============== KE Relative Error Plot ===============
# =====================================================

time_factor = 1.0 / (2.0 * np.pi)  # to convert to gyroperiods if desired

v_ps = solution_ps[3:6]   
E_ps = npfloat(0.5) * np.sum(v_ps**2, axis=0, dtype=npfloat)
rel_drift_ps = np.abs(E_ps - E_ps[0]) / E_ps[0]

final_ps = rel_drift_ps[-1]

# === If using Hamiltonian in RKG ==
if USE_RKG:
    r_rkg = solution_rkg[:, 0:3]
    p_rkg = solution_rkg[:, 3:6]
    A_rkg = np.zeros_like(r_rkg)
    for i in range(len(r_rkg)):
        A_rkg[i] = vector_potential_dipole(r_rkg[i])
    v_rkg = p_rkg - A_rkg
    E_rkg = npfloat(0.5) * np.sum(v_rkg**2, axis=1, dtype=npfloat)
    rel_drift_rkg = np.abs(E_rkg - E_rkg[0]) / E_rkg[0]
    ratio_rkg_ps = rel_drift_rkg[-1]/final_ps
    order_mag_rkg = int(np.floor(np.log10(abs(ratio_rkg_ps))))

# === If using Lorentz force in RKG ==
# r_rkg = solution_rkg[:, 0:3]
# v_rkg = solution_rkg[:, 3:6]  


if USE_RK45:
    v_rk45 = solution_rk45.y[3:6]
    E_rk45 = 0.5 * np.sum(v_rk45**2, axis=0)
    rel_drift_rk45 = np.abs(E_rk45 - E_rk45[0]) / E_rk45[0]
    ratio_rk45_ps = rel_drift_rk45[-1]/final_ps
    order_mag_rk45 = int(np.floor(np.log10(abs(ratio_rk45_ps))))


if USE_RK4:
    v_rk4 = solution_rk4[3:6]  
    E_rk4 = npfloat(0.5) * np.sum(v_rk4**2, axis=0, dtype=npfloat)
    rel_drift_rk4 = np.abs(E_rk4 - E_rk4[0]) / E_rk4[0]
    ratio_rk4_ps = rel_drift_rk4[-1]/final_ps
    order_mag_rk4 = int(np.floor(np.log10(abs(ratio_rk4_ps))))
    
# === Plot =====
fig, ax = plt.subplots(figsize=(10, 5))
if USE_RK45:
    lnrk45, = ax.semilogy(t_eval_rk45
    [1:]*time_factor, np.abs(rel_drift_rk45[1:]), label='RK45', color='#E69F00', linestyle='--')
if USE_RK4:
    lnrk4, = ax.semilogy(t_eval_rk4[1:]*time_factor, np.abs(rel_drift_rk4[1:]), label='RK4', alpha=0.8, color='#CC79A7', linestyle='-.')
if USE_RKG:
    lnrkg, = ax.semilogy(t_eval_rkg[1:]*time_factor, np.abs(rel_drift_rkg[1:]), label='RKG', alpha=0.8, color='#CC0000', linestyle='-.')
lnps, = ax.semilogy(t_eval_ps[1:]*time_factor, np.abs(rel_drift_ps[1:]), label=f"PS{orders_used.max()}", alpha=0.8, color='#009E73', linestyle=':')

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

ax.set_xlabel(r"\tau/T")
ax.set_ylabel(r"$|\Delta E|/E_0$")

if USE_PLOT_TITLES: ax.set_title(f"{particle_type} Relative Kinetic Energy Error in Dipole B Field")
# fig.tight_layout()

fig.subplots_adjust(right=0.9)
fig.canvas.draw()
ax_pos = ax.get_position()  # Bbox in figure coords
x_fig_label = ax_pos.x1   # a small gap to the right of axes

endpoints = []
if USE_RK45:
    endpoints.append((t_eval_rk45[-1], np.abs(rel_drift_rk45[-1]), "RK45", lnrk45.get_color()))
if USE_RK4:
    endpoints.append((t_eval_rk4[-1], np.abs(rel_drift_rk4[-1]), "RK4", lnrk4.get_color()))
    # endpoints.append((t_eval_rk4[-1], np.abs(rel_drift_rk4[-1]), f"RK4", lnrk4.get_color()))
if USE_RKG:
    endpoints.append((t_eval_rkg[-1], np.abs(rel_drift_rkg[-1]), "RKG", lnrkg.get_color()))
endpoints.append((t_eval_ps[-1], np.abs(rel_drift_ps[-1]), "PS", lnps.get_color()))


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


# === Save and Close ===
fig.canvas.draw()   
fig.savefig( f"{output_folder}/{stem}_DipoleB_{particle_type}_{KE_particle:.1e}eV_{ps_step}step_PS{orders_used.max()}_pitch{pitch_deg}_phi{phi_deg}_{norm_time:.2e}s_{npfloat.__name__}_KEerror.png", dpi=600, bbox_inches="tight")
plt.close(fig)  

# ============================================================
# ================ Magnetic Moment Deviations ================
# ============================================================
finalnum = max(1, int(steps_ps/PS_DECIMATE * (.9))) 

if USE_RKG:
    r_rkg = solution_rkg[:, 0:3]
    p_rkg = solution_rkg[:, 3:6]
    A_rkg = np.zeros_like(r_rkg)
    for i in range(len(r_rkg)):
        A_rkg[i] = vector_potential_dipole(r_rkg[i])  # already normalized

    v_rkg = p_rkg - A_rkg
    state_rkg = np.hstack((r_rkg, v_rkg))
    mu_rkg = compute_mu_rk(state_rkg, mass)
    mu0_rkg = mu_rkg[0]
    mudrift_rkg = np.abs(mu_rkg - mu0_rkg)/mu0_rkg

if USE_RK45:
    mu_rk45 = compute_mu_rk(solution_rk45.y.T, mass)
    mu0_rk45 = mu_rk45[0]
    mudrift_rk45 = np.abs(mu_rk45 - mu0_rk45)/mu0_rk45

if USE_RK4:
    mu_rk4 = compute_mu_rk(solution_rk4.T, mass)
    mu0_rk4  = mu_rk4[0]
    mudrift_rk4  = np.abs(mu_rk4  - mu0_rk4)/mu0_rk4

mu_ps = compute_mu_ps(solution_ps, mass)
mu0_ps   = mu_ps[0]
mudrift_ps   = np.abs(mu_ps   - mu0_ps)/mu0_ps

fig, ax = plt.subplots(figsize=(10, 5))
if USE_RK45:
    lnrk45, = ax.semilogy(t_eval_rk45[finalnum:]*time_factor, np.abs(mudrift_rk45[finalnum:]), label='RK45', color='#E69F00', linestyle='--')
if USE_RK4:
    lnrk4, = ax.semilogy(t_eval_rk4[finalnum:]*time_factor, np.abs(mudrift_rk4[finalnum:]), label='RK4', alpha=0.8, color='#CC79A7', linestyle='-.')
if USE_RKG:
    lnrkg, = ax.semilogy(t_eval_rkg[finalnum:]*time_factor, np.abs(mudrift_rkg[finalnum:]), label='RKG', alpha=0.8, color='#CC0000', linestyle='-.')
lnps, = ax.semilogy(t_eval_ps[finalnum:]*time_factor, np.abs(mudrift_ps[finalnum:]), label=f"PS{orders_used.max()}", alpha=0.8, color='#009E73', linestyle=':')

# Getting log lines to work, mess with at your own risk
ax.margins(x=0.01)
ax.set_yscale('log') 
# ax.set_xscale('log') 
ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=100))
ax.yaxis.set_major_formatter(LogFormatterSciNotation(base=10.0))  
ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=[]))
ax.yaxis.set_minor_formatter(NullFormatter())
# ax.xaxis.set_major_locator(LogLocator(base=10.0, numticks=100))
# ax.xaxis.set_major_formatter(LogFormatterSciNotation(base=10.0))  
# ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=[]))
# ax.xaxis.set_minor_formatter(NullFormatter())
ax.grid(True, which='major', linestyle='--', linewidth=0.7)
# ax.yaxis.set_major_formatter(FuncFormatter(sparse_labels))
# ax.xaxis.set_major_formatter(FuncFormatter(sparse_labels))
# ax.set_aspect('equal', adjustable='box')

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.set_xlabel(r"\tau/T")
ax.set_ylabel(r"$|\Delta \mu|/\mu_0$")

if USE_PLOT_TITLES: ax.set_title(f"{particle_type} Magnetic Moment Deviations in Dipole B Field")
# fig.tight_layout()

fig.subplots_adjust(right=0.9)
fig.canvas.draw()
ax_pos = ax.get_position()  # Bbox in figure coords
x_fig_label = ax_pos.x1   # a small gap to the right of axes

endpoints = []
if USE_RK45:
    endpoints.append((t_eval_rk45[-1], np.abs(mudrift_rk45[-1]), "RK45", lnrk45.get_color()))
if USE_RK4:
    endpoints.append((t_eval_rk4[-1], np.abs(mudrift_rk4[-1]), "RK4", lnrk4.get_color()))
if USE_RKG:
    endpoints.append((t_eval_rkg[-1], np.abs(mudrift_rkg[-1]), "RKG", lnrkg.get_color()))
endpoints.append((t_eval_ps[-1], np.abs(mudrift_ps[-1]), f"PS{orders_used.max()}", lnps.get_color()))

labels = []
for x, y, label, color in endpoints:
    _, fy = data_to_fig(x, y, ax, fig)
    # Clamp to axis bounds
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
fig.savefig( f"{output_folder}/{stem}_DipoleB_{particle_type}_{KE_particle:.1e}eV_{ps_step}step_PS{orders_used.max()}_pitch{pitch_deg}_phi{phi_deg}_{norm_time:.2e}s_{npfloat.__name__}_mu.png", dpi=600, bbox_inches="tight")
plt.close(fig)  

# ===================================================
# ================ Mirror and Drift  ================
# ===================================================
idxs, crossings_tau = mirror_times_from_PS(solution_ps, ps_step, interp=True, min_gap = user_min_gap)
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
    final_coeff_matrix=solution_ps,
    dt_tau=ps_step,
    mirror_times_tau=crossings_tau if has_mirrors else None,
    sample='mirrors' if has_mirrors else 'raw',
    time_scale_sec=tau_time,
    min_phase_rad = user_min_phase, 
    return_details=False
)

# 3) Report (prefer crossings-mean, fallback to slope-fit)
T_drift_s   = drift_stats["period_s_mean"] if drift_stats["period_s_mean"] is not None else drift_stats["period_s_fit"]
T_drift_tau = drift_stats["period_tau_mean"] if drift_stats["period_tau_mean"] is not None else drift_stats["period_tau_fit"]
direction   = drift_stats["direction"]

if T_drift_s is None:
    print("Drift period: not enough azimuthal motion to estimate (yet).")
else:
    print(f"Drift period ≈ {T_drift_s:.6g} s  (≈ {T_drift_tau:.6g} (normalized time), direction {'east' if direction>0 else 'west'})")


# ====================================
# === Write Summary Output to File ===
# ====================================

finalnum = max(1, int(steps_ps * 0.01))  # Number of steps to average over, last 1%

def summarize_error(label, err, f):
    mean_val = np.mean(err[-finalnum:])
    max_val  = np.max(np.abs(err[-finalnum:]))
    rms_val  = np.sqrt(np.mean(err[-finalnum:]**2))
    f.write(f"  {label:<8}: mean = {mean_val:.2e}, max = {max_val:.2e}, rms = {rms_val:.2e}\n")


output_filename = f"{output_folder}/{stem}_DipoleB_{particle_type}_{KE_particle:.1e}eV_{ps_step}step_PS{orders_used.max()}_pitch{pitch_deg}_phi{phi_deg}_{norm_time:.2e}s_{npfloat.__name__}_simulation_summary.txt"


with open(output_filename, "w") as f:
    if WRITE_DATA or READ_DATA: f.write(f"Run Data: {stem}.hd5\n\n")
    f.write("=== Simulation Summary ===\n")
    f.write(f"Initial Conditions:\n")
    f.write(f"  particle      = {particle_type}\n")
    f.write(f"  mass          = {mass_si} kg\n")
    f.write(f"  rel mass      = {mass} kg\n")
    f.write(f"  Energy        = {KE_particle} eV\n")
    f.write(f"  pitch_deg     = {pitch_deg}°\n")
    f.write(f"  phi_deg       = {phi_deg}°\n")
    f.write(f"  tau           = {tau_time} s\n")
    f.write(f"  v_tau         = {v_tau} RE/tau0\n")
    f.write(f"  x_initial     = {x_initial} RE\n")
    f.write(f"  y_initial     = {y_initial} RE\n")
    f.write(f"  z_initial     = {z_initial} RE\n")
    f.write(f"  vx_initial    = {vx_initial} RE/tau0\n")
    f.write(f"  vy_initial    = {vy_initial} RE/tau0\n")
    f.write(f"  vz_initial    = {vz_initial} RE/tau0\n")
    f.write(f"  Initial Bfield= {B_0} T\n")
    
    f.write(f"  float type    = {npfloat.__name__}\n\n")
    
    f.write("=== Bounce and Drift ===\n")
    f.write(f"  Mirror crossings          = {bounce_stats['n_crossings']}\n")
    if bounce_stats['full_mean_s'] is not None:
        f.write(f"  Full bounce period (mean) = {bounce_stats['full_mean_s']:.6g} s\n")
    else:
        f.write("  Full bounce period (mean) = N/A (no bounce detected)\n")
    if drift_stats.get("period_s_fit") is not None:
        f.write(f"  Drift period (fit)        = {drift_stats['period_s_fit']:.6g} s  "
                f"(direction: {'eastward' if drift_stats['direction']>0 else 'westward'})\n")
    else:
        f.write("  Drift period (fit)        = N/A (not enough drift detected)\n")

    f.write("=== Timing Summary ===\n")
    if USE_RK45:
        f.write(f"  Run Time RK45 = {timing['rk45']:.2f} s\n")    
    if USE_RK4:
        f.write(f"  Run Time RK4  = {timing['rk4']:.2f} s\n")
    if USE_RKG:  
        f.write(f"  Run Time RKG  = {timing['rkg']:.2f} s\n")
    f.write(f"  Run Time PS   = {timing['ps']:.2f}  s\n")
    f.write(f"  PS Orders     = max={orders_used.max()}, mean={orders_used.mean():.1f}\n")
    f.write(f"  norm_time     = {norm_time}\n")
    f.write(f"  physical time = {physical_time}\n")
    if USE_RK4:
        f.write(f"  rk4 step size = {rk4_step}\n")
    if USE_RKG:
        f.write(f"  rkg step size = {rkg_step}\n")
    f.write(f"  ps step size  = {ps_step}\n")
    if USE_RK4:
        f.write(f"  rk4 steps     = {steps_rk4}\n")
    if USE_RKG:
        f.write(f"  rkg steps     = {steps_rkg}\n")
    f.write(f"  ps steps      = {steps_ps}\n\n")

    f.write(f"=== |detla E|/E0 (relative, last {finalnum} steps)===\n")
    if USE_RK45:
        summarize_error("RK45", rel_drift_rk45, f)
    if USE_RK4:
        summarize_error("RK4",  rel_drift_rk4, f)
    if USE_RKG:
        summarize_error("RKG",  rel_drift_rkg, f)
    summarize_error("PS",   rel_drift_ps,  f)

    f.write(f"\n=== |delta mu|/mu0(relative, last {finalnum} steps)===\n")
    if USE_RK45:
        summarize_error("RK45", mudrift_rk45, f)
    if USE_RK4:
        summarize_error("RK4",  mudrift_rk4, f)
    if USE_RKG:
        summarize_error("RKG",  mudrift_rkg, f)
    summarize_error("PS",   mudrift_ps,  f)

# === Shared metadata ===
run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
x_0, y_0, z_0 = x_initial, y_initial, z_initial
steps = steps_ps
dt = ps_step
last_n = max(1, int(0.01 * steps))

# === Error summarizer ===
def summarize(err):
    return {
        "mean": np.mean(err[-last_n:]),
        "max":  np.max(np.abs(err[-last_n:])),
        "rms":  np.sqrt(np.mean(err[-last_n:]**2))
    }

# === Collect rows (one per method) ===
def make_record(method, e_drift, mu_drift):
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
