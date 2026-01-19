from project_setup import * 
logger = setup_logger("dipole_logger", "dipole_chunk.log", level=logging.DEBUG)

legacy_h5_path = None  # hard disable for most runs,
DEBUG = True
if DEBUG: tracemalloc.start()

# Run load, Allow command-line override
run = "demo"   # key options: "demo", "paper1", "paper2", "paper3", unless a new input is made. Demo mode is a quick test run. Paper modes can take upwards of half an hour. 
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

# ===============================================
# ============= Legacy File Load ================
# ===============================================
"""
this allows legacy files to be loaded directly through the 'legacy' run in the test particle function, 
early runs didn't have all the parameters we are not tracking so the scanning doesn't work. The 
functions take the old h5 files we did have and reconstructions a dictionary in the format we are using now.
"""
USE_LEGACY_FILE = legacy_h5_path is not None and os.path.exists(legacy_h5_path)
if USE_LEGACY_FILE:
    cache_path = legacy_h5_path
    print(f"You have loaded a LEGACY file: {cache_path} — loading.\n")
    summary, datasets, params, h5_handle = load_legacy_file(cache_path)
    
    mass_si = summary["meta"]["mass_si"]
    q_e = summary["meta"]["q_e"]
    B_0 = summary["meta"]["B0_T"]
    x_initial = summary["meta"]["x0"]
    y_initial = summary["meta"]["y0"]
    z_initial = summary["meta"]["z0"]
    pitch_deg = summary["meta"]["pitch_deg"]
    phi_deg = summary["meta"]["phi_deg"]
    norm_time = summary["meta"]["norm_time"]
    KE_particle = summary["meta"]["energy_eV"]
    USE_PS= summary["ps"]["enabled"]
    USE_RK4= summary["rk4"]["enabled"]
    USE_RK45= summary["rk45"]["enabled"]
    USE_RKG= summary["rkg"]["enabled"]

    T_gyro = 2.0 * np.pi * (x_initial**3) 

    timing = summary["meta"]["timing"]
    stem = summary["meta"]["stem"]

    if USE_PS:
        ps_step= summary["ps"]["dt"]
        steps_ps = summary["ps"]["steps"]
        PS_decimate = summary["ps"]["decimate"]
        gyroperiods= npfloat(steps_ps) / T_gyro
        N_STEPS_PER_GYRO_ps = summary["ps"]["numberstepspergyro"]
        PS_CHUNKING=summary["ps"]["streaming"]
        solution_ps = datasets["ps_y"][()]
        orders_used = datasets["ps_orders"][()]

    if USE_RK4: 
        solution_rk4 = datasets["rk4_y"][()]
        steps_rk4 = summary["rk4"]["steps"]
        rk4_step= summary["rk4"]["dt"]
        N_STEPS_PER_GYRO_rk4 = summary["rk4"]["numberstepspergyro"]
    if USE_RK45:
        class _Obj: pass
        solution_rk45 = _Obj()
        solution_rk45.t = datasets["rk45_t"][()]
        solution_rk45.y = datasets["rk45_y"][()]
        solution_rk45.sol = None

    if USE_RKG: 
        solution_rkg = datasets["rkg_y"][()]
        steps_rkg = summary["rkg"]["steps"]
        rkg_step= summary["rkg"]["dt"]
        if not USE_PS: gyroperiods= npfloat(steps_rkg) / T_gyro
        N_STEPS_PER_GYRO_rkg = summary["rkg"]["numberstepspergyro"]

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
gyro_radius_RE=float(gyro_radius_si/RE)
initial_pos_vel = np.array([x_initial, y_initial, z_initial, vx_initial, vy_initial, vz_initial], dtype=npfloat)  

if DEBUG: 
    logger.info("Starting chunked dipole run.")
    logger.debug(f"Initial velocity: {vx_initial}, {vy_initial}, {vz_initial}")
    logger.debug(f"Initial position: {x_initial}, {y_initial}, {z_initial}")
    logger.debug(f"Initial gyroradius: {gyro_radius_RE}")

# --- Initial invariants for E0 and mu0 for h5 file ---
"""
To streamline memory for large files, rather than loading everything, we are often slicing out what we need
directly from the h5 file, this establishes the E0 and mu0 values for those calculations
"""
vx0, vy0, vz0 = initial_pos_vel[3:6]
E0_ps = npfloat(0.5) * (vx0*vx0 + vy0*vy0 + vz0*vz0)
y0_ps = np.zeros((17, 1), dtype=npfloat)
y0_ps[0:6, 0] = initial_pos_vel
x0, y0, z0 = initial_pos_vel[0:3]
r2 = x0*x0 + y0*y0 + z0*z0
r5inv = r2**(-2.5)
y0_ps[14, 0] = -3 * x0 * z0 * r5inv
y0_ps[15, 0] = -3 * y0 * z0 * r5inv
y0_ps[16, 0] = -(3*z0*z0 - r2) * r5inv
mu0_ps = compute_mu_ps(y0_ps, mass)[0]

# === Build parameter tracer & check cache ===
"""
this is scanning the files already stored to see if we already have the data, beware that these files can 
be GB size for dipole. Similarly it saves a filebased on these specific parameters so if you toggle one 
right now, even save chunking or decimate it will run as a new file with a new name
"""
if not USE_LEGACY_FILE:
    params = get_run_params(USE_RK45, USE_RK4, USE_RKG, USE_PS, PS_decimate, PS_CHUNKING,   # parameters it is scanning
                    mass_si, q_e, B_0, gamma, user_min_phase,
                    x_initial, y_initial, z_initial,
                    pitch_deg, phi_deg,
                    norm_time, ps_step, rk4_step, rkg_step,
                    PS_order, tol, qoverm, rtol_rk45, atol_rk45)
    cache_path = h5_path_for(params, run_storage)

    if os.path.exists(cache_path) and READ_DATA:
        print(f"Found existing results: {os.path.basename(cache_path)} — loading.\n")

        with h5py.File(cache_path, "r") as cached:

            # creating summary dictionary, legacy files should create a similar dictionary now
            if "summary_json" not in cached.attrs:
                raise RuntimeError(
                    "Cached file missing summary_json. "
                    "This file was written by an older version."
                )

            summary = json.loads(cached.attrs["summary_json"])

            # ---- meta ----
            meta = summary["meta"]
            timing = meta["timing"]

            stem = meta["stem"]
            particle_type = meta["particle"]
            KE_particle = meta["energy_eV"]
            pitch_deg = meta["pitch_deg"]
            phi_deg = meta["phi_deg"]
            x_initial = meta["x0"]
            y_initial = meta["y0"]
            z_initial = meta["z0"]
            B_0 = meta["B0_T"]
            gyroperiods = meta["gyroperiods"]
            norm_time = meta["norm_time"]
            npfloat = np.dtype(meta["dtype"]).type  # optional

            # ---- PS config ----
            ps_cfg = summary["ps"]
            USE_PS = ps_cfg["enabled"]
            PS_CHUNKING = ps_cfg["streaming"]
            ps_step = ps_cfg["dt"]
            steps_ps = ps_cfg["steps"]
            PS_decimate = ps_cfg["decimate"]
            PS_chunk_steps = ps_cfg["chunksize"]
            N_STEPS_PER_GYRO_ps = ps_cfg["numberstepspergyro"]
            max_ps_value = ps_cfg["max_ps"]
            E0_ps = ps_cfg["E0"]
            mu0_ps = ps_cfg["mu0"]

            # ---- RK4 config ----
            rk4_cfg = summary["rk4"]
            USE_RK4 = rk4_cfg["enabled"]
            rk4_step = rk4_cfg["dt"]
            steps_rk4 = rk4_cfg["steps"]
            N_STEPS_PER_GYRO_rk4 = rk4_cfg["numberstepspergyro"]


            # ---- RK45 config ----
            rk45_cfg = summary["rk45"]
            USE_RK45 = rk45_cfg["enabled"]
            rtol_rk45 = rk45_cfg["rtol"]
            atol_rk45 = rk45_cfg["atol"]

            # ---- RKG config ----
            rkg_cfg = summary["rkg"]
            USE_RKG = rkg_cfg["enabled"]
            rkg_step = rkg_cfg["dt"]
            steps_rkg = rkg_cfg["steps"]
            N_STEPS_PER_GYRO_rkg = rkg_cfg["numberstepspergyro"]


            # ---- Load solver data ------
            """
            Earlier editions of the code loaded everything into memory, for extended runs this has become untenable. 
            Chunking allows files to be written and read in chunks which takes far less memory. However, the option 
            to still run the original way is left for now, in case we find need for it. Note, right now ONLY PS method 
            does the chunking method. I have not tried to apply it to RK method until we find specific needs.
            """

            # === PS ===
            if USE_PS and "ps" in cached:
                ps_group = cached["ps"]

                if PS_CHUNKING:
                    solution_ps = None
                    orders_used = None
                else:  # this is memory intensive for long PS runs, I recommend working in PS_CHUNKING from the start
                    solution_ps = ps_group["y"][()]
                    orders_used = ps_group["orders"][()] if "orders" in ps_group else None

            # === RK4 ===
            if USE_RK4 and "rk4" in cached:
                solution_rk4 = cached["rk4"]["y"][()]

            # === RK45 ===
            if USE_RK45 and "rk45" in cached:
                class _Obj: pass
                solution_rk45 = _Obj()
                solution_rk45.t = cached["rk45"]["t"][()]
                solution_rk45.y = cached["rk45"]["y"][()]
                solution_rk45.sol = None

            # === RKG ===
            if USE_RKG and "rkg" in cached:
                solution_rkg = cached["rkg"]["y"][()]
    else:
        print("No matching file or 'Read Data' skipped. Running solvers...\n")

        # Common grid size (used by RK45, PS needs to be enabled)
        steps_ps = int(norm_time / ps_step)
        # ====== Run PS ======
        max_ps = None
        if USE_PS:
            start_time_ps = time.time()

            if PS_CHUNKING:
                max_ps, elapsed_ps = run_ps_streaming_with_decimation(
                    initial_pos_vel_ps=initial_pos_vel,
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
                    N_STEPS_PER_GYRO_ps=N_STEPS_PER_GYRO_ps,
                    user_min_phase=user_min_phase,
                )
                solution_ps = None
                orders_used = None
            else:
                solution_ps, orders_used = PS_dipoleB(
                    PS_order, steps_ps, initial_pos_vel, tol, qoverm, ps_step
                )
            end_time_ps = time.time() 

        # ====== Run RK45 ======
        if USE_RK45:
            start_time_rk45 = time.time()
            t_common = ps_step * np.arange(steps_ps + 1, dtype=npfloat)
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
                "particle": particle_type,
                "mass_si": mass_si,
                "q_e": q_e,
                "energy_eV": npfloat(KE_particle),   
                "pitch_deg": npfloat(pitch_deg),
                "phi_deg": npfloat(phi_deg),
                "x0": npfloat(x_initial),
                "y0": npfloat(y_initial),
                "z0": npfloat(z_initial),
                "B0_T": npfloat(B_0),
                "gyroperiods": npfloat(gyroperiods),
                "dtype": npfloat.__name__,  
            }
        }

        if USE_PS:
            if PS_CHUNKING:
                max_ps_value = int(max_ps) if max_ps is not None else None
            else:
                max_ps_value = int(orders_used.max()) if orders_used is not None else None
        else:
            max_ps_value = None
        
        results["ps"] = { "enabled": bool(USE_PS),}
        if USE_PS:
            results["ps"].update({
                "y": solution_ps if not PS_CHUNKING else None,
                "orders": orders_used if not PS_CHUNKING else None,
                "ordercap": PS_order,
                "max_ps": max_ps_value,
                "numberstepspergyro": N_STEPS_PER_GYRO_ps,
                "dt": ps_step,
                "steps": steps_ps,
                "streaming": PS_CHUNKING,
                "chunksize": PS_chunk_steps,
                "decimate": PS_decimate,
                "tol": tol,
                "minphase" : user_min_phase,
                "E0": float(E0_ps),
                "mu0": float(mu0_ps),
                "t0": 0.0,
            })
            results["meta"]["timing"]["ps"] = end_time_ps - start_time_ps

        results["rk4"] = { "enabled": bool(USE_RK4),}
        if USE_RK4:
            results["rk4"].update({
                "y": solution_rk4,
                "numberstepspergyro": N_STEPS_PER_GYRO_rk4,
                "dt": npfloat(rk4_step),
                "steps": int(steps_rk4),
                "t0": 0.0,
            })
            results["meta"]["timing"]["rk4"] = end_time_rk4 - start_time_rk4

        results["rk45"] = { "enabled": bool(USE_RK45),}
        if USE_RK45:
            results["rk45"].update({
                "y": solution_rk45.y,
                "t": solution_rk45.t,
                "rtol": rtol_rk45,
                "atol": atol_rk45,
            })
            results["meta"]["timing"]["rk45"] = end_time_rk45 - start_time_rk45

        results["rkg"] = { "enabled": bool(USE_RKG),}
        if USE_RKG:
            results["rkg"].update({
                "y": solution_rkg,
                "numberstepspergyro": N_STEPS_PER_GYRO_rkg,
                "dt": npfloat(rkg_step),
                "steps": int(steps_rkg),
                "t0": 0.0
            })
            results["meta"]["timing"]["rkg"] = end_time_rkg - start_time_rkg

        # =========================
        # ====== Save Results =====
        # =========================
        stem = os.path.splitext(os.path.basename(cache_path))[0]
        timing = results["meta"]["timing"]
        results["meta"]["stem"]=stem
        if WRITE_DATA:
            summary = {
                "meta": {
                    "stem": stem,
                    "particle": particle_type,
                    "mass_si": mass_si,
                    "q_e": q_e,
                    "energy_eV": float(KE_particle),
                    "pitch_deg": float(pitch_deg),
                    "phi_deg": float(phi_deg),
                    "x0": float(x_initial),
                    "y0": float(y_initial),
                    "z0": float(z_initial),
                    "B0_T": float(B_0),
                    "gyroperiods": float(gyroperiods),
                    "norm_time": float(norm_time),
                    "physical_time": float(physical_time),
                    "percent_c": float(v_si/spdlight),
                    "qoverm": float(qoverm),
                    "dtype": npfloat.__name__,
                    "timing": results["meta"]["timing"],
                },
                "ps": {
                    "enabled": USE_PS,
                    "dt": ps_step if USE_PS else None,
                    "steps": steps_ps if USE_PS else None,
                    "streaming": PS_CHUNKING if USE_PS else None,
                    "ordercap": PS_order if USE_PS else None,
                    "max_ps": max_ps_value,
                    "chunksize": PS_chunk_steps if (USE_PS and PS_CHUNKING) else None,
                    "decimate": PS_decimate if USE_PS else None,
                    "numberstepspergyro": N_STEPS_PER_GYRO_ps if USE_PS else None,
                    "E0": float(E0_ps) if USE_PS else None,
                    "mu0": float(mu0_ps) if USE_PS else None,
                    "minphase": user_min_phase if USE_PS else None,
                    "tol": float(tol)
                },
                "rk4": {
                    "enabled": USE_RK4,
                    "dt": float(rk4_step) if USE_RK4 else None,
                    "steps": int(steps_rk4) if USE_RK4 else None,
                    "numberstepspergyro": N_STEPS_PER_GYRO_rk4 if USE_RK4 else None,
                },
                "rk45": {
                    "enabled": USE_RK45,
                    "rtol": rtol_rk45 if USE_RK45 else None,
                    "atol": atol_rk45 if USE_RK45 else None,
                },
                "rkg": {
                    "enabled": USE_RKG,
                    "dt": float(rkg_step) if USE_RKG else None,
                    "steps": int(steps_rkg) if USE_RKG else None,
                    "numberstepspergyro": N_STEPS_PER_GYRO_rkg if USE_RKG else None,
                },
            }

            # ====== h5 file creation =============
            if USE_PS and PS_CHUNKING:
                append_results_h5(cache_path, results, summary)
                print(f"Updated streamed file → {os.path.basename(cache_path)}")
            else:
                save_results_h5(cache_path, results, summary)
                print(f"Saved results → {os.path.basename(cache_path)}")


if DEBUG: 
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    logger.info(f"Peak memory usage for load/write h5: {peak / 1024**2:.2f} MB\n")

# ==================================
# ==== Dictionary of run params ====
# ==================================
"""
These are the plotting parameters, these can be varied without impacting the h5 file or scanned parameters
and are saved to the summary text file. They are not appended to the h5 file though and should not be as the 
raw date remains unchanged
"""


summary["plot"] = {
    "trajwindow_s": window_time,
    "slicemode": slice_mode,
    "NGYRO" : N_GYRO,
    "gyroslice": gyro_window,
    "maxplotpoints": MAX_PLOT_POINTS,
    "externalps": external_h5_ps if USE_EXTERNAL_H5_ps else None,
    "externalrk4": external_h5_rk4 if USE_EXTERNAL_H5_rk4 else None,
    "externalrk45": external_h5_rk45 if USE_EXTERNAL_H5_rk45 else None,
    "externalrkg": external_h5_rkg if USE_EXTERNAL_H5_rkg else None,
}

# ===============================
# Build RK45 solution on PS grid 
# ===============================
"""
this is building RK45 time base for points we want on PS grid. Not meant for long runs
as this can be a memory hog but rk45 is not great on long runs anyways
"""
if USE_RK45 and not USE_PS:
    raise RuntimeError(
        "RK45 requires USE_PS=True in this workflow; it builds a grid to match PS."
    )

if USE_RK45:
    y_rk45_common = solution_rk45.y

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
        n_ps = steps_ps
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
"""
Can be data heavy for long runs right now. Could be refactored similar to KE error section in the future
if need was high enough.
"""

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
    ax.set_xlabel(r"x")
    ax.set_ylabel(r"y")
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
    fig_path_2D = build_figure_filename( summary , output_folder , stem , figure_tag="2D", ext="png")
    plt.savefig(fig_path_2D, dpi=600, bbox_inches="tight") 
    plt.close(fig)  


# =====================================================
# ============== Full 3D Trajectory Plot ==============
# =====================================================
"""
Can be data heavy for long runs right now. Could be refactored similar to KE error section in the future
if need was high enough.
"""

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
    fig_path_3D = build_figure_filename( summary , output_folder , stem , figure_tag="3D", ext="png")
    plt.savefig(fig_path_3D, dpi=600, bbox_inches="tight") 
    plt.close(fig) 

# ========================================================================
# ================ Creating Plot Window (slice of time) ==================
# ========================================================================
"""
Generally only interested in a specific window of time for a run, like 'first' and 'last' parts of the run. Test particle
file lets you specify in physical seconds how big you want this window to be via window_time. Generally looking at a drift
or several bounce periods is useful. If you don't know or have these numbers, they are an output of the calculations and after
completing the initial run, you can use this information to adjust plotting (no impact to h5 file creation).
"""
if DEBUG: tracemalloc.start()

if slice_mode == "first":
    t_start = 0.0
    t_end   = min(norm_time, window_duration)
elif slice_mode == "last":
    t_end   = norm_time
    t_start = max(0.0, norm_time - window_duration)
else:
    raise ValueError("slice_mode must be 'first' or 'last'")


# =========== PS Window Load ===============


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

    # ------Load ONLY the window--------

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
    # if USE_PS: x_end_time = j1 * ps_store_stride * ps_step
    # logger.debug(f"[Window] mode={slice_mode}, t_start={t_start:.6e}, t_end={t_end:.6e}")
    # if USE_PS: logger.debug(f"[PS] x_start={y_win[0,0]:.6e}, x_end={y_win[0,-1]:.6e}")

# =====================================================
# ================ 2D Trajectory Slice ================
# =====================================================
"Plots the window of time created above in 2D"

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
    fig_path_2Dslice = build_figure_filename( summary , output_folder , stem , figure_tag="2Dslice", ext="png")
    plt.savefig(fig_path_2Dslice, dpi=600, bbox_inches="tight") 
    plt.close(fig) 


# =====================================================
# ================ 3D Trajectory Slice ================
# =====================================================
"Plots the window of time created above in 3D"


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
fig_path_3Dslice = build_figure_filename( summary , output_folder , stem , figure_tag="3Dslice", ext="png")
plt.savefig(fig_path_3Dslice, dpi=600, bbox_inches="tight") 
plt.close(fig) 
 

# =====================================================
# ============== KE Relative Error Plot ===============
# =====================================================
"""
This section calculates the relative KE error plot over the entire run. This is done in chunks
"""

if DEBUG:
    tracemalloc.start()

equatorial_r3 = x_initial**3
time_factor = 1.0 / (2.0 * np.pi * equatorial_r3)  # to convert plots to gyroperiods
fig, ax = plt.subplots(figsize=(10, 5))

if USE_PS: energy_stride = max(1, n_ps // MAX_PLOT_POINTS)
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
fig_path_KEerror = build_figure_filename( summary , output_folder , stem , figure_tag="KEerror", ext="png")
plt.savefig(fig_path_KEerror, dpi=600, bbox_inches="tight") 
plt.close(fig)  

if DEBUG:
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    logger.info(f"Peak memory usage for KE analysis: {peak / 1024**2:.2f} MB")
    if USE_PS: midpoint_ps = int(round(len(rel_drift_ps) / 2))
    if USE_PS: logger.info(f"energy stride: {energy_stride}")
    if USE_PS: logger.debug(f"[PS] E rel drift initial ={rel_drift_ps[0]:.2e}, E rel drift mid ={rel_drift_ps[midpoint_ps]:.2e}, E rel drift final ={rel_drift_ps[-1]:.2e}")
    
    if USE_RK4: midpoint_rk4 = int(round(len(rel_drift_rk4) / 2))
    if USE_RKG: midpoint_rkg = int(round(len(rel_drift_rkg) / 2))
    if USE_RK45: midpoint_rk45 = int(round(len(rel_drift_rk45) / 2))
    
    if USE_RKG: logger.debug(f"[RKG] E rel drift initial ={rel_drift_rkg[0]:.2e}, E rel drift mid ={rel_drift_rkg[midpoint_rkg]:.2e}, E rel drift final ={rel_drift_rkg[-1]:.2e}")
    if USE_RK4: logger.debug(f"[RK4] E rel drift initial ={rel_drift_rk4[0]:.2e}, E rel drift mid ={rel_drift_rk4[midpoint_rk4]:.2e}, E rel drift final ={rel_drift_rk4[-1]:.2e}")
    if USE_RK45: logger.debug(f"[RK45] E rel drift initial ={rel_drift_rk45[0]:.2e}, E rel drift mid ={rel_drift_rk45[midpoint_rk45]:.2e}, E rel drift final ={rel_drift_rk45[-1]:.2e}")

# ============================================================
# ================ Magnetic Moment Deviations ================
# ============================================================
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

# ===== Plotting ========
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
fig_path_mu = build_figure_filename( summary , output_folder , stem , figure_tag="mu", ext="png")
plt.savefig(fig_path_mu, dpi=600, bbox_inches="tight") 
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

    if USE_PS: logger.info(f"moment stride: {stride}")
    if USE_PS: logger.debug(f"[PS] mu rel drift initial ={mudrift_ps[0]:.2e}, mu rel drift mid ={mudrift_ps[mumidpoint_ps]:.2e}, mu rel drift final ={mudrift_ps[-1]:.2e}")
    if USE_RKG: logger.debug(f"[RKG] mu rel drift initial ={mudrift_rkg[0]:.2e}, mu rel drift mid ={mudrift_rkg[mumidpoint_rkg]:.2e}, mu rel drift final ={mudrift_rkg[-1]:.2e}")
    if USE_RK4: logger.debug(f"[RK4] mu rel drift initial ={mudrift_rk4[0]:.2e}, mu rel drift mid ={mudrift_rk4[mumidpoint_rk4]:.2e}, mu rel drift final ={mudrift_rk4[-1]:.2e}")
    if USE_RK45: 
        logger.debug(f"[RK45] mu rel drift initial ={mudrift_rk45[0]:.2e}, mu rel drift mid ={mudrift_rk45[mumidpoint_rk45]:.2e}, mu rel drift final ={mudrift_rk45[-1]:.2e}")
        logger.debug(f"[RK45 Slice] t_start={t_rk45[0]:.3e}, t_end={t_rk45[-1]:.3e}, len={len(t_rk45)}")

# ===================================================
# ================ Mirror and Drift  ================
# ===================================================
"""
Similar to other sections of code, this is currently retaining the original (i.e. load from RAM) approach to
calculating drift and bounce as well as the new way utilizing chunking. Once we are certain that we do not need the original
method, the first part of 'if' statement can go and this can be reduced to "if USE_PS'. Note that drift and bounce are
only calcualted for PS method. 
"""
bounce_results = None
drift_results  = None

if USE_PS and not PS_CHUNKING:

    v_eps = npfloat(1e-14) * v_tau
    user_min_gap = max(3, int(0.5 * T_gyro / ps_step))

    ps_analysis = solution_ps
    dt_ps_eff = ps_step

    # --- Bounce ---
    idxs, crossings_tau = mirror_times_from_PS(
        ps_analysis,
        dt_ps_eff,
        interp=True,
        min_gap=user_min_gap,
        s_eps=v_eps
    )

    bounce_stats = bounce_summary(crossings_tau, time_scale_sec=tau_time)

    if bounce_stats["full_mean_s"] is not None:
        print("Mirror crossings:", bounce_stats["n_crossings"])
        print(f"Full bounce period (mean): {bounce_stats['full_mean_s']:.6g} s")
        print("Bounce frequency [Hz]:", bounce_stats["bounce_frequency_hz"])
    else:
        print("No mirror motion detected (no full-bounce interval).")

    bounce_results = {
        "n_crossings": bounce_stats["n_crossings"],
        "full_mean_tau": bounce_stats["full_mean_tau"],
        "full_mean_s": bounce_stats["full_mean_s"],
        "frequency_hz": bounce_stats["bounce_frequency_hz"],
    }

    # --- Drift ---
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

    T_drift_s   = drift_stats["period_s_mean"] or drift_stats["period_s_fit"]
    T_drift_tau = drift_stats["period_tau_mean"] or drift_stats["period_tau_fit"]
    direction   = drift_stats["direction"]

    if T_drift_s is None:
        print("Drift period: not enough azimuthal motion to estimate (yet).")
    else:
        print(
            f"Drift period ≈ {T_drift_s:.6g} s "
            f"(direction {'east' if direction > 0 else 'west'})"
        )

    drift_results = {
        "period_s": T_drift_s,
        "period_tau": T_drift_tau,
        "direction": direction,
    }


elif USE_PS and PS_CHUNKING:
    v_eps = npfloat(1e-14) * v_tau
    user_min_gap = max(3, int(0.5 * T_gyro / ps_step))

    bounce_state = init_bounce_stream_state()
    drift_state  = init_drift_stream_state()

    ps_store_stride = PS_decimate if PS_decimate > 1 else 1
    dt_store = ps_step * ps_store_stride

    with h5py.File(cache_path, "r") as ps_h5:
        ps_y = ps_h5["ps"]["y"]
        n_store = ps_y.shape[1]

        for j0 in range(0, n_store, PS_chunk_steps):
            j1 = min(j0 + PS_chunk_steps, n_store)

            y_chunk = ps_y[:, j0:j1]
            t_chunk = dt_store * np.arange(j0, j1, dtype=npfloat)

            process_bounce_and_drift_chunk(
                y_chunk=y_chunk,
                t_chunk=t_chunk,
                bounce_state=bounce_state,
                drift_state=drift_state,
                min_gap_tau=user_min_gap * ps_step,
                s_eps=v_eps,
            )

    # --- Bounce ---
    bounce_stats = bounce_summary(
        bounce_state["crossing_times"],
        time_scale_sec=tau_time
    )

    if bounce_stats["full_mean_s"] is not None:
        print("Mirror crossings:", bounce_stats["n_crossings"])
        print(f"Full bounce period (mean): {bounce_stats['full_mean_s']:.6g} s")
        print("Bounce frequency [Hz]:", bounce_stats["bounce_frequency_hz"])
    else:
        print("No mirror motion detected (no full-bounce interval).")

    bounce_results = {
        "n_crossings": bounce_stats["n_crossings"],
        "full_mean_tau": bounce_stats["full_mean_tau"],
        "full_mean_s": bounce_stats["full_mean_s"],
        "frequency_hz": bounce_stats["bounce_frequency_hz"],
    }

    # --- Drift ---
    drift_stats = finalize_drift_stream(
        drift_state,
        time_scale_sec=tau_time,
        min_phase_rad=user_min_phase,
    )

    T_drift_s   = drift_stats["period_s_fit"]
    T_drift_tau = drift_stats.get("period_tau_fit", None)
    direction   = drift_stats["direction"]

    if T_drift_s is None:
        print("Drift period: not enough azimuthal motion to estimate (yet).")
    else:
        print(
            f"Drift period ≈ {T_drift_s:.6g} s "
            f"(direction {'east' if direction > 0 else 'west'})"
        )

    drift_results = {
        "period_s": T_drift_s,
        "period_tau": T_drift_tau,
        "direction": direction,
    }


# ==================================================
# ========= Write Summary Output to File ==========
# ==================================================
"""
The Write Summary Output file contains some averaging over the last points of the run for E and mu errors/deviations.
This first part is setting a max fraction to look at an eventually capping the number of points that are looked at so
that there is not a memory issue with the large files. 
"""

if DEBUG: tracemalloc.start()

# ------  Define tail for last fraction of invariants -------
if gyroperiods < 1e6:
    TAIL_FRAC = 0.01        # last 1%
else:
    TAIL_FRAC = 0.0001     # last 0.01%

tail_start = (1.0 - TAIL_FRAC) * npfloat(norm_time)

MAX_TAIL_STEPS = 500000   # hard safety cap
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

# ------  Build tail masks for last fraction of invariants -------

if USE_PS:
    step_ps = ps_store_stride * ps_step
    tail_masks["PS"], j0_ps = make_tail_mask(rel_drift_ps.size, step_ps, "PS")

if USE_RK45:
    tail_masks["RK45"], j0_rk45 = make_tail_mask(len(rel_drift_rk45), ps_step, "RK45")

if USE_RK4:
    tail_masks["RK4"], j0_rk4 = make_tail_mask(len(rel_drift_rk4), rk4_step, "RK4")

if USE_RKG:
    tail_masks["RKG"], j0_rkg = make_tail_mask(len(rel_drift_rkg), rkg_step, "RKG")

# ============ Write summary file ==============

output_filename = build_figure_filename( summary , output_folder , stem , figure_tag="simulation_summary", ext="txt")


with open(output_filename, "w") as f:
    f.write("=== Simulation Summary ===\n")
    write_dict(f, summary)    # dumps the dictionary in as text
    f.write("\n")

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

    # === Bounce & Drift (PS only) ===
    if USE_PS:
        f.write("\n=== Bounce and Drift Motion ===\n")

        if bounce_results is None or bounce_results.get("full_mean_s") is None:
            f.write("Bounce: not detected / insufficient mirror crossings\n")
        else:
            f.write(f"Mirror crossings        : {bounce_results['n_crossings']}\n")
            f.write(f"Bounce period (s)       : {bounce_results['full_mean_s']:.6g}\n")
            f.write(f"Bounce frequency (Hz)   : {bounce_results['frequency_hz']:.6g}\n")

        if drift_results is None or drift_results.get("period_s") is None:
            f.write("Drift: not enough azimuthal phase to estimate\n")
        else:
            direction = drift_results.get("direction", 0)
            dir_str = "eastward" if direction > 0 else "westward"

            f.write(f"Drift period (s)        : {drift_results['period_s']:.6g}\n")
            f.write(f"Drift direction         : {dir_str}\n")

        f.write("\n")


if DEBUG:
    if USE_RK4:logger.debug(f"  rk4 step size = {rk4_step}")
    if USE_RKG: logger.debug(f"  rkg step size = {rkg_step}")
    if USE_RK4: logger.debug(f"  rk4 steps     = {steps_rk4}")
    if USE_RKG: logger.debug(f"  rkg steps     = {steps_rkg}")
    if USE_PS: logger.debug(f"  ps steps      = {steps_ps}")

# === Shared metadata ===
run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
x_0, y_0, z_0 = x_initial, y_initial, z_initial

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