from utility_scripts.project_setup import *

DEBUG = False # WARNING: Adds computation time. TURN OFF FOR LONG RUNS
if DEBUG:
    logger = setup_logger("dipole_logger", "dipoleB.log", level=logging.DEBUG) #This logger will log to a file in the working directory, it will overwrite each run unless you change the filename
    tracemalloc.start()


"""
Run selection:
    python dipoleB.py demo                  # named config  → configs/demo.yml
    python dipoleB.py configs/my_run.yml    # direct YAML path
    python dipoleB.py paper1                # falls back to load_params() if no YAML found
Defaults to "demo" if nothing is selected.
"""

run = "demo"
if len(sys.argv) > 1:       # this is scanning for the argument after dipoleb.py in the ternminal
    run = sys.argv[1]       # if it finds somethin thing it rewrites run as that argument
    print(f"Run mode set from command line: {run}\n")
else: print(f"Using default run mode: {run}\n")

_configs_dir = os.path.join(os.path.dirname(__file__), "configs") # _file_ is built-in variable that contains the path to the current script

if run.endswith((".yml", ".yaml")) and os.path.isfile(run):
    # Direct path to a YAML file
    print(f"Loading YAML config: {run}\n")
    params = load_config(run, npfloat=npfloat)
elif os.path.isfile(os.path.join(_configs_dir, f"{run}.yml")):
    # Named config → configs/<name>.yml
    _yaml_path = os.path.join(_configs_dir, f"{run}.yml")
    print(f"Loading YAML config: {_yaml_path}\n")
    params = load_config(_yaml_path, npfloat=npfloat)

# =========================================================
# ============= Assign YML file parameters ================
# =========================================================

# --- Always needed (from _defaults + every run mode) ---
READ_DATA       = params["READ_DATA"]
USE_RK45        = params["USE_RK45"]
USE_RK4         = params["USE_RK4"]
USE_RKG         = params["USE_RKG"]
USE_PS          = params["USE_PS"]
USE_ADAPTIVE    = params["USE_ADAPTIVE"]
PS_decimate     = params["PS_decimate"]
y_initial       = params["y_initial"]
z_initial       = params["z_initial"]
USE_PLOT_TITLES = params["USE_PLOT_TITLES"]
USE_FULL_PLOT   = params["USE_FULL_PLOT"]
slice_mode      = params["slice_mode"]
gyro_window     = params["gyro_window"]
output_folder   = params["output_folder"]
window_time     = params["window_time"]
N_GYRO          = params["N_GYRO"]

USE_EXTERNAL_H5_ps   = params["USE_EXTERNAL_H5_ps"]
USE_EXTERNAL_H5_rk4  = params["USE_EXTERNAL_H5_rk4"]
USE_EXTERNAL_H5_rk45 = params["USE_EXTERNAL_H5_rk45"]
USE_EXTERNAL_H5_rkg  = params["USE_EXTERNAL_H5_rkg"]
external_h5_ps       = params["external_h5_ps"]
external_h5_rk4      = params["external_h5_rk4"]
external_h5_rk45     = params["external_h5_rk45"]
external_h5_rkg      = params["external_h5_rkg"]

# --- Physics (set by all modes except manual, which load from h5) ---
pitch_deg    = params.get("pitch_deg",    None)
phi_deg      = params.get("phi_deg",      None)
x_initial    = params.get("x_initial",    None)
KE_particle  = params.get("KE_particle",  None)
mass_si      = params.get("mass_si",      None)
T_gyro       = params.get("T_gyro",       None)
gyroperiods  = params.get("gyroperiods",  None)
norm_time    = params.get("norm_time",    None)

# --- Step sizes (set by all modes except legacy/manual) ---
ps_step              = params.get("ps_step",  None)
rk4_step             = params.get("rk4_step", None)
rkg_step             = params.get("rkg_step", None)
N_STEPS_PER_GYRO_ps  = params.get("N_STEPS_PER_GYRO_ps",  None)
N_STEPS_PER_GYRO_rk4 = params.get("N_STEPS_PER_GYRO_rk4", None)
N_STEPS_PER_GYRO_rkg = params.get("N_STEPS_PER_GYRO_rkg", None)

# --- Optional overrides (only some modes set these) ---
PS_order       = params.get("PS_order", PS_order)             
manual_h5_path = params.get("manual_h5_path", None)

# === Misc Odds and Ends ===
PS_CHUNKING = True     # PS data always streamed to disk in chunks (no in-memory option)
WRITE_DATA  = True     # always write h5 (required by chunked streaming)
plt_config(scale=1)                        # config file for setting plot sizes and fonts (from Dr. W)
os.makedirs(run_storage, exist_ok=True)    # ensures file for the storagae for raw data exists
os.makedirs(output_folder, exist_ok=True)  # ensures file for the storagae for images and text file exists
plt.ioff()                                 # turn off interactive mode for plots
if USE_FLOAT128: USE_RKG = False


# ===============================================
# ============= Manual File Load ================
# ===============================================

USE_MANUAL_FILE = manual_h5_path is not None and os.path.exists(manual_h5_path)
if USE_MANUAL_FILE:   
    cache_path = manual_h5_path
    print(f"You have manually selected a file: {cache_path}\n") 
    if os.path.exists(cache_path):
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
            mass_si = summary["meta"]["mass_si"]
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

            T_gyro = 2.0 * np.pi * (x_initial**3) 

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

            # === PS (chunked — data stays on disk, read in slices later) ===
            if USE_PS and "ps" in cached:
                solution_ps = None
                orders_used = None

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

# for file/plot naming
if mass_si == m_e: particle_type = "Electron"
elif mass_si == m_p: particle_type = "Proton"
else: particle_type = "Particle"

qoverm      = npfloat(-1) if mass_si == m_e else npfloat(1)
charge_sign = float(qoverm)   # +1 proton, -1 electron (used in Dragt canonical momentum)

# === Misc Conversions  ===
KE_joules = KE_particle * evtoj                     # converting KE from eV to Joules
gamma = 1.0 + KE_joules / (mass_si * spdlight**2)   # Lorentz factor
mass = gamma * mass_si                              # Relativistic mass used for magnetic moment calculations

v_si = spdlight * np.sqrt(1.0 - 1.0 / gamma**2)     # m/s
tau_time = gamma * mass_si / (abs(q_e) * abs(B_0))  # this is tau0 from paper 
v_tau = v_si * tau_time / RE                        # dimensionless velocity

physical_time = norm_time * abs(tau_time)           # actual physical time, t; normalized time =t/tau_time
window_duration = window_time/tau_time              # converting window_time to dimensionless time
tol = npfloat(tol) * tau_time                       # Scale tolerance by tau_0
 
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
To streamline memory for large files, we are slicing out what we need
directly from the h5 file, this just establishes the E0 and mu0 values for those calculations
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
This first part is scanning the files already stored in 'run_storage' based on input parameters (not specifically
lodaded legacy files) in the yml to see if we already have the data. If it finds the data, it will 
load relevant parameters. If it does not find a file, it will start running the solvers to get the needed data. 
Beware that these files can be GB size for dipole.
"""
if not USE_MANUAL_FILE:
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
            # === PS ===
            if USE_PS and "ps" in cached:
                solution_ps = None
                orders_used = None

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

            # --- Dragt conservation monitor ---
            # Compute L-shell from initial canonical momentum (same logic as post-run)
            _rho_i = np.sqrt(x_initial**2 + y_initial**2)
            _vphi_i = (x_initial * vy_initial - y_initial * vx_initial) / _rho_i
            _Pphi_i = _rho_i * _vphi_i - charge_sign / _rho_i
            if charge_sign * _Pphi_i < 0:
                _L_mon = float(-charge_sign / _Pphi_i)
            else:
                _r_i = np.sqrt(x_initial**2 + y_initial**2 + z_initial**2)
                _L_mon = float(_r_i**3 / _rho_i**2)
            dragt_mon = DragtMonitor(_L_mon, charge_sign,
                                     check_every=1, rtol=1e-4)
            # ----------------------------------
            _stream_args = dict(
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
                dragt_monitor=dragt_mon,
            )
            if USE_ADAPTIVE:
                _stream_args.update(
                    order_low=50,
                    order_high=300,
                    grow_factor=1.5,
                    shrink_factor=0.5,
                    steps_per_local_gyro=200,
                    min_fast_path_N=100,
                )
                max_ps, elapsed_ps = run_ps_streaming_adaptive(**_stream_args)
            else:
                max_ps, elapsed_ps = run_ps_streaming_with_decimation(**_stream_args)
            dragt_mon.summary()
            solution_ps = None
            orders_used = None
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
                "tau0": npfloat(tau_time),
                "dtype": npfloat.__name__,  
            }
        }

        if USE_PS:
            max_ps_value = int(max_ps) if max_ps is not None else None
        else:
            max_ps_value = None

        results["ps"] = { "enabled": bool(USE_PS),}
        if USE_PS:
            results["ps"].update({
                "y": None,
                "orders": None,
                "ordercap": PS_order,
                "max_ps": max_ps_value,
                "numberstepspergyro": N_STEPS_PER_GYRO_ps,
                "dt": ps_step,
                "steps": steps_ps,
                "streaming": True,
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
                    "tau0": tau_time,
                    "timing": results["meta"]["timing"],
                },
                "ps": {
                    "enabled": USE_PS,
                    "dt": ps_step if USE_PS else None,
                    "steps": steps_ps if USE_PS else None,
                    "streaming": True if USE_PS else None,
                    "ordercap": PS_order if USE_PS else None,
                    "max_ps": max_ps_value,
                    "chunksize": PS_chunk_steps if USE_PS else None,
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
            if USE_PS:
                append_results_h5(cache_path, results, summary)
                print(f"Updated streamed file → {os.path.basename(cache_path)}")
            else:
                save_results_h5(cache_path, results, summary)
                print(f"Saved results → {os.path.basename(cache_path)}")

if DEBUG: 
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    logger.info(f"Peak memory usage for load/write h5: {peak / 1024**2:.2f} MB\n")
    logger.debug(check_time_grids(
    norm_time=norm_time,
    ps_step=ps_step if USE_PS else None,
    steps_ps=steps_ps if USE_PS else None,
    rk4_step=rk4_step if USE_RK4 else None,
    steps_rk4=steps_rk4 if USE_RK4 else None,
    rkg_step=rkg_step if USE_RKG else None,
    steps_rkg=steps_rkg if USE_RKG else None,
    rk45_t=solution_rk45.t if USE_RK45 else None,
))


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

ps_order_label = None # for plotting later

if USE_PS:
    with h5py.File(cache_path, "r") as ps_h5:
        ps_grp = ps_h5["ps"]
        n_ps = steps_ps
        stride = max(1, n_ps // MAX_PLOT_POINTS)
        ps_order_label = int(ps_grp.attrs["max_ps"])

        if USE_FULL_PLOT:
            ps_y_h5 = ps_grp["y"]
            x_ps_plot = ps_y_h5[0, ::stride]
            y_ps_plot = ps_y_h5[1, ::stride]
            z_ps_plot = ps_y_h5[2, ::stride]

if DEBUG:
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    logger.debug(f"Data access for plottings: {peak / 1024**2:.2f} MB\n")

# === paper specific adjustments for aeshetic purposes ====
if run == "paper1": USE_RK4 = False 
if run == "paper2": USE_RKG = False


print(f"\n{'='*60}")
print(f"  Run Statistics")
print(f"{'='*60}")
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
print(f"PS Orders       : max={ps_order_label}")
print(f"% of c          : {100*v_si/spdlight:.8f}")

if DEBUG: 
    logger.debug(f"Norm Time: {norm_time:.2e} MB")
    logger.debug(f"Physical Time   : {physical_time:.2e} s")
    logger.debug(f"ps_step: {ps_step}, norm_time: {norm_time}, steps_ps: {steps_ps}")
    # logger.debug(f"t_common[0]: {t_common[0]}, t_common[-1]: {t_common[-1]}")
print(f"{'='*60}")


# === Create run-specific output subfolder ===
# All plots, txt summaries, and CSVs go here. Raw h5 data stays in run_storage.
run_folder = os.path.join(output_folder, stem)
os.makedirs(run_folder, exist_ok=True)

# =====================================================
# ============== Full Trajectory Plots ================
# =====================================================
plotbounds = x_initial + 1.1

if USE_FULL_PLOT:
    _traj_common = dict(
        summary=summary, run_folder=run_folder, stem=stem,
        particle_type=particle_type, plotbounds=plotbounds,
        ps_order_label=ps_order_label, USE_PLOT_TITLES=USE_PLOT_TITLES,
        USE_RK45=USE_RK45, USE_RK4=USE_RK4, USE_RKG=USE_RKG, USE_PS=USE_PS,
        solution_rk45=solution_rk45 if USE_RK45 else None,
        solution_rk4=solution_rk4 if USE_RK4 else None,
        solution_rkg=solution_rkg if USE_RKG else None,
        x_ps_plot=x_ps_plot if USE_PS else None,
        y_ps_plot=y_ps_plot if USE_PS else None,
    )
    plot_full_2d(**_traj_common)
    plot_full_3d(**_traj_common, z_ps_plot=z_ps_plot if USE_PS else None)

# ========================================================================
# ================ Creating Plot Window (slice of time) ==================
# ========================================================================
"""
Generally only interested in a specific window of time for a run, like 'first' and 'last' parts of the run. Test particle
yml file lets you specify in physical seconds how big you want this window to be via window_time. Generally looking at a drift
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

    ps_store_stride = PS_decimate if PS_decimate > 1 else 1

    # --- map physical → stored indices ---
    j0 = int(np.ceil(i0_phys / ps_store_stride))
    j1 = int(np.floor(i1_phys / ps_store_stride))

    if j1 < j0:
        raise RuntimeError("Empty PS stored slice window")

    # --- Load ONLY the window from HDF5 ---
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
    tol_factor = 1.5  # allow ~1 timestep mismatch

    def _check_window(label, t0, t1, dt):
        if t0 is None or t1 is None or dt is None:
            return
        dt_tol = tol_factor * dt
        if abs(t0 - t_start) > dt_tol or abs(t1 - t_end) > dt_tol:
            logger.warning(
                f"[SLICE MISMATCH] {label}: "
                f"[{t0:.6e}, {t1:.6e}] vs "
                f"expected [{t_start:.6e}, {t_end:.6e}] "
                f"(dt≈{dt:.2e})"
            )
        else:
            logger.debug(
                f"[SLICE OK] {label}: "
                f"[{t0:.6e}, {t1:.6e}]"
            )

    # ---- PS ----
    if USE_PS:
        t_ps_start = j0 * ps_store_stride * ps_step
        t_ps_end   = j1 * ps_store_stride * ps_step
        _check_window("PS", t_ps_start, t_ps_end, ps_step)

    # ---- RK4 ----
    if USE_RK4:
        t_rk4_start = t_rk4[0] if len(t_rk4) else None
        t_rk4_end   = t_rk4[-1] if len(t_rk4) else None
        _check_window("RK4", t_rk4_start, t_rk4_end, rk4_step)

    # ---- RKG ----
    if USE_RKG:
        t_rkg_start = t_rkg[0] if len(t_rkg) else None
        t_rkg_end   = t_rkg[-1] if len(t_rkg) else None
        _check_window("RKG", t_rkg_start, t_rkg_end, rkg_step)

    # ---- RK45 ----
    if USE_RK45:
        t_rk45_start = t_rk45[0] if len(t_rk45) else None
        t_rk45_end   = t_rk45[-1] if len(t_rk45) else None
        _check_window("RK45", t_rk45_start, t_rk45_end, ps_step)



# =====================================================
# ================ Trajectory Slice Plots =============
# =====================================================
_slice_common = dict(
    summary=summary, run_folder=run_folder, stem=stem,
    particle_type=particle_type, ps_order_label=ps_order_label,
    USE_PLOT_TITLES=USE_PLOT_TITLES,
    USE_RK45=USE_RK45, USE_RK4=USE_RK4, USE_RKG=USE_RKG, USE_PS=USE_PS,
    rk45_x_slice=rk45_x_slice if USE_RK45 else None,
    rk45_y_slice=rk45_y_slice if USE_RK45 else None,
    rk4_x_slice=rk4_x_slice if USE_RK4 else None,
    rk4_y_slice=rk4_y_slice if USE_RK4 else None,
    rkg_x_slice=rkg_x_slice if USE_RKG else None,
    rkg_y_slice=rkg_y_slice if USE_RKG else None,
    ps_x_slice=ps_x_slice if USE_PS else None,
    ps_y_slice=ps_y_slice if USE_PS else None,
)

if USE_FULL_PLOT:
    plot_slice_2d(**_slice_common)

plot_slice_3d(
    **_slice_common, plotbounds=plotbounds,
    rk45_z_slice=rk45_z_slice if USE_RK45 else None,
    rk4_z_slice=rk4_z_slice if USE_RK4 else None,
    rkg_z_slice=rkg_z_slice if USE_RKG else None,
    ps_z_slice=ps_z_slice if USE_PS else None,
)
 

# =====================================================
# ============== KE Relative Error Plot ===============
# =====================================================
"""
This section calculates the relative KE error plot over the entire run. This is done in chunks
"""

if DEBUG: tracemalloc.start()

time_factor = 1.0 / T_gyro  # to convert normalized time to gyroperiods

if USE_PS: energy_stride = max(1, n_ps // MAX_PLOT_POINTS)
# energy_stride=1 


if USE_EXTERNAL_H5_ps:
    # Open the file directly from the SSD to avoid loading 200GB into RAM...it will die a horrible death
    with h5py.File(external_h5_ps, 'r') as external:
        ext_ps = external["ps"]
        
        # Create a reference to the 'y' dataset on the disk
        y_ext = ext_ps["y"]
        n_store = y_ext.shape[1]

        # Read scalar values from the metadata attributes (.attrs)
        ps_step_ext = ext_ps.attrs["dt"]
        ps_decimate_ext = ext_ps.attrs.get("decimate", 1)
        dt_store_ext = ps_step_ext * ps_decimate_ext

        energy_stride_ext = max(1, n_store // MAX_PLOT_POINTS)
        
        # ---- plot-ready time axis ----
        idx = np.arange(0, n_store, energy_stride_ext)
        t_eval_ps_ext = idx * dt_store_ext

        # ---- strided energy ----
        # The ::stride syntax pulls ONLY the needed points directly from the SSD
        vxe = y_ext[3, ::energy_stride_ext].astype(np.float64)
        vye = y_ext[4, ::energy_stride_ext].astype(np.float64)
        vze = y_ext[5, ::energy_stride_ext].astype(np.float64)

        E_ext = 0.5 * (vxe*vxe + vye*vye + vze*vze)
        rel_drift_ps_ext = (E_ext - E_ext[0]) / E_ext[0]

        PS_order_ext = ext_ps.attrs.get("max_ps", None)

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

    y_rk45_ext = ext_rk45["y"]

    # ensure shape consistency
    if y_rk45_ext.shape[0] != 6:
        y_rk45_ext = y_rk45_ext.T

    n_store = y_rk45_ext.shape[1]

    # ---- time base ----
    if "t" in ext_rk45 and ext_rk45["t"] is not None:
        t_ext = np.asarray(ext_rk45["t"])

    else:
        # RK45 on PS grid → respect PS decimation
        ps_step_ext = ext_rk45.get("dt", ps_step)
        ps_decimate_ext = ext_rk45.get("decimate", 1)
        dt_store_ext = ps_step_ext * ps_decimate_ext
        t_ext = dt_store_ext * np.arange(n_store, dtype=npfloat)

    # ---- energy stride (plot-only) ----
    energy_stride_ext = max(1, n_store // MAX_PLOT_POINTS)
    idx = np.arange(0, n_store, energy_stride_ext)

    t_eval_rk45_ext = t_ext[idx]

    # ---- velocity, energy ----
    v = y_rk45_ext[3:6, idx].astype(np.float64)
    E = 0.5 * np.sum(v*v, axis=0)
    rel_drift_rk45_ext = (E - E[0]) / E[0]


if USE_EXTERNAL_H5_rkg:
    with h5py.File(external_h5_rkg, 'r') as external_file:
        ext_rkg = external_file["rkg"]
        y_dataset = ext_rkg["y"]   

        is_transposed = (y_dataset.shape[0] == 6)
        n_steps = y_dataset.shape[1] if is_transposed else y_dataset.shape[0]
        rkg_stride = max(1, n_steps // MAX_PLOT_POINTS)

        # ---- Handle Time Axis ----
        if "t" in ext_rkg:
            t_ext_rkg = ext_rkg["t"][::rkg_stride]
        else:
            dt_rkg = ext_rkg.attrs.get("dt", ext_rkg.get("dt", None))
            
            if dt_rkg is None and "params_json" in external_file.attrs:
                params = json.loads(external_file.attrs["params_json"])
                dt_rkg = params.get("rkg_step") 
                
            if dt_rkg is not None:
                if hasattr(dt_rkg, 'value'): dt_rkg = dt_rkg[()]
                idx = np.arange(0, n_steps, rkg_stride)
                
                t_ext_rkg = dt_rkg * idx.astype(npfloat) 
            else:
                raise ValueError("External RKG H5 file has no time info.")

        if is_transposed:
            y_ext_rkg = y_dataset[:, ::rkg_stride].T 
        else:
            y_ext_rkg = y_dataset[::rkg_stride, :]

    # =========================================================================
    # Operating only on the strided subset in RAM now.
    # =========================================================================

    r_rkg_ext = y_ext_rkg[:, 0:3]
    p_rkg_ext = y_ext_rkg[:, 3:6]

    A_rkg_ext = np.zeros_like(r_rkg_ext)
    for i in range(len(r_rkg_ext)):
        A_rkg_ext[i] = vector_potential_dipole(r_rkg_ext[i])

    v_rkg_ext = p_rkg_ext - A_rkg_ext
    E_rkg_ext = npfloat(0.5) * np.sum(v_rkg_ext**2, axis=1, dtype=npfloat)
    rel_drift_ext_rkg = np.abs(E_rkg_ext - E_rkg_ext[0]) / E_rkg_ext[0]

if USE_PS:
    with h5py.File(cache_path, "r") as ps_h5:
        ps_y_h5 = ps_h5["ps"]["y"]

        t_ps_plot, rel_drift_ps = compute_energy_ps_chunked(
            ps_y_h5=ps_y_h5,
            E0_ps=E0_ps,
            dt_ps_store=ps_step * (PS_decimate if PS_decimate > 1 else 1),
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

# --- build time arrays for non-PS solvers ---
if USE_RK4:
    t_rk4 = rk4_step * np.arange(len(rel_drift_rk4), dtype=npfloat)
if USE_RKG:
    t_rkg = rkg_step * np.arange(len(rel_drift_rkg), dtype=npfloat)
if USE_RK45:
    t_rk45 = ps_step * np.arange(len(rel_drift_rk45), dtype=npfloat)

# --- assemble plot data tuples ---
_ke_ps   = (t_ps_plot, rel_drift_ps) if USE_PS else None
_ke_rk4  = (t_rk4, rel_drift_rk4) if USE_RK4 else None
_ke_rk45 = (t_rk45, rel_drift_rk45) if USE_RK45 else None
_ke_rkg  = (t_rkg, rel_drift_rkg) if USE_RKG else None

_ke_ext_ps   = (t_eval_ps_ext, rel_drift_ps_ext, PS_order_ext) if USE_EXTERNAL_H5_ps else None
_ke_ext_rk4  = (t_eval_rk4_ext, rel_drift_rk4_ext) if USE_EXTERNAL_H5_rk4 else None
_ke_ext_rk45 = (t_eval_rk45_ext, rel_drift_rk45_ext) if USE_EXTERNAL_H5_rk45 else None
_ke_ext_rkg  = (t_ext_rkg, rel_drift_ext_rkg) if USE_EXTERNAL_H5_rkg else None

plot_ke_error(
    summary=summary, run_folder=run_folder, stem=stem,
    particle_type=particle_type, ps_order_label=ps_order_label,
    USE_PLOT_TITLES=USE_PLOT_TITLES, time_factor=time_factor, norm_time=norm_time,
    ps_data=_ke_ps, rk4_data=_ke_rk4, rk45_data=_ke_rk45, rkg_data=_ke_rkg,
    ext_ps_data=_ke_ext_ps, ext_rk4_data=_ke_ext_rk4,
    ext_rk45_data=_ke_ext_rk45, ext_rkg_data=_ke_ext_rkg,
)  

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

# ==================================================
# --- Dragt physics defaults (populated below if PS data exists) ---
dragt_log = {
    "L_eff": None, "W0_sq": None, "boundary": None,
    "mu_sq": None, "orbit_character": None,
    "eps_initial": None, "eps_mean": None, "eps_max": None,
    "hit_atmosphere": False, "hit_atm_r": None,
}
# ==================================================
# ======== Dragt Poincaré Surface of Section =======
# ==================================================
plt.close('all')

# L-shell for Dragt normalization: derived from the conserved canonical momentum.
# P_phi = rho*v_phi - charge_sign/rho  (code's upward-B convention)
# L = 1/|P_phi|  →  L is L-independent and orbit-conserved, consistent with dragt.py.
#
# For a proton (charge_sign=+1) drifting westward: P_phi < 0, L = -1/P_phi
# For an electron (charge_sign=-1) drifting eastward: P_phi > 0, L = +1/P_phi
#
# Fallback to field-line L (r^3/rho^2) for degenerate cases (e.g. purely radial launch).
_rho_init    = np.sqrt(x_initial**2 + y_initial**2)
_v_phi_init  = (x_initial * vy_initial - y_initial * vx_initial) / _rho_init
# Canonical momentum in code's upward-B convention: P_phi = rho*v_phi - charge_sign/rho
#   Proton  (charge_sign=+1): P_phi = rho*v_phi - 1/rho  < 0 for trapped (westward drift)
#   Electron (charge_sign=-1): P_phi = rho*v_phi + 1/rho  > 0 for trapped (eastward drift)
# Trapped condition (both species): charge_sign * P_phi < 0
# L-shell from canonical momentum: L = 1/|P_phi| = -charge_sign / P_phi
_P_phi_code = _rho_init * _v_phi_init - charge_sign / _rho_init
if charge_sign * _P_phi_code < 0:
    L_shell_dragt = float(-charge_sign / _P_phi_code)  # = 1/|P_phi|
else:
    _r_init = np.sqrt(x_initial**2 + y_initial**2 + z_initial**2)
    L_shell_dragt = float(_r_init**3 / _rho_init**2)
    print("  WARNING: P_phi_code indicates open/untrapped orbit, falling back to field-line L-shell")

print(f"\n{'='*60}")
print(f"  Dragt Info")
print(f"{'='*60}")
print(f"Dragt L-shell (from conserved canonical momentum): {L_shell_dragt:.4f} R_E")

if USE_PS:
    # --- Initial state from h5 ---
    with h5py.File(cache_path, "r") as _h5_init:
        _y0 = _h5_init["ps"]["y"][:6, 0].astype(float)

    # --- Cache consistency check ---
    _v_mag_ps0 = float(np.sqrt(_y0[3]**2 + _y0[4]**2 + _y0[5]**2))
    _v_tau_expected = float(v_tau)
    _v_rel_err = abs(_v_mag_ps0 - _v_tau_expected) / _v_tau_expected
    if _v_rel_err > 0.005:
        print(f"\n  *** CACHE MISMATCH WARNING ***")
        print(f"  PS trajectory v_mag = {_v_mag_ps0:.6f}  (from cached run)")
        print(f"  Current v_tau       = {_v_tau_expected:.6f}  (from current KE_particle/mass_si)")
        print(f"  Relative error      = {_v_rel_err*100:.2f}%")
        print(f"  The cached trajectory does not match the current input parameters.")
        print(f"  Re-run the simulation (disable cache loading) to get consistent results.\n")

    # --- Dragt parameters from initial conditions only ---
    _x0a  = np.array([_y0[0]]); _y0a = np.array([_y0[1]]); _z0a = np.array([_y0[2]])
    _vx0a = np.array([_y0[3]]); _vy0a = np.array([_y0[4]]); _vz0a = np.array([_y0[5]])
    dp         = compute_dragt_params(_x0a, _y0a, _z0a, _vx0a, _vy0a, _vz0a, L_shell_dragt,
                                      charge_sign=charge_sign)
    W0_sq_calc = dp["W0_sq"]
    P_phi      = dp["P_phi"]
    rho_0_sim  = dp["rho_0_sim"]
    _rho_dot_0_sim = float((_y0[0]*_y0[3] + _y0[1]*_y0[4]) / rho_0_sim)

    # --- Analytical boundary ---
    rho_bnd, rho_dot_bnd = compute_dragt_boundary(W0_sq_calc, P_phi, charge_sign=charge_sign)

    # --- Chunked Dragt analysis (adiabaticity, meridian, crossings) ---
    _dragt = dragt_analysis_chunked(cache_path, L_shell_dragt, ps_step, time_factor)
    _dragt_eps_arr     = _dragt["eps_arr"]
    _dragt_t_arr       = _dragt["t_arr"]
    _dragt_rho_arr     = _dragt["rho_arr"]
    _dragt_z_arr       = _dragt["z_arr"]
    _dragt_eps_initial = _dragt["eps_initial"]
    _dragt_eps_mean    = _dragt["eps_mean"]
    _dragt_eps_max     = _dragt["eps_max"]
    crossings          = _dragt["crossings"]

    # --- Poincaré surface of section ---
    plot_dragt_poincare(
        run_folder=run_folder, L_shell_dragt=L_shell_dragt, gamma=gamma,
        rho_bnd=rho_bnd, rho_dot_bnd=rho_dot_bnd,
        rho_0_sim=rho_0_sim, rho_dot_0_sim=_rho_dot_0_sim,
        crossings=crossings,
    )

    # --- Gyrophase / magnetic moment plots (only if crossings exist) ---
    if crossings is not None:
        rho_dragt, rho_dot_dragt, x_cross, y_cross, vx_cross, vy_cross = crossings
        gyrophase, mu_cross = compute_gyrophase_mu(x_cross, y_cross, vx_cross, vy_cross)
        plot_gyrophase_mu(run_folder, gyrophase, mu_cross)
        plot_polar_phase_space(run_folder, gyrophase, mu_cross)

    # --- Meridian plane ---
    plot_meridian_plane(run_folder, _dragt_rho_arr, _dragt_z_arr)

    # --- Adiabaticity parameter ---
    plot_adiabaticity(run_folder, _dragt_t_arr, _dragt_eps_arr,
                      _dragt_eps_initial, _dragt_eps_mean, _dragt_eps_max)

    # --- Populate Dragt physics log ---
    dragt_log["L_eff"]           = L_shell_dragt
    dragt_log["W0_sq"]           = W0_sq_calc
    dragt_log["boundary"]        = dp["boundary_status"]
    dragt_log["mu_sq"]           = dp["mu_sq"]
    dragt_log["orbit_character"] = dp["orbit_character"]
    dragt_log["eps_initial"]     = _dragt_eps_initial
    dragt_log["eps_mean"]        = _dragt_eps_mean
    dragt_log["eps_max"]         = _dragt_eps_max

    # --- Atmosphere flag from h5 ---
    try:
        with h5py.File(cache_path, "r") as _h5:
            if "ps" in _h5 and "hit_atmosphere" in _h5["ps"].attrs:
                dragt_log["hit_atmosphere"] = bool(_h5["ps"].attrs["hit_atmosphere"])
                dragt_log["hit_atm_r"]     = float(_h5["ps"].attrs["hit_atm_r"])
    except Exception:
        pass

else:
    print("PS is not enabled. Cannot run Dragt Comparison.")

plt.close('all')



# =========================================================
# PLOT RELATIVE ERROR OF CANONICAL ANGULAR MOMENTUM
# =========================================================

# _y0 is the initial state vector set earlier in the Dragt section
pphi = compute_pphi_error_chunked(cache_path, _y0, charge_sign, ps_step, time_factor)
plot_pphi_error(run_folder, pphi["t_gyro"], pphi["rel_error_log"],
                pphi["P_phi_initial"], pphi["max_err"], pphi["ylabel"])


# ============================================================
# ================ Magnetic Moment Deviations ================
# ============================================================
if DEBUG: tracemalloc.start()

if USE_RK4:
    window_steps_rk4 = int(round(N_GYRO * N_STEPS_PER_GYRO_rk4))
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
    window_steps_rkg = int(round(N_GYRO * N_STEPS_PER_GYRO_rkg))
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
    window_steps_ps = int(round(N_GYRO * N_STEPS_PER_GYRO_ps))
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

if USE_PS:
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
    j0 = int(np.floor(i0_phys / ps_store_stride))
    j1 = int(np.ceil(i1_phys / ps_store_stride))

    with h5py.File(cache_path, "r") as ps_h5:
        ps_grp = ps_h5["ps"]
        ps_y = ps_grp["y"]
        ps_order_label = int(ps_grp.attrs["max_ps"])
        n_store = ps_y.shape[1]

        j0 = max(0, min(j0, n_store))
        j1 = max(0, min(j1, n_store))

        if j1 <= j0:
            raise RuntimeError("Empty PS μ window (chunked)")

        y_ps_win = expand_h5_to_full(ps_y[:, j0:j1])
        mu_ps = compute_mu_ps(y_ps_win, mass)
        mudrift_ps = np.abs(mu_ps - mu0_ps) / mu0_ps

        dt_ps_store = ps_step * ps_store_stride
        t_ps_store = np.arange(j0, j1, dtype=npfloat) * dt_ps_store
        moment_stride = max(1, round(len(mu_ps) // MAX_PLOT_POINTS))
        t_ps_plot = t_ps_store[::moment_stride] * time_factor
        mudrift_ps_plot = mudrift_ps[::moment_stride]

# ===== Plotting ========
plot_mu_deviation(
    summary, run_folder, stem, particle_type, ps_order_label,
    USE_PLOT_TITLES,
    ps_data=(t_ps_plot, mudrift_ps_plot) if USE_PS else None,
    rk4_data=(t_rk4_plot, mudrift_rk4) if USE_RK4 else None,
    rk45_data=(t_rk45_plot, mudrift_rk45) if USE_RK45 else None,
    rkg_data=(t_rkg_plot, mudrift_rkg) if USE_RKG else None,
)


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

    if USE_RK4: logger.debug(f"mu-window RK4  : {window_steps_rk4 * rk4_step:.6e}")
    if USE_RKG: logger.debug(f"mu-window RKG  : {window_steps_rkg * rkg_step:.6e}")
    if USE_RK45: logger.debug(f"mu-window RK45 : {window_steps_ps * ps_step:.6e}")
    if USE_PS: logger.debug(f"mu-window PS   : {window_steps_ps * ps_step:.6e}")


# ===================================================
# ================ Mirror and Drift  ================
# ===================================================
# Bounce and drift are only calculated for PS method, using chunked h5 streaming.
bounce_results = None
drift_results  = None

print(f"\n{'='*60}")
print(f"  Bounce/Drift Statistics")
print(f"{'='*60}")

if USE_PS:
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

            y_chunk = expand_h5_to_full(ps_y[:, j0:j1])
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

    print(f"Initial gyroradius: {gyro_radius_si:.4e} m  ({gyro_radius_RE:.4f} R_E)")
    gyro_freq_hz = 1.0 / (T_gyro * abs(tau_time))
    print(f"Gyrofrequency: {gyro_freq_hz:.4f} Hz  (period: {T_gyro * abs(tau_time):.4e} s)")

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

if DEBUG: tracemalloc.start()

write_summary_txt(
    summary=summary, run_folder=run_folder, stem=stem,
    dragt_log=dragt_log, bounce_results=bounce_results, drift_results=drift_results,
    gyroperiods=gyroperiods, norm_time=norm_time, mass=mass, cache_path=cache_path,
    USE_PS=USE_PS, USE_RK4=USE_RK4, USE_RK45=USE_RK45, USE_RKG=USE_RKG,
    PS_decimate=PS_decimate,
    ps_step=ps_step,
    rk4_step=rk4_step if USE_RK4 else None,
    rkg_step=rkg_step if USE_RKG else None,
    rel_drift_ps=rel_drift_ps if USE_PS else None,
    rel_drift_rk4=rel_drift_rk4 if USE_RK4 else None,
    rel_drift_rk45=rel_drift_rk45 if USE_RK45 else None,
    rel_drift_rkg=rel_drift_rkg if USE_RKG else None,
    mu0_ps=mu0_ps if USE_PS else None,
    mu0_rk4=mu0_rk4 if USE_RK4 else None,
    mu0_rk45=mu0_rk45 if USE_RK45 else None,
    mu0_rkg=mu0_rkg if USE_RKG else None,
    solution_rk4=solution_rk4 if USE_RK4 else None,
    solution_rkg=solution_rkg if USE_RKG else None,
    y_rk45_common=y_rk45_common if USE_RK45 else None,
    ps_store_stride=ps_store_stride if USE_PS else 1,
    npfloat=npfloat,
)

if DEBUG:
    if USE_RK4:logger.debug(f"  rk4 step size = {rk4_step}")
    if USE_RKG: logger.debug(f"  rkg step size = {rkg_step}")
    if USE_RK4: logger.debug(f"  rk4 steps     = {steps_rk4}")
    if USE_RKG: logger.debug(f"  rkg steps     = {steps_rkg}")
    if USE_PS: logger.debug(f"  ps steps      = {steps_ps}")

# === Write to master simulation log CSV ===
_method_records = []
if USE_RK4: _method_records.append(("RK4",  steps_rk4,  rk4_step,  rel_drift_rk4,  mudrift_rk4))
if USE_RK45: _method_records.append(("RK45", steps_ps,   ps_step,   rel_drift_rk45, mudrift_rk45))
if USE_RKG: _method_records.append(("RKG",  steps_rkg,  rkg_step,  rel_drift_rkg,  mudrift_rkg))
if USE_PS: _method_records.append(("PS",   steps_ps,   ps_step,   rel_drift_ps,   mudrift_ps))

write_master_csv(
    output_folder=output_folder, stem=stem, particle_type=particle_type,
    KE_particle=KE_particle, x_initial=x_initial, y_initial=y_initial,
    z_initial=z_initial, pitch_deg=pitch_deg, phi_deg=phi_deg,
    dragt_log=dragt_log,
    method_records=_method_records,
)

print(f"\nRun Complete → {run_folder}")


if DEBUG:
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    logger.info(f"Peak memory usage for summary write up: {peak / 1024**2:.2f} MB")
