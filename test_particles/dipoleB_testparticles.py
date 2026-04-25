import numpy as np
import os
import matplotlib.pyplot as mpl
import json as _json


USE_FLOAT128 = False  # RKG Will be diabled if this is True

if USE_FLOAT128: npfloat = np.float128
else: npfloat = np.float64

# ===== Physical Constants (from shared module) =====
from constants import q_e, m_e, m_p, evtoj, spdlight, RE, B_0


# ===== Tolerances/Truncations =====
PS_order = 40                       # Max Power Series Order, system will truncate
tol = 1.0 * np.finfo(npfloat).eps   # setting tolerance to machine epsilon to drop terms later, will be multiplied by tau_0
rtol_rk45 = 1e-8                    # RK45 relative tolerance
atol_rk45 = 1e-10                   # RK45 adapative tolerance
user_min_phase = npfloat(.1)        # Minimum radians to extrapolate drift from,   .0000001 needed for L=1, 10 eV electron
mpl.rcParams['agg.path.chunksize'] = 100   # may have to adjust if matplotlib barfs on large datasets

if USE_FLOAT128: mpl.rcParams['agg.path.chunksize'] = 100000  
else: mpl.rcParams['agg.path.chunksize'] = 1000


MAX_PLOT_POINTS= 1000000 # cap number of points on each graph to speed things up
PS_chunk_steps = int(1e4)

run_storage = "outputs/outputs_rawdata"       # where raw trajectory files go when USE_WRITE_DATA = True
# run_storage = "/Volumes/Extreme SSD/Thesis/PS_method/outputs_rawdata"       # where raw trajectory files go when USE_WRITE_DATA = True, set to external drive for large runs


# ===================================================================
# ==============Toggle Parameters for Dipole Script ================
# ===================================================================
"""
Toggles:
    USE_RK45        -- Set to True to include RK45 (Dormand-Prince) analysis
    USE_RK4         -- Set to True to include RK4 analysis
    USE_RKG         -- Set to True to include RKG analysis (protons only)
    USE_PS          -- Set to True to include PS analysis
    PS_CHUNKING     -- Set to True to stream PS data to disk in chunks (recommended for long runs)
    PS_decimate     -- Save every Nth PS step to reduce file size (1 = save all)
    READ_DATA       -- Set to True to scan for saved runs and load from cache
    WRITE_DATA      -- Set to True to write trajectory data to h5 file
    USE_PLOT_TITLES -- Set to True to include plot titles
    USE_FULL_PLOT   -- Set to False for paper plots only, True for all diagnostic plots

    USE_EXTERNAL_H5_* -- Set to True to load an external h5 file for energy comparison,
                         useful when comparing runs with different parameter sets or timescales
    external_h5_*     -- Path to external h5 file; for PS also set PS_order_ext to match

Physics parameters (changing these creates a new h5 file):
    pitch_deg   -- equatorial pitch angle (degrees)
    phi_deg     -- initial gyrophase (degrees)
    x_initial   -- launch L-shell (Earth radii, R_E)
    y_initial   -- initial y position (R_E), typically 0
    z_initial   -- initial z position (R_E), typically 0
    KE_particle -- kinetic energy (eV)
    mass_si     -- particle mass: m_p for protons, m_e for electrons, or manual (kg)
    T_gyro      -- characteristic gyroperiod at equator (normalized time): 2*pi*L^3

Integration parameters:
    N_STEPS_PER_GYRO_* -- number of steps per characteristic gyroperiod for each integrator
    rk4_step    -- RK4 time step: T_gyro / N_STEPS_PER_GYRO_rk4
    ps_step     -- PS time step: set equal to rk4_step for direct comparison
    rkg_step    -- RKG time step: set equal to rk4_step for direct comparison
    gyroperiods -- number of characteristic gyroperiods to simulate
    norm_time   -- total integration time in normalized units: gyroperiods * T_gyro

Plotting parameters (do not affect h5 data):
    window_time  -- time window in physical seconds (SI) for trajectory slice inspection,
                    typically one drift period; converted to normalized time internally via tau_0
    slice_mode   -- "last" or "first", slices window from end or beginning of simulation
    N_GYRO       -- number of gyroperiods to display in slice
    gyro_window  -- "first", "last", or "all", which gyroperiods to display in the window

Run modes:
    demo         -- quick verification run (seconds)
    paper1-3     -- reproduce paper figures (minutes each)
    batch:path   -- batch mode, reads parameters from JSON config (used by batch_flux_runner.py)
    legacy       -- load old-format h5 files
    manual       -- load specific h5 file
    monster_ps   -- extended PS-only run

This file is shared by both dipoleB.py (fixed-step) and dipoleB_adp.py (adaptive).
"""

def load_params(run):
    if run == "demo": #paper2 simulation at reduced norm time for quick demo
        if USE_FLOAT128: print("Running DEMO simulation in float128...this may take a few minutes\n")
        else: print("Running DEMO simulation...this takes just few seconds\n")
        output_folder = "outputs/outputs_demo"
        os.makedirs(output_folder, exist_ok=True)

        READ_DATA = True
        WRITE_DATA = True

        # -------- physics parameters -------
        """
        Note, changing these parameters will results in a new h5 file creation. H5 files are constructed
        and named based on these entries and altering them changes the structure/physics of the data.
        """

        USE_RK45 = True  
        USE_RK4 = True
        USE_RKG = False  # does not work for electrons, see paper
        USE_PS = True
        PS_decimate = 1    # only works with chunking
        PS_CHUNKING = True

        pitch_deg = npfloat(60.0)              
        phi_deg = npfloat(90.0)
        x_initial = npfloat(5)                 
        y_initial = npfloat(0)
        z_initial = npfloat(0)
        KE_particle = npfloat(100e6) 
        mass_si = m_e   
        T_gyro = 2.0 * np.pi * (x_initial**3)  

        N_STEPS_PER_GYRO_rk4= 65
        N_STEPS_PER_GYRO_ps=65
        N_STEPS_PER_GYRO_rkg=65
        rk4_step = npfloat(round(T_gyro/N_STEPS_PER_GYRO_rk4,1))               
        ps_step = npfloat(round(T_gyro/N_STEPS_PER_GYRO_ps,1))                                  
        rkg_step = npfloat(round(T_gyro/N_STEPS_PER_GYRO_rkg,1))                       
        gyroperiods = 4.1e2
        norm_time = npfloat(gyroperiods) * T_gyro

        # -------- plotting parameters -------
        """
        Note, changing these parameters will not change the physics of the above parameters and will
        not cause a new h5 file creation. They will be captured in the summary text file associated with 
        the run.
        """

        USE_PLOT_TITLES = True
        USE_FULL_PLOT = False

        window_time = npfloat(11.6) # only interested in one drift period, SI units
        slice_mode = "first"  
        N_GYRO = 75
        gyro_window = "last"     

        USE_EXTERNAL_H5_ps = False
        USE_EXTERNAL_H5_rk4 = False
        USE_EXTERNAL_H5_rk45 = False
        USE_EXTERNAL_H5_rkg = False

        external_h5_ps = "outputs/outputs_rawdata/" 
        PS_order_ext = 1    # pull from summary text file 
        external_h5_rk4 = "outputs/outputs_rawdata/" 
        external_h5_rk45 = "outputs/outputs_rawdata/" 
        external_h5_rkg = "outputs/outputs_rawdata/" 

    elif run == "paper1": #100 keV proton, 30deg pitch, 5RE, B0 at Earth surface
        if USE_FLOAT128: print("Running PAPER simulation in float128...this may take a >30 minutes\n")
        else: print("Running full PAPER simulation...this can take a few minutes\n")
        output_folder = "outputs/paper2"
        os.makedirs(output_folder, exist_ok=True)

        READ_DATA = False 
        WRITE_DATA = True

        # -------- physics parameters -------
        """
        Note, changing these parameters will results in a new h5 file creation. H5 files are constructed
        and named based on these entries and altering them changes the structure/physics of the data.
        """
        USE_RK45 = True  
        USE_RK4 = True  # removed from paper plots due to failure
        USE_RKG = True
        USE_PS = True
        PS_decimate = 1 # only works with chunking
        PS_CHUNKING = True

        pitch_deg = npfloat(30.0)              
        phi_deg = npfloat(90.0)
        x_initial = npfloat(5)                 
        y_initial = npfloat(0)
        z_initial = npfloat(0)
        KE_particle = npfloat(100e3)              
        mass_si = m_p   
        T_gyro = 2.0 * np.pi * (x_initial**3)    

        # used for paper only, see "tinker" for cleaner approach
        N_STEPS_PER_GYRO_rk4= 65
        N_STEPS_PER_GYRO_ps=65
        N_STEPS_PER_GYRO_rkg=65
        rk4_step = npfloat(round(T_gyro/N_STEPS_PER_GYRO_rk4,1))               
        ps_step = npfloat(round(T_gyro/N_STEPS_PER_GYRO_ps,1))                                  
        rkg_step = npfloat(round(T_gyro/N_STEPS_PER_GYRO_rkg,1))  
        totatl_integration_steps = 5e4
        norm_time = npfloat(totatl_integration_steps) * ps_step
        gyroperiods= npfloat(totatl_integration_steps) * ps_step / T_gyro

        # -------- plotting parameters -------
        """
        Note, changing these parameters will not change the physics of the above parameters and will
        not cause a new h5 file creation. They will be captured in the summary text file associated with 
        the run.
        """

        USE_PLOT_TITLES = False
        USE_FULL_PLOT = False

        window_time = npfloat(6209.0) # only interested in ~one drift period so same as slice
        slice_mode = "last"   
        N_GYRO = 150
        gyro_window = "last"   

        USE_EXTERNAL_H5_ps = False
        USE_EXTERNAL_H5_rk4 = False
        USE_EXTERNAL_H5_rk45 = False
        USE_EXTERNAL_H5_rkg = False

        external_h5_ps = "/Volumes/Extreme SSD/Thesis/PS_method/outputs_rawdata/run_1f911496e90d7c4b.h5" # big PS run
        # PS_order_ext = 16    # old h5 files did not have max_ps captured, new do. Us inspect_hdf5.py to check 
        external_h5_rk4 = "outputs/outputs_rawdata/" 
        external_h5_rk45 = "outputs/outputs_rawdata/" 
        external_h5_rkg = "/Volumes/Extreme SSD/Thesis/PS_method/outputs_rawdata/run_ae2d63d764f68d00.h5"

    elif run == "paper2": #100 MeV electron, 60 degree pitch, 5RE, B0 at Earth surface
        if USE_FLOAT128: print("Running PAPER simulation in float128...this may take a >30 minutes\n")
        else: print("Running full PAPER simulation...this can take a few minutes\n")
        output_folder = "outputs/paper"
        os.makedirs(output_folder, exist_ok=True)

        READ_DATA = True
        WRITE_DATA = True

        # -------- physics parameters -------
        """
        Note, changing these parameters will results in a new h5 file creation. H5 files are constructed
        and named based on these entries and altering them changes the structure/physics of the data.
        """
        USE_RK45 = True  
        USE_RK4 = True
        USE_RKG = True  # does not work for electrons, see paper
        USE_PS = True
        PS_decimate = 1   # only works with chunking
        PS_CHUNKING = True

        pitch_deg = npfloat(60.0)              
        phi_deg = npfloat(90.0)
        x_initial = npfloat(5)                 
        y_initial = npfloat(0)
        z_initial = npfloat(0)
        KE_particle = npfloat(100e6) 
        mass_si = m_e   
        T_gyro = 2.0 * np.pi * (x_initial**3)  

        # used for paper, see "tinker" for cleaner approach
        N_STEPS_PER_GYRO_rk4= 65
        N_STEPS_PER_GYRO_ps=65
        N_STEPS_PER_GYRO_rkg=65
        rk4_step = npfloat(round(T_gyro/N_STEPS_PER_GYRO_rk4,1))               
        ps_step = npfloat(round(T_gyro/N_STEPS_PER_GYRO_ps,1))                                  
        rkg_step = npfloat(round(T_gyro/N_STEPS_PER_GYRO_rkg,1))  
        totatl_integration_steps = 1e7
        norm_time = npfloat(totatl_integration_steps) * ps_step
        gyroperiods= npfloat(totatl_integration_steps) * ps_step / T_gyro

        # -------- plotting parameters -------
        """
        Note, changing these parameters will not change the physics of the above parameters and will
        not cause a new h5 file creation. They will be captured in the summary text file associated with 
        the run.
        """

        USE_PLOT_TITLES = True
        USE_FULL_PLOT = False

        window_time = npfloat(11.6) # only interested in one drift period, SI units 
        slice_mode = "last"  
        N_GYRO = 75
        gyro_window = "last"      

        USE_EXTERNAL_H5_ps = False
        USE_EXTERNAL_H5_rk4 = False
        USE_EXTERNAL_H5_rk45 = False
        USE_EXTERNAL_H5_rkg = False

        external_h5_ps = "outputs/outputs_rawdata/run_5f2698f4194712e0.h5" #big PS run
        external_h5_rk4 = "outputs/outputs_rawdata/" 
        external_h5_rk45 = "outputs/outputs_rawdata/" 
        external_h5_rkg = "outputs/outputs_rawdata/"       

    elif run == "paper3": #paper1 simulation at larger ps_step and smaller rk4_step
        if USE_FLOAT128: print("Running PAPER simulation in float128...this may take a >30 minutes\n")
        else: print("Running full PAPER simulation...this can take a few minutes\n")
        output_folder = "outputs/paper"
        os.makedirs(output_folder, exist_ok=True)

        READ_DATA = True
        WRITE_DATA = True

      # -------- physics parameters -------
        """
        Note, changing these parameters will results in a new h5 file creation. H5 files are constructed
        and named based on these entries and altering them changes the structure/physics of the data.
        """
        USE_RK45 = False  
        USE_RK4 = True 
        USE_RKG = False  
        USE_PS = True
        PS_decimate = 1  # only works with chunking
        PS_CHUNKING = True  

        pitch_deg = npfloat(30.0)              
        phi_deg = npfloat(90.0)
        x_initial = npfloat(5)                 
        y_initial = npfloat(0)
        z_initial = npfloat(0)
        KE_particle = npfloat(100e3)              
        mass_si = m_p   
        T_gyro = 2.0 * np.pi * (x_initial**3)  

        PS_order = 16

        N_STEPS_PER_GYRO_rk4= 1570.8 # 1570.796327 
        N_STEPS_PER_GYRO_ps= 14.5 # 14.544 
        N_STEPS_PER_GYRO_rkg = 65

        rk4_step = npfloat(round(T_gyro/N_STEPS_PER_GYRO_rk4, 1))               
        ps_step = npfloat(round(T_gyro/N_STEPS_PER_GYRO_ps, 1))                                  
        rkg_step = npfloat(round(T_gyro/N_STEPS_PER_GYRO_rkg, 1))  
        norm_time = npfloat(6206.0*4/.00033464094804535314)      # only interested in one drift period so same as slice

        gyroperiods= npfloat(norm_time) / T_gyro

        rk4_step = npfloat(0.5)                
        ps_step = npfloat(54.0)

        # -------- plotting parameters -------
        """
        Note, changing these parameters will not change the physics of the above parameters and will
        not cause a new h5 file creation. They will be captured in the summary text file associated with 
        the run.
        """

        USE_PLOT_TITLES = False
        USE_FULL_PLOT = True

        window_time = npfloat(6209.0) # only interested in ~one drift period so same as slice
        slice_mode = "last"   
        N_GYRO = 150
        gyro_window = "last"   

        USE_EXTERNAL_H5_ps = False
        USE_EXTERNAL_H5_rk4 = False
        USE_EXTERNAL_H5_rk45 = False
        USE_EXTERNAL_H5_rkg = False

        external_h5_ps = "outputs/outputs_rawdata/run_36f4c2c523cdbb17.h5" 
        external_h5_rk4 = "outputs/outputs_rawdata/run_7b2f03f027541ae1.h5" 
        external_h5_rk45 = "outputs/outputs_rawdata/"
        external_h5_rkg = "outputs/outputs_rawdata/"

    elif run == "dragt": # using this one to play with parameters 
        if USE_FLOAT128: 
            print("Running PAPER simulation in float128...this may take >30 minutes\n")
        else: 
            print("Running full PAPER simulation...this can take a few minutes\n")
            
        output_folder = "outputs/dragt"
        os.makedirs(output_folder, exist_ok=True)

        READ_DATA = True 
        WRITE_DATA = True   

        # -------- Physics Parameters --------
        USE_RK45 = False  
        USE_RK4 = False
        USE_RKG = False
        USE_PS = True
        PS_decimate = 1  
        PS_CHUNKING = True

        # Initial Conditions
        pitch_deg   = npfloat(67.61877044327187)    
        phi_deg     = npfloat(-90.0)
        x_initial   = npfloat(3.34483896)
        y_initial   = npfloat(0)
        z_initial   = npfloat(0)
        KE_particle = npfloat(210.5306105318196e6)   
        mass_si = m_p
        
        # --- Relativistic Gyro-Physics Calculation ---
        E_kinetic = KE_particle * abs(q_e)
        E_rest    = mass_si * (spdlight**2)
        gamma     = 1.0 + (E_kinetic / E_rest)
        v_total   = spdlight * np.sqrt(1.0 - (1.0 / gamma**2))
        alpha_rad = np.radians(pitch_deg)
        v_perp    = v_total * np.sin(alpha_rad)

        # --- 2. Calculate the "Actual" Orbit Center (L_eff) ---
        # We find the gyroradius to see how far the particle shifts
        B_at_launch = B_0 / (x_initial**3)
        omega_init  = (abs(q_e) * B_at_launch) / (gamma * mass_si)
        r_g_RE      = (v_perp / omega_init) / RE
        
        # Shift the L-shell based on the phase angle phi
        phi_rad     = np.radians(phi_deg)
        L_eff       = x_initial + (r_g_RE * np.sin(phi_rad))
        # L_eff = 7

        # # --- 3. Set Timing based on the Physical Orbit ---
        # # This ensures the 'Gyroperiod' unit matches what the particle actually feels
        T_gyro_physics = 2.0 * np.pi * (L_eff**3)  

        # Update your step sizes using the physics-based period
        N_STEPS_PER_GYRO_rk4 = 65
        N_STEPS_PER_GYRO_ps  = 65
        N_STEPS_PER_GYRO_rkg = 65
        
        rk4_step = npfloat(T_gyro_physics / N_STEPS_PER_GYRO_rk4)              
        ps_step  = npfloat(T_gyro_physics / N_STEPS_PER_GYRO_ps)                                  
        rkg_step = npfloat(T_gyro_physics / N_STEPS_PER_GYRO_rkg)  

        # Set total duration to exactly 1e5 physical cycles
        gyroperiods = 5e6
        norm_time   = npfloat(gyroperiods) * T_gyro_physics
        
        #Sync back to global variables so the plots use the same unit
        T_gyro = T_gyro_physics

        PS_order = 1000

        # -------- plotting parameters -------
        """
        Note, changing these parameters will not change the physics of the above parameters and will
        not cause a new h5 file creation. They will be captured in the summary text file associated with 
        the run.
        """

        USE_PLOT_TITLES = True
        USE_FULL_PLOT = True

        window_time = npfloat(5)
        slice_mode = "first"   
        N_GYRO = 175
        gyro_window = "first"   

        USE_EXTERNAL_H5_ps = False
        USE_EXTERNAL_H5_rk4 = False
        USE_EXTERNAL_H5_rk45 = False
        USE_EXTERNAL_H5_rkg = False

        external_h5_ps = "outputs/outputs_rawdata/run_304603211ada647c.h5" 
        external_h5_rk4 = "outputs/outputs_rawdata/run_60a0558ebec9f956.h5" 
        external_h5_rk45 = "outputs/outputs_rawdata/run_60a0558ebec9f956.h5" 
        external_h5_rkg = "outputs/outputs_rawdata/run_60a0558ebec9f956.h5"

    elif run == "walt": 
        if USE_FLOAT128: 
            print("Running PAPER simulation in float128...this may take >30 minutes\n")
        else: 
            print("Running full PAPER simulation...this can take a few minutes\n")
            
        output_folder = "outputs/trash"
        os.makedirs(output_folder, exist_ok=True)

        READ_DATA = False 
        WRITE_DATA = True   

        # -------- Physics Parameters --------
        USE_RK45 = False  
        USE_RK4 = False
        USE_RKG = False
        USE_PS = True
        PS_decimate = 1  # WILL MESS UP SURFACE OF SECTIONS Fits
        PS_CHUNKING = True

        # Initial Conditions
        pitch_deg   = npfloat(89.0)    
        phi_deg     = npfloat(0.0)
        x_initial   = npfloat(5.7045989)
        y_initial   = npfloat(0)
        z_initial   = npfloat(0)
        KE_particle = npfloat(1e8)   
        mass_si = m_p
        
        # --- Relativistic Gyro-Physics Calculation ---
        E_kinetic = KE_particle * abs(q_e)
        E_rest    = mass_si * (spdlight**2)
        gamma     = 1.0 + (E_kinetic / E_rest)
        v_total   = spdlight * np.sqrt(1.0 - (1.0 / gamma**2))
        alpha_rad = np.radians(pitch_deg)
        v_perp    = v_total * np.sin(alpha_rad)

        # --- Calculate the "Actual" Orbit Center (L_eff) ---
        B_at_launch = B_0 / (x_initial**3)
        omega_init  = (abs(q_e) * B_at_launch) / (gamma * mass_si)
        r_g_RE      = (v_perp / omega_init) / RE
        
        # Shift the L-shell based on the phase angle phi
        phi_rad     = np.radians(phi_deg)
        L_eff       = x_initial + (r_g_RE * np.sin(phi_rad))

        # --- 3. Set Timing based on the Physical Orbit ---
        T_gyro_physics = 2.0 * np.pi * (L_eff**3)  

        # Update your step sizes using the physics-based period
        N_STEPS_PER_GYRO_rk4 = 65
        N_STEPS_PER_GYRO_ps  = 20
        N_STEPS_PER_GYRO_rkg = 65
        
        rk4_step = npfloat(T_gyro_physics / N_STEPS_PER_GYRO_rk4)              
        ps_step  = npfloat(T_gyro_physics / N_STEPS_PER_GYRO_ps)                                  
        rkg_step = npfloat(T_gyro_physics / N_STEPS_PER_GYRO_rkg)  

        # Set total duration to exactly 1e5 physical cycles
        gyroperiods = 1e4
        norm_time   = npfloat(gyroperiods) * T_gyro_physics
        
        # Sync back to global variables so the plots use the same unit
        T_gyro = T_gyro_physics
        PS_order = 1000

        # -------- plotting parameters -------
        """
        Note, changing these parameters will not change the physics of the above parameters and will
        not cause a new h5 file creation. They will be captured in the summary text file associated with 
        the run.
        """

        USE_PLOT_TITLES = True
        USE_FULL_PLOT = True

        window_time = npfloat(5)
        slice_mode = "first"   
        N_GYRO = 175
        gyro_window = "last"   

        USE_EXTERNAL_H5_ps = False
        USE_EXTERNAL_H5_rk4 = False
        USE_EXTERNAL_H5_rk45 = False
        USE_EXTERNAL_H5_rkg = False

        external_h5_ps = "outputs/outputs_rawdata/run_304603211ada647c.h5" 
        external_h5_rk4 = "outputs/outputs_rawdata/run_60a0558ebec9f956.h5" 
        external_h5_rk45 = "outputs/outputs_rawdata/run_60a0558ebec9f956.h5" 
        external_h5_rkg = "outputs/outputs_rawdata/run_60a0558ebec9f956.h5"

    elif run.startswith("batch"):
        """
        Batch mode: reads parameters from a JSON config file.
        Used by utility_scripts/batch_flux_runner.py for parameter sweeps.
        Does not modify any hardcoded parameters in this file.

        Accepts: "batch"  (reads default utility_scripts/batch_config.json)
                 "batch:/path/to/config.json"  (reads specified config file)
        The colon-path form allows parallel workers to each use their own
        config file without race conditions.
        """
        if ":" in run:
            _config_path = run.split(":", 1)[1]
        else:
            _config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                         "utility_scripts", "batch_config.json")
        with open(_config_path, "r") as _f:
            _cfg = _json.load(_f)

        output_folder = _cfg.get("output_folder", "outputs/flux_map")
        os.makedirs(output_folder, exist_ok=True)

        READ_DATA  = True    # will load from cache if available
        WRITE_DATA = True    # write h5 for trajectory analysis

        USE_RK45 = False
        USE_RK4  = False
        USE_RKG  = False
        USE_PS   = True
        PS_decimate = 1
        PS_CHUNKING = True

        pitch_deg   = npfloat(_cfg["pitch_deg"])
        phi_deg     = npfloat(_cfg.get("phi_deg", 0.0))
        x_initial   = npfloat(_cfg["L_shell"])
        y_initial   = npfloat(0)
        z_initial   = npfloat(0)
        KE_particle = npfloat(_cfg["energy_eV"])
        mass_si     = m_p

        # --- Relativistic gyro-physics ---
        E_kinetic = KE_particle * abs(q_e)
        E_rest    = mass_si * (spdlight**2)
        gamma     = 1.0 + (E_kinetic / E_rest)
        v_total   = spdlight * np.sqrt(1.0 - (1.0 / gamma**2))
        alpha_rad = np.radians(pitch_deg)
        v_perp    = v_total * np.sin(alpha_rad)

        B_at_launch = B_0 / (x_initial**3)
        omega_init  = (abs(q_e) * B_at_launch) / (gamma * mass_si)
        r_g_RE      = (v_perp / omega_init) / RE
        phi_rad     = np.radians(phi_deg)
        L_eff       = x_initial + (r_g_RE * np.sin(phi_rad))

        T_gyro_physics = 2.0 * np.pi * (L_eff**3)
        T_gyro = T_gyro_physics

        N_STEPS_PER_GYRO_ps  = 65
        N_STEPS_PER_GYRO_rk4 = 20
        N_STEPS_PER_GYRO_rkg = 65
        rk4_step = npfloat(T_gyro / N_STEPS_PER_GYRO_rk4)
        ps_step  = npfloat(T_gyro / N_STEPS_PER_GYRO_ps)
        rkg_step = npfloat(T_gyro / N_STEPS_PER_GYRO_rkg)

        gyroperiods = _cfg.get("gyroperiods", 5e4)
        norm_time   = npfloat(gyroperiods) * T_gyro

        PS_order = 1000

        # --- Plotting parameters (minimal for batch — no interactive plots) ---
        USE_PLOT_TITLES = False
        USE_FULL_PLOT   = False

        window_time  = npfloat(5)
        slice_mode   = "first"
        N_GYRO       = 50
        gyro_window  = "last"

        USE_EXTERNAL_H5_ps  = False
        USE_EXTERNAL_H5_rk4 = False
        USE_EXTERNAL_H5_rk45 = False
        USE_EXTERNAL_H5_rkg = False
        external_h5_ps   = "outputs_rawdata/"
        external_h5_rk4  = "outputs_rawdata/"
        external_h5_rk45 = "outputs/outputs_rawdata/"
        external_h5_rkg  = "outputs_rawdata/"

        print(f"BATCH: E={KE_particle/1e6:.0f} MeV  L={x_initial:.2f}  "
              f"pitch={pitch_deg:.1f}°  gyroperiods={gyroperiods:.0e}")

    elif run == "legacy": # using this one to play with parameters
        if USE_FLOAT128: print("Running PAPER simulation in float128...this may take a >30 minutes\n")
        else: print("Running full PAPER simulation...this can take a few minutes\n")
        output_folder = "outputs/outputs_tinker"
        os.makedirs(output_folder, exist_ok=True)
        """
        This allows legacy files to be loaded directly through the 'legacy' run in the test particle function.
        Early runs didn't have all the parameters we are now tracking, so the scanning doesn't work properly.
        The functions take the old h5 files we did have and reconstruct a dictionary in the format we are using now.
        """


        legacy_h5_path = "outputs_rawdata/run_98a2efbd7550732a-1.h5"

        # -------- plotting parameters -------
        """
        Note, changing these parameters will not change the physics of the above parameters and will
        not cause a new h5 file creation. They will be captured in the summary text file associated with 
        the run.
        """
        USE_FULL_PLOT = False
        USE_PLOT_TITLES = False

        window_time = npfloat(6209.0)
        slice_mode = "last"   
        N_GYRO = 150
        gyro_window = "last"   

        USE_EXTERNAL_H5_ps = False
        USE_EXTERNAL_H5_rk4 = False
        USE_EXTERNAL_H5_rk45 = False
        USE_EXTERNAL_H5_rkg = False

        external_h5_ps = "outputs/outputs_rawdata/" 
        external_h5_rk4 = "outputs/outputs_rawdata/" 
        external_h5_rk45 = "outputs/outputs_rawdata/" 
        external_h5_rkg = "outputs/outputs_rawdata/"  

    elif run == "manual": # using this one to play with parameters 
        if USE_FLOAT128: print("Running PAPER simulation in float128...this may take a >30 minutes\n")
        else: print("Running full PAPER simulation...this can take a few minutes\n")
        output_folder = "outputs/outputs_tinker"
        os.makedirs(output_folder, exist_ok=True)

        "This allows manual file load if needed. Must be in current format though"

        manual_h5_path = "outputs/outputs_rawdata/run_60f693535958025f.h5"

        # -------- plotting parameters -------
        """
        Note, changing these parameters will not change the physics of the above parameters and will
        not cause a new h5 file creation. They will be captured in the summary text file associated with 
        the run.
        """

        USE_FULL_PLOT = False
        USE_PLOT_TITLES = False

        window_time = npfloat(5.0)
        slice_mode = "last"   
        N_GYRO = 150
        gyro_window = "last"   

        USE_EXTERNAL_H5_ps = False
        USE_EXTERNAL_H5_rk4 = False
        USE_EXTERNAL_H5_rk45 = False
        USE_EXTERNAL_H5_rkg = False

        external_h5_ps = "outputs/outputs_rawdata/" 
        external_h5_rk4 = "outputs/outputs_rawdata/" 
        external_h5_rk45 = "outputs/outputs_rawdata/" 
        external_h5_rkg = "outputs/outputs_rawdata/"  

    elif run == "monster_ps": #100 keV proton, 30deg pitch, 5RE, B0 at Earth surface
        if USE_FLOAT128: print("Running PAPER simulation in float128...this may take a >30 minutes\n")
        else: print("Running full PAPER simulation...this can take a few minutes\n")
        output_folder = "outputs/outputs_extended_runs/proposal"
        os.makedirs(output_folder, exist_ok=True)

        READ_DATA = False 
        WRITE_DATA = True

       # -------- physics parameters -------
        """
        Note, changing these parameters will results in a new h5 file creation. H5 files are constructed
        and named based on these entries and altering them changes the structure/physics of the data.
        """

        USE_RK45 = False   
        USE_RK4 = False  
        USE_RKG = False
        USE_PS = True
        PS_decimate = 1   # only works with chunking
        PS_CHUNKING = True

        pitch_deg = npfloat(85.0)              
        phi_deg = npfloat(90.0)
        x_initial = npfloat(8)                 
        y_initial = npfloat(0)
        z_initial = npfloat(0)
        KE_particle = npfloat(1e7)              
        mass_si = m_p   
        T_gyro = 2.0 * np.pi * (x_initial**3)  

        N_STEPS_PER_GYRO_rk4= 65
        N_STEPS_PER_GYRO_ps= 65
        N_STEPS_PER_GYRO_rkg= 65
        rk4_step = npfloat(round(T_gyro/N_STEPS_PER_GYRO_rk4,1))               
        ps_step = npfloat(round(T_gyro/N_STEPS_PER_GYRO_ps,1))                                  
        rkg_step = npfloat(round(T_gyro/N_STEPS_PER_GYRO_rkg,1))                         
        gyroperiods = 1e4    #5e8
        norm_time = npfloat(gyroperiods) * T_gyro
        # norm_time= 314159265358.9793 # from paper
        # gyroperiods= npfloat(norm_time) / T_gyro # for paper


        # -------- plotting parameters -------
        """
        Note, changing these parameters will not change the physics of the above parameters and will
        not cause a new h5 file creation. They will be captured in the summary text file associated with 
        the run.
        """

        USE_PLOT_TITLES = False
        USE_FULL_PLOT = False

        window_time = npfloat(6209.0)
        slice_mode = "last"   
        N_GYRO = 150
        gyro_window = "first"   

        USE_EXTERNAL_H5_ps = False
        USE_EXTERNAL_H5_rk4 = False
        USE_EXTERNAL_H5_rk45 = False
        USE_EXTERNAL_H5_rkg = False

        external_h5_ps = "outputs/outputs_rawdata/" 
        external_h5_rk4 = "outputs/outputs_rawdata/" 
        external_h5_rk45 = "outputs/outputs_rawdata/" 
        external_h5_rkg = "outputs/outputs_rawdata/" 

    else:
        raise ValueError("run must be 'demo', 'paper1', 'paper2', or 'paper3'")

    return locals()
