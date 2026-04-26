import numpy as np
import os
import matplotlib as mpl

USE_FLOAT128 = False

if USE_FLOAT128: npfloat = np.float128 
else: npfloat = np.float64

# ===== Physical Constants (from shared module) =====
from ps_method.constants import q_e, m_e, m_p, evtoj

# ===== Tolerances/Truncations =====
PS_order = 40                           # Max Power Series Order, system will truncate
tol = 1.0 * np.finfo(npfloat).eps       # setting tolerance to machine epsilon to drop terms later
rtol_rk45 = 1e-12                       # RK45 relative tolerance
atol_rk45 = 1e-14                       # RK45 adapative tolerance

if USE_FLOAT128: mpl.rcParams['agg.path.chunksize'] = 100000  
else: mpl.rcParams['agg.path.chunksize'] = 100

run_storage = "outputs/outputs_rawdata"        # where raw trajectory files go when USE_WRITE_DATA = True

# ===================================================================
# ==============Toggle Parameters for Hyper B Script ================
# ===================================================================
"""
Toggles:
    USE_RK45        -- Set to True to include RK45 analysis
    USE_RK4         -- Set to True to include RK4 analysis
    READ_DATA       -- Set to True to scan for saved runs and load from cache
    WRITE_DATA      -- Set to True to write trajectory data to h5 file
    USE_PLOT_TITLES -- Set to True to include plot titles
    USE_FULL_PLOT   -- Set to True for full trajectory plot (only useful for short runs)
    USE_EXTERNAL_H5 -- Set to True to load external h5 for comparison

Physics parameters:
    pitch_deg    -- pitch angle (degrees)
    phi_deg      -- gyrophase (degrees)
    delta        -- current sheet half-thickness (km)
    x_initial_si -- initial x position (km)
    y_initial_si -- initial y position (km)
    z_initial_si -- initial z position (km)
    KE_particle  -- kinetic energy (eV)
    B_0          -- asymptotic magnetic field strength (T)
    mass_si      -- particle mass: m_e or m_p, or manual (kg)

Integration parameters:
    rk4_step    -- RK4 time step: 2*pi/N where N is steps per gyroperiod
    ps_step     -- PS time step: set equal to rk4_step for direct comparison
    gyroperiods -- number of gyroperiods to simulate
    norm_time   -- total integration time: gyroperiods * 2*pi

Plotting parameters:
    window_duration -- time window (normalized) for trajectory slice inspection
    slice_mode      -- "last" or "first", slices from end or beginning of simulation

Run modes: "demo", "paper1", "paper2", "paper3", "paper4"     
"""

def load_params(run):
    if run == "demo": #paper1 simulation at reduced norm time for quick demo
        print("Running DEMO simulation...this should be done in a couple seconds\n")
        output_folder = "outputs/outputs_demo"
        os.makedirs(output_folder, exist_ok=True)
        USE_RK45 = True        
        USE_RK4 = True         
        READ_DATA = True      
        WRITE_DATA = True      
        USE_PLOT_TITLES = True 
        USE_FULL_PLOT = False 

        USE_EXTERNAL_H5 = False
        USE_EXTERNAL_H5b = False

        external_h5 = "outputs/outputs_rawdata/" 
        PS_order_ext = 1
        external_h5b = "outputs/outputs_rawdata/" 
        PS_order_ext = 1

        
        pitch_deg = npfloat(75.0)
        phi_deg = npfloat(45.0)
        delta = 500                             
        x_initial_si = npfloat(0.0)             
        y_initial_si = npfloat(0.25 * delta)
        z_initial_si = npfloat(0.0)
        KE_particle = npfloat(10e3)             
        B_0 = npfloat(10e-9)                    
        mass_si = m_e  

        window_duration = npfloat(8*2*np.pi) # only interested in a couple gyroperiods
        slice_mode = "last"        
 
        rk4_step = npfloat(0.063)               
        ps_step = rk4_step             
        gyroperiods = 1e2
        norm_time = (gyroperiods) * 2 * np.pi  

    elif run == "paper1": # 10 keV electron, 75deg pitch, delta=500km, B0=10nT
        if USE_FLOAT128: print("Running full PAPER simulation in float128...this may take a ~30 minutes\n")
        else: print("Running full PAPER simulation...this will take a few minutes\n")
        output_folder = "outputs/outputs_paper"
        os.makedirs(output_folder, exist_ok=True)
        USE_RK45 = True       
        USE_RK4 = True        
        READ_DATA = True      
        WRITE_DATA = True     
        USE_PLOT_TITLES = False
        USE_FULL_PLOT = False 
        USE_EXTERNAL_H5 = False
        USE_EXTERNAL_H5b = False

        external_h5 = "outputs/outputs_rawdata/run_8d05c562f1137db2.h5" 
        PS_order_ext = "25"
        external_h5b = "outputs/outputs_rawdata/run_2e64db24ec88cb7e.h5" 
        PS_order_extb = "10"


        pitch_deg = npfloat(75.0)
        phi_deg = npfloat(45.0)
        delta = 500                             
        x_initial_si = npfloat(0.0)             
        y_initial_si = npfloat(0.25 * delta)
        z_initial_si = npfloat(0.0)
        KE_particle = npfloat(10e3)             
        B_0 = npfloat(10e-9)                    
        mass_si = m_e    

        window_duration = npfloat(8*2*np.pi) # only interested in a couple gyroperiods
        slice_mode = "last"   

        rk4_step = npfloat(0.063)               
        ps_step = rk4_step             
        gyroperiods = 1e5                 
        norm_time = (gyroperiods) * 2 * np.pi   

    elif run == "paper2": # 10 keV electron, 75deg pitch, delta=50km, B0=10nT
        if USE_FLOAT128: print("Running full PAPER simulation in float128...this may take a ~30 minutes\n")
        else: print("Running full PAPER simulation...this will take a few minutes\n")
        output_folder = "outputs/outputs_paper"
        os.makedirs(output_folder, exist_ok=True)
        USE_RK45 = True        
        USE_RK4 = True        
        READ_DATA = True      
        WRITE_DATA = True     
        USE_PLOT_TITLES = False 
        USE_FULL_PLOT = False 
        USE_EXTERNAL_H5 = False
        USE_EXTERNAL_H5b = False

    
        external_h5 = "outputs/outputs_rawdata/run_6fc9daec43008056.h5"
        PS_order_ext = "40"
        external_h5b = "outputs/outputs_rawdata/run_73287ee231150e97.h5" 
        PS_order_extb = "10" 

        pitch_deg = npfloat(75.0)
        phi_deg = npfloat(10.0)
        delta = 50                             
        x_initial_si = npfloat(0.0)             
        y_initial_si = npfloat(.1 * delta)
        z_initial_si = npfloat(0.0)
        KE_particle = npfloat(10e3)             
        B_0 = npfloat(10e-9)                    
        mass_si = m_e    

        window_duration = npfloat(8*2*np.pi) # only interested in a couple gyroperiods
        slice_mode = "last"   

        rk4_step = npfloat(0.0315)               
        ps_step = rk4_step     
        gyroperiods = 1e5               
        norm_time = (gyroperiods) * 2 * np.pi     

    elif run == "paper3": # 100 keV proton, -15deg pitch, delta=200km, B0=10nT
        if USE_FLOAT128: print("Running full PAPER simulation in float128...this may take a ~30 minutes\n")
        else: print("Running full PAPER simulation...this will take a few minutes\n")
        output_folder = "outputs/outputs_paper"
        os.makedirs(output_folder, exist_ok=True)
        USE_RK45 = True        
        USE_RK4 = True         
        READ_DATA = True      
        WRITE_DATA = True      
        USE_PLOT_TITLES = False 
        USE_FULL_PLOT = False 
        USE_EXTERNAL_H5 = False
        USE_EXTERNAL_H5b = False


        external_h5 = "outputs/outputs_rawdata/run_d72c9d579dd24595.h5" 
        PS_order_ext = "40"
        external_h5b = "outputs/outputs_rawdata/run_bbb58e0c8a29e72f.h5"
        PS_order_extb = "15"



        pitch_deg = npfloat(-15.0)
        phi_deg = npfloat(45.0)
        delta = 200                             
        x_initial_si = npfloat(0.0)             
        y_initial_si = npfloat(0.01 * delta)
        z_initial_si = npfloat(0.0)
        KE_particle = npfloat(100e3)             
        B_0 = npfloat(10e-9)                    
        mass_si = m_p   

        window_duration = npfloat(8*2*np.pi) # only interested in a couple gyroperiods
        slice_mode = "last"   

        rk4_step = npfloat(0.063)               
        ps_step = rk4_step
        gyroperiods = 1e5               
        norm_time = (gyroperiods) * 2 * np.pi   

    elif run == "paper4": # paper1 simulation at larger ps_step
        if USE_FLOAT128: print("Running full PAPER simulation in float128...this may take a ~30 minutes\n")
        else: print("Running full PAPER simulation...this will take a few minutes\n")
        output_folder = "outputs/outputs_paper"
        os.makedirs(output_folder, exist_ok=True)
        USE_RK45 = True       
        USE_RK4 = True        
        READ_DATA = True      
        WRITE_DATA = True     
        USE_PLOT_TITLES = False
        USE_FULL_PLOT = False  
        USE_EXTERNAL_H5 = False
        USE_EXTERNAL_H5b = False

        external_h5 = "outputs/outputs_rawdata/" 
        external_h5b = "outputs/outputs_rawdata/" 


        pitch_deg = npfloat(75.0)
        phi_deg = npfloat(45.0)
        delta = 500                             
        x_initial_si = npfloat(0.0)             
        y_initial_si = npfloat(0.25 * delta)
        z_initial_si = npfloat(0.0)
        KE_particle = npfloat(10e3)             
        B_0 = npfloat(10e-9)                    
        mass_si = m_e

        window_duration = npfloat(8*2*np.pi) # only interested in a couple gyroperiods
        slice_mode = "last"   

        rk4_step = npfloat(0.063)               
        ps_step = npfloat(0.63)
        gyroperiods = 1e5               
        norm_time = (gyroperiods) * 2 * np.pi                

    else:
        raise ValueError("run must be 'demo', 'paper1', 'paper2', 'paper3', or 'paper4'")

    return locals()    

