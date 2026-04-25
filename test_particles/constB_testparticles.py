import numpy as np
import os
import matplotlib.pyplot as mpl

USE_FLOAT128 = False

if USE_FLOAT128: npfloat = np.float128 
else: npfloat = np.float64


# ===== Physical Constants (from shared module) =====
from constants import q_e, m_e, m_p, evtoj

# ========== Tolerances/Truncation =========
PS_order = 40                          # Max Power Series Order, code will truncate  
tol = 1.0 * np.finfo(npfloat).eps      # setting tolerance to machine epsilon to truncate terms 
rtol_rk45 = 1e-8                       # RK45 relative tolerance
atol_rk45 = 1e-10                      # RK45 adapative tolerance

if USE_FLOAT128: mpl.rcParams['agg.path.chunksize'] = 100000  
else: mpl.rcParams['agg.path.chunksize'] = 1000

run_storage = "outputs/outputs_rawdata"      # where trajectory files go

# ===================================================================
# ==============Toggle Parameters for Const B Script ================
# ===================================================================
"""
Toggles:
    USE_RK45       -- Set to True to include RK45 analysis
    USE_RK4        -- Set to True to include RK4 analysis
    USE_ANALYTICAL -- Set to True to include analytical solution (only valid for B_z;
                      set USE_ANALYTICAL=False in constB.py for other field orientations)
    USE_PLOT_TITLES -- Set to True to include plot titles
    USE_FULL_PLOT   -- Set to False for paper plots only, True for all plots
    READ_DATA       -- Set to True to scan for saved runs and load from cache
    WRITE_DATA      -- Set to True to write trajectory data to h5 file
    USE_EXTERNAL_H5 -- Set to True to load external h5 for comparison

Physics parameters:
    pitch_deg       -- pitch angle (degrees)
    phi_deg         -- gyrophase (degrees)
    x_initial       -- initial x position (normalized, typically 0)
    y_initial       -- initial y position (normalized, typically 0)
    z_initial       -- initial z position (normalized, typically 0)
    KE_particle     -- kinetic energy (eV)
    Bfield_si       -- magnetic field vector [Bx, By, Bz] (T)
    mass            -- particle mass: m_e or m_p, or manual (kg)
    gyro_plot_slice -- number of gyroperiods to slice for visual inspection (suggest 1-10)

Integration parameters:
    rk4_step    -- RK4 time step: 2*pi/N where N is steps per gyroperiod
    ps_step     -- PS time step: set equal to rk4_step for direct comparison
    gyroperiods -- number of gyroperiods to simulate
    norm_time   -- total integration time: gyroperiods * 2*pi

Run modes: "demo", "paper"
"""

def load_params(run):
    if run == "paper": 
        print("Running full PAPER simulation...this can take a few minutes\n")
        output_folder = "outputs/outputs_paper/ConstB"
        os.makedirs(output_folder, exist_ok=True)
        USE_RK45 = True
        USE_RK4 = True
        USE_ANALYTICAL = True
        USE_PLOT_TITLES = False
        USE_FULL_PLOT = True
        READ_DATA = True      
        WRITE_DATA = True
        USE_EXTERNAL_H5 = False
        USE_EXTERNAL_H5b = False

        external_h5 = "outputs/outputs_rawdata/run_29620be4f429e0cf.h5" 
        PS_order_ext= 19
        external_h5b = "outputs/outputs_rawdata/run_344c6be65eafa517.h5" 

        pitch_deg = npfloat(45.0)              
        phi_deg = npfloat(45.0)
        x_initial = npfloat(0.0)               
        y_initial = npfloat(0.0)
        z_initial = npfloat(0.0)
        KE_particle = npfloat(100) 
        Bfield_si = np.array([0, 0, npfloat(10e-3)]) 
        mass = m_e
        gyro_plot_slice = 1.5

        rk4_step = npfloat(0.063)              
        ps_step = rk4_step
        # rk4_step = npfloat(0.008)              
        # ps_step = npfloat(1.26)              
        gyroperiods = 1e5                     
        norm_time = gyroperiods *2 * np.pi
        

    elif run == "demo":
        print("Running DEMO simulation...this should be done in a couple seconds\n")
        output_folder = "outputs/outputs_demo/ConstB"
        os.makedirs(output_folder, exist_ok=True)
        USE_RK45 = True
        USE_RK4 = True
        USE_ANALYTICAL = True
        USE_PLOT_TITLES = True
        USE_FULL_PLOT = True
        READ_DATA = False      
        WRITE_DATA = True 
        USE_EXTERNAL_H5 = False
        USE_EXTERNAL_H5b = False

        external_h5 = None
        external_h5b = None

        pitch_deg = npfloat(45.0)              
        phi_deg = npfloat(45.0)
        x_initial = npfloat(0.0)               
        y_initial = npfloat(0.0)
        z_initial = npfloat(0.0)
        KE_particle = npfloat(100) 
        Bfield_si = np.array([0, 0, npfloat(10e-3)]) 
        mass = m_p
        gyro_plot_slice = 1.5

        rk4_step = npfloat(0.063)              
        ps_step = rk4_step
        gyroperiods = 10.0                     
        norm_time = gyroperiods *2 * np.pi
 
    else:
        raise ValueError("run must be 'paper' or 'demo'")

    return locals()
