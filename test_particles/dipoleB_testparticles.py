import numpy as np
import os

USE_FLOAT128 = False  # RKG Will be diabled if this is True

if USE_FLOAT128: npfloat = np.float128 
else: npfloat = np.float64

# ===== System Constants =====
q_e = npfloat(-1.602176634e-19)        # C
m_e = npfloat(9.1093837139e-31)        # kg
m_p = npfloat(1.67262192595e-27)       # kg
evtoj = npfloat(1.602176634e-19)       # eV to J
spdlight = npfloat(299792458.0)
RE = npfloat(6378137.0)                # m, Radius of Earth

# ===== Tolerances/Truncations =====
PS_order = 40                           # Max Power Series Order, system will truncate
# tol = 1.0 * np.finfo(npfloat).eps       # setting tolerance to machine epsilon to drop terms later
tol = 1.0e-45
rtol_rk45 = 1e-8                    # RK45 relative tolerance
atol_rk45 = 1e-10                   # RK45 adapative tolerance
user_min_gap = npfloat(10)          # start with 10, adjust as needed
user_min_phase = npfloat(.1)        # this is the minimum phase it's looking for to be allowed to extrap from for drift


# ===================================================================
# ==============Toggle Parameters for Hyper B Script ================
# ===================================================================
"""
USE_RK45 --  Set to True to include RK45 analysis
USE_RK4 --   Set to True to include RK4 analysis
USE_RKG --   Set to True to include RKG analysis
READ_DATA -- Set to True to scan for saved runs and load
WRITE_DATA -- Set to True to write saved run data to hdf file (what READ_DATA looks for)
USE_PLOT_TITLES -- Set to True to include plot titles
USE_FULL_PLOT -- Set to True to plot entire trajectory (only useful for short runs, large runs slice last orbits)

pitch_deg -- (degrees)
phi_deg  -- (degrees)
x_initial_si -- (Earth Radius: RE)
y_initial_si -- (RE)
z_initial_si -- (RE)
KE_particle -- (eV)
B_0 -- (T)
mass_si -- m_e or m_p, otherwise manual (kg)
gyro_plot_slice -- slices last gyroperiods for visual inspection, suggest 500 (note gyroperiod based on B_0)

rk4_step -- 2π/N where N is ~integration points per gyroperiod               
ps_step --  set equal to rk4_step for one-to-one comparison, can be anything
norm_time -- this should be some multiple of gyroperiods designed (norm_time/2π = gyroperiods), 
    calculated drift/tau_0 to plot one cycle (you'll need to run once)     
"""

def load_params(run):
    if run == "demo":
        if USE_FLOAT128: print("Running full PAPER simulation in float128...this may take a >30 minutes\n")
        else: print("Running full PAPER simulation...this can take a few minutes\n")
        output_folder = "dipoleB_outputs_paper"
        os.makedirs(output_folder, exist_ok=True)
        USE_RK45 = True  
        USE_RK4 = True 
        USE_RKG = True  
        USE_PLOT_TITLES = True
        READ_DATA = False
        WRITE_DATA = True
        USE_FULL_PLOT = True

        pitch_deg = npfloat(30.0)              
        phi_deg = npfloat(90.0)
        x_initial = npfloat(5)                 
        y_initial = npfloat(0)
        z_initial = npfloat(0)
        KE_particle = npfloat(100e3)              
        B_0 = npfloat(3.12e-5)  
        mass_si = m_p   
        gyro_plot_slice = 500
                       
        rk4_step = npfloat(12.1)                
        ps_step = rk4_step                      
        rkg_step = rk4_step
        norm_time = (1e5) * ps_step



    elif run == "paper":
        print("Running DEMO simulation...this should be done in a couple seconds\n")
        output_folder = "dipoleB_outputs_demo"
        os.makedirs(output_folder, exist_ok=True)
          

    else:
        raise ValueError("run must be 'paperx' or 'demo'")

    return locals()    

