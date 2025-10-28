import numpy as np
import os

USE_FLOAT128 = False

if USE_FLOAT128: npfloat = np.float128 
else: npfloat = np.float64


# ===== Mass and Charge Constants =====
q_e = npfloat(-1.602176634e-19)        # C
m_e = npfloat(9.1093837139e-31)        # kg
m_p = npfloat(1.67262192595e-27)       # kg
evtoj = npfloat(1.602176634e-19)       # eV to J

# ========== Tolerances/Truncation =========
PS_order = 40                          # Max Power Series Order, code will truncate  
tol = 1.0 * np.finfo(npfloat).eps      # setting tolerance to machine epsilon to truncate terms 
rtol_rk45 = 1e-8                       # RK45 relative tolerance
atol_rk45 = 1e-10                      # RK45 adapative tolerance


# ===================================================================
# ==============Toggle Parameters for Const B Script ================
# ===================================================================
"""
USE_RK45 --  Set to True to include RK45 analysis
USE_RK4 --   Set to True to include RK4 analysis
USE_Analytical -- Set to True to include RK4 analysis if using anything other than B_z, set USE_ANALYTICAL=False in constB.py, it was only
set up for B_z. All other methods fine.
USE_PLOT_TITLES -- Set to True to include plot titles
USE_FULL_PLOT -- Set to False for paper plots only, Set to True to enable all plots (not all are useful)


pitch_deg -- (degrees)
phi_deg  -- (degrees)
x_initial_si -- (m)
y_initial_si -- (m)
z_initial_si -- (m)
KE_particle -- (eV)
B_0 -- (T)
mass_si -- m_e or m_p, otherwise (kg)
gyro_plot_slice -- slices last gyroperiods for visual inspection, suggest 1-10 


rk4_step -- 2π/N where N is ~integration points per gyroperiod               
ps_step --  set equal to rk4_step for one-to-one comparison, can be anything
norm_time -- this should be some multiple of gyroperiods desired (norm_time/2π = gyroperiods)     
"""

def load_params(run):
    if run == "paper": 
        print("Running full PAPER simulation...this can take a few minutes\n")
        output_folder = "outputs_paper"
        os.makedirs(output_folder, exist_ok=True)
        USE_RK45 = True
        USE_RK4 = True
        USE_ANALYTICAL = True
        USE_PLOT_TITLES = False
        USE_FULL_PLOT = False
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
        norm_time = (1e6) * ps_step            

    elif run == "demo":
        print("Running DEMO simulation...this should be done in a couple seconds\n")
        output_folder = "outputs_demo"
        os.makedirs(output_folder, exist_ok=True)
        USE_RK45 = True
        USE_RK4 = True
        USE_ANALYTICAL = True
        USE_PLOT_TITLES = True
        USE_FULL_PLOT = True
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
        norm_time = 10 * 2 * np.pi 
 
    else:
        raise ValueError("run must be 'paper' or 'demo'")

    return locals()