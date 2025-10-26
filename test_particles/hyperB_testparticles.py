import numpy as np
import os

USE_FLOAT128 = True

if USE_FLOAT128: npfloat = np.float128 
else: npfloat = np.float64

# ===== Constants =====
q_e = npfloat(-1.602176634e-19)        # C
m_e = npfloat(9.1093837139e-31)        # kg
m_p = npfloat(1.67262192595e-27)       # kg
evtoj = npfloat(1.602176634e-19)       # eV to J

# ===== Tolerances/Truncations =====
PS_order = 40                           # Max Power Series Order, system will truncate
tol = 1.0 * np.finfo(npfloat).eps       # setting tolerance to machine epsilon to drop terms later
rtol_rk45 = 1e-12                       # RK45 relative tolerance
atol_rk45 = 1e-14                       # RK45 adapative tolerance

# ===================================================================
# ==============Toggle Parameters for Hyper B Script ================
# ===================================================================
"""
USE_RK45 --  Set to True to include RK45 analysis
USE_RK4 --   Set to True to include RK4 analysis
READ_DATA -- Set to True to scan for saved runs and load
WRITE_DATA -- Set to True to write saved run data to hdf file (what READ_DATA looks for)
USE_PLOT_TITLES -- Set to True to include plot titles
USE_FULL_PLOT -- Set to True to plot entire trajectory (only useful for short runs, large runs slice last orbits)

pitch_deg -- (degrees)
phi_deg  -- (degrees)
delta -- (km)
x_initial_si -- (km)
y_initial_si -- (km)
z_initial_si -- (km)
KE_particle -- (eV)
B_0 -- (T)
mass_si -- m_e or m_p, otherwise manual (kg)
gyro_plot_slice -- slices last gyroperiods for visual inspection, suggest 8-15 (note gyroperiod based on B_0)

rk4_step -- 2π/N where N is ~integration points per gyroperiod               
ps_step --  set equal to rk4_step for one-to-one comparison, can be anything
norm_time -- this should be some multiple of gyroperiods designed (norm_time/2π = gyroperiods)     
"""

def load_params(run):
    if run == "paper1":
        if USE_FLOAT128: print("Running full PAPER simulation in float128...this may take a ~30 minutes\n")
        else: print("Running full PAPER simulation...this can take a few minutes\n")
        output_folder = "outputs_hyperB_paper"
        os.makedirs(output_folder, exist_ok=True)
        USE_RK45 = True       
        USE_RK4 = True        
        READ_DATA = False      
        WRITE_DATA = True     
        USE_PLOT_TITLES = False
        USE_FULL_PLOT = False 

        pitch_deg = npfloat(75.0)
        phi_deg = npfloat(45.0)
        delta = 500                             
        x_initial_si = npfloat(0.0)             
        y_initial_si = npfloat(0.25 * delta)
        z_initial_si = npfloat(0.0)
        KE_particle = npfloat(10e3)             
        B_0 = npfloat(10e-9)                    
        mass_si = m_e    
        gyro_plot_slice = 8

        rk4_step = npfloat(0.063)               
        ps_step = rk4_step             
        norm_time = (1e6) * ps_step         

    elif run == "demo":
        print("Running DEMO simulation...this should be done in a couple seconds\n")
        output_folder = "outputs_hyperB_demo"
        os.makedirs(output_folder, exist_ok=True)
        USE_RK45 = True        
        USE_RK4 = True         
        READ_DATA = False      
        WRITE_DATA = False      
        USE_PLOT_TITLES = True 
        USE_FULL_PLOT = True 
        
        pitch_deg = npfloat(75.0)
        phi_deg = npfloat(45.0)
        delta = 500                             
        x_initial_si = npfloat(0.0)             
        y_initial_si = npfloat(0.25 * delta)
        z_initial_si = npfloat(0.0)
        KE_particle = npfloat(10e3)             
        B_0 = npfloat(10e-9)                    
        mass_si = m_e  
        gyro_plot_slice = 8        
 
        rk4_step = npfloat(0.063)               
        ps_step = rk4_step
        norm_time = 50 * 2 * np.pi

    elif run == "paper2":
        print("Running full PAPER simulation...this can take a few minutes\n")
        output_folder = "outputs_hyperB_paper"
        os.makedirs(output_folder, exist_ok=True)
        USE_RK45 = True        
        USE_RK4 = True         
        READ_DATA = True      
        WRITE_DATA = True      
        USE_PLOT_TITLES = False 
        USE_FULL_PLOT = False 

        pitch_deg = npfloat(-15.0)
        phi_deg = npfloat(45.0)
        delta = 200                             
        x_initial_si = npfloat(0.0)             
        y_initial_si = npfloat(0.25 * delta)
        z_initial_si = npfloat(0.0)
        KE_particle = npfloat(10e3)             
        B_0 = npfloat(10e-9)                    
        mass_si = m_e     
        gyro_plot_slice = 8    

        rk4_step = npfloat(0.063)               
        ps_step = rk4_step             
        norm_time = (1e6) * ps_step     


    elif run == "paper3":
        print("Running full PAPER3 simulation...this can take a few minutes\n")
        output_folder = "outputs_hyperB_paper"
        os.makedirs(output_folder, exist_ok=True)
        USE_RK45 = True        
        USE_RK4 = True         
        READ_DATA = True      
        WRITE_DATA = True      
        USE_PLOT_TITLES = False 
        USE_FULL_PLOT = False 

        pitch_deg = npfloat(-15.0)
        phi_deg = npfloat(45.0)
        delta = 200                             
        x_initial_si = npfloat(0.0)             
        y_initial_si = npfloat(0.01 * delta)
        z_initial_si = npfloat(0.0)
        KE_particle = npfloat(100e3)             
        B_0 = npfloat(10e-9)                    
        mass_si = m_p   
        gyro_plot_slice = 8    

        rk4_step = npfloat(0.063)               
        ps_step = rk4_step
        norm_time = (1e6) * ps_step 


    elif run == "paper4":
        print("Running full PAPER simulation...this can take a few minutes\n")
        output_folder = "outputs_hyperB_paper"
        os.makedirs(output_folder, exist_ok=True)
        USE_RK45 = True       
        USE_RK4 = True        
        READ_DATA = True      
        WRITE_DATA = True     
        USE_PLOT_TITLES = False
        USE_FULL_PLOT = False  

        pitch_deg = npfloat(75.0)
        phi_deg = npfloat(45.0)
        delta = 500                             
        x_initial_si = npfloat(0.0)             
        y_initial_si = npfloat(0.25 * delta)
        z_initial_si = npfloat(0.0)
        KE_particle = npfloat(10e3)             
        B_0 = npfloat(10e-9)                    
        mass_si = m_e
        gyro_plot_slice = 8        
    
        rk4_step = npfloat(0.063)               
        ps_step = npfloat(0.63)              
        norm_time = (1e6) * rk4_step # same norm time as paper 1   

    else:
        raise ValueError("run must be 'paperx' or 'demo'")

    return locals()    

