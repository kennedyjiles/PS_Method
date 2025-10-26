import numpy as np

npfloat = np.float64                 # Use to Toggle between 64 and 128, note 128 is very slow

# ===== System Constants =====
q_e = npfloat(-1.602176634e-19)        # C
m_e = npfloat(9.1093837139e-31)        # kg
m_p = npfloat(1.67262192595e-27)       # kg
mass = m_p                               # m_e or m_p
charge = q_e if mass == m_e else abs(q_e)

# ========== Scaling ==========
system_scale = npfloat(1)                # scales system for everything and all methods
scale_time = npfloat(1)                  # increase to stretch dynamics

# ===== Number of Steps  =====
rk4_step = npfloat(12.57)               # .01 is good start
ps_step = rk4_step/scale_time            # set equal for one-to-one comparison, can be anything
rkg_step = rk4_step

# ===== System Parameters =====
norm_time = (1e3) * ps_step
PS_order = 25                          # Max Power Series Order, system will truncate
KE_particle = npfloat(100e3)            # eV
evtoj = npfloat(1.602176634e-19)        # eV to J
KE_joules = KE_particle * evtoj

B_0 = npfloat(3.12e-5)                # T

# ===== Initial Conditions (Position & Vel ocity) =====
x_initial = npfloat(5)                 # all in RE
y_initial = npfloat(0)
z_initial = npfloat(0)
pitch_deg = npfloat(90.0)              
phi_deg = npfloat(90.0)

qoverm = npfloat(-1) if mass == m_e else npfloat(1)
