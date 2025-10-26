import numpy as np

npfloat = np.float64                 # Use to Toggle between 64 and 128, note 128 is very slow

# ===== System Constants =====
q_e = npfloat(-1.602176634e-19)        # C
m_e = npfloat(9.1093837139e-31)        # kg
m_p = npfloat(1.67262192595e-27)       # kg
RE = npfloat(6378137.0)                # m


# ========== Scaling ==========
system_scale=npfloat(1)                # scales system for everything and all methods
scale_time=npfloat(1)                  # increase to stretch dynamics


# ===== System Parameters =====
norm_time=10000                         # 100,000 used in paper
PS_order = 50                           # Max Power Series Order, system will truncate
KE_particle = npfloat(10e3)             # eV e: 10s eV to >100 keV for sheets, p: 100s eV to 100s keV
B_0 = npfloat(10e-9)                    # T, use 5-20nT....10nT used in paper
alpha= npfloat(100)                      # alpha= 1/gamma, gamma= .015-.3 for real world sheets
mass=m_e                                # m_e or m_p


# ===== Number of Steps  =====
rk4_step = npfloat(0.01)                # .01 used in paper
ps_step=rk4_step/scale_time             # set equal for one-to-one comparison, can be anything
# symp_step=npfloat(.01)


# ===== Initial Conditions (Position & Velocity) =====
x_initial = npfloat(0.08)               # all in units of RE
y_initial = npfloat(0.0101)
z_initial = npfloat(0.1)
pitch_deg = npfloat(-17.5)
phi_deg = npfloat(45.0)


qoverm = npfloat(-1) if mass == m_e else npfloat(1)

