#############################################################
# This script can help you determine a time step for either 
# the dipole or hyperbolic case based on the Larmor radius
#############################################################

#############################################################
# This script computes the normalized time step (dtau)
# based on the local gyroperiod for a given pitch and field
#############################################################

import importlib.util
import numpy as np

npfloat = np.float64
c = npfloat(299792458.0)
evtoj = npfloat(1.602176634e-19)

# === Prompt for file path ===
file_path = input("Enter path to particle file (e.g., particle1.py): ")

# === Prompt for field type ===
field_type = ""
while field_type.lower() not in ["d", "h"]:
    field_type = input("Choose field type: dipole (d) or hyperbolic (h): ").strip().lower()

# === Dynamically import particle config ===
spec = importlib.util.spec_from_file_location("particle", file_path)
particle = importlib.util.module_from_spec(spec)
spec.loader.exec_module(particle)

# === Extract particle params ===
pitch_rad = np.radians(particle.pitch_deg)
x = particle.x_initial
y = particle.y_initial
z = particle.z_initial
mass = particle.mass_si if hasattr(particle, 'mass_si') else particle.mass
q = particle.q_e
B_0 = particle.B_0
KE_joules = particle.KE_particle * evtoj
gamma = 1.0 + KE_joules / (mass * c**2)
v_si = c * np.sqrt(1.0 - 1.0 / gamma**2)

# === Determine local B field ===
if field_type == "d":
    r2 = x**2 + y**2 + z**2
    B_loc = B_0 * (1 / np.sqrt(r2))**3
    print("Dipole field selected.")
elif field_type == "h":
    if not hasattr(particle, 'alpha'):
        raise ValueError("Missing `alpha` in particle file for hyperbolic field.")
    alpha = particle.alpha
    B_loc = B_0 * np.tanh(alpha * y)
    print("Hyperbolic field selected.")

# === Compute perpendicular velocity and gyro quantities ===
v_perp_si = v_si * np.sin(pitch_rad)

# === Gyroperiod in SI units at current B_loc ===
T_gyro_si = 2 * np.pi * mass / (abs(q) * B_loc)

# === Time normalization factor ===
tau_time = mass / (abs(q) * B_0)  # consistent with normalization

# === Normalized local gyroperiod ===
T_gyro_tau = T_gyro_si / tau_time

# === Current step size from particle file (normalized) ===
if hasattr(particle, 'ps_step'):
    ps_step = particle.ps_step
    N_actual = T_gyro_tau / ps_step
    print(f"Actual ps_step = {ps_step:.5e} → {N_actual:.2f} steps/gyroperiod at starting B_loc")
else:
    print("No ps_step found in particle file.")

# === For reference: what dtau would be at fixed N ===
for N in [65, 100, 200]:
    dtau_ref = T_gyro_tau / N
    print(f"Reference: N = {N:3d} steps/gyro → dtau = {dtau_ref:.5e} (normalized)")

