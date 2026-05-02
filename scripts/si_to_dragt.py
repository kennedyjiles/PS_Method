"""
SI-to-Dragt Converter
=====================
Given physical inputs (KE, pitch angle, launch phi, target L-shell,
particle type), computes:
  - x_initial such that the conserved canonical momentum gives L_dragt = L_target exactly
  - Dragt dimensionless parameters (W0^2, rho, P_phi)
  - Simulator initial conditions ready to paste into dipoleb_testparticles.py
  - Trapping and stability diagnostics

The launch position is computed from the exact constraint:
    P_phi = x * v_phi - charge_sign / x = -charge_sign / L_target
This is a quadratic in x with an analytic solution — no iteration needed.
The approximate guiding center position is also reported for reference.

Usage:
    Edit the USER INPUTS section below, then run:
        python si_to_dragt.py
"""
import numpy as np

# ==========================
#       USER INPUTS
# ==========================
L_target    = 5.7              # Dragt L-shell 
KE_MeV      = 100           # Kinetic energy in MeV
pitch_deg   = 60.0           # Equatorial pitch angle in degrees
phi_deg     = 0.0           # degrees
particle    = 'proton'       # 'proton' or 'electron'



##################################################################################
"""STOP! NOTHING DOWN HERE SHOULD BE CHANGED UNLESS YOU KNOW WHAT YOU'RE DOING."""
##################################################################################


#-----constants------
if particle == 'proton':
    mass = 1.67262192595e-27      # kg
    m0c2 = 938.27208943e6           # eV
elif particle == 'electron':
    mass = 9.1093837139e-31        # kg
    m0c2 = 0.51099895e6           # eV
else:
    raise ValueError(f"Unknown particle: {particle}")

charge_sign  = 1 if particle == 'proton' else -1
q_charge     = 1.602176634e-19      # C
earth_radius = 6378137.0          # m 
B0           = 3.12e-5             # T 
M_earth      = B0 * earth_radius**3
speed_light  = 299792458.0         # m/s

# ==========================
#   RELATIVISTIC KINEMATICS
# ==========================
KE_eV         = KE_MeV * 1e6
lorentz_gamma = 1.0 + KE_eV / m0c2
speed         = speed_light * np.sqrt(1.0 - 1.0 / lorentz_gamma**2)
beta          = speed / speed_light
pitch_rad     = np.radians(pitch_deg)
phi_rad       = np.radians(phi_deg)

v_perp_si = speed * np.sin(pitch_rad)
v_par_si  = speed * np.cos(pitch_rad)

# ==========================
#   SIMULATOR VELOCITY
# ==========================
# These depend only on KE and pitch, not on x_initial
tau_0 = lorentz_gamma * mass / (q_charge * B0)
v_tau = speed * tau_0 / earth_radius

v_par_sim  = v_tau * np.cos(pitch_rad)
v_perp_sim = v_tau * np.sin(pitch_rad)

# v_phi at launch (y=0, so v_phi = vy)
vy_launch = v_perp_sim * np.sin(phi_rad)

# ==========================
#   SOLVE FOR x_initial
# ==========================
# Constraint: P_phi = x * vy - charge_sign / x = -charge_sign / L_target
#
# Multiply by x:  vy * x^2 + (charge_sign / L_target) * x - charge_sign = 0
#
# Quadratic:  a = vy,  b = charge_sign / L_target,  c = -charge_sign
# x = (-b ± sqrt(b^2 - 4ac)) / (2a)
# Both roots satisfy P_phi, but only one is near L_target (the physical root).
# For protons the + root is physical; for electrons the - root is physical.
# We compute both and pick the one closest to L_target.

a_coeff = vy_launch
b_coeff = charge_sign / L_target
c_coeff = -charge_sign

# When phi = 0 or 180, vy_launch ~ 0 and the quadratic degenerates to a
# linear equation:  b*x + c = 0  =>  x = -c/b = L_target.
# This is the physically obvious case: no azimuthal velocity means the
# canonical angular momentum is purely from A_phi, so x_initial = L_target.
if abs(a_coeff) < 1e-30:
    x_initial = -c_coeff / b_coeff   # = charge_sign / (charge_sign/L_target) = L_target
    if x_initial <= 0:
        print(f"\n  ERROR: Linear solve gave non-physical x={x_initial:.4f} for L={L_target}")
        print(f"  with {KE_MeV} MeV {particle}, pitch={pitch_deg}, phi={phi_deg}.\n")
        raise SystemExit(1)
    print(f"  [Note: phi={phi_deg} => vy=0, linear solve gives x_initial = {x_initial:.6f} R_E]")
else:
    stuff_in_sqrt = b_coeff**2 - 4.0 * a_coeff * c_coeff

    if stuff_in_sqrt < 0:
        print(f"\n  ERROR: No solution exists for L={L_target}")
        print(f"  with {KE_MeV} MeV {particle}, pitch={pitch_deg}, phi={phi_deg}.")
        print(f"  Try a different phi or lower energy.\n")
        raise SystemExit(1)

    sqrt_disc = np.sqrt(stuff_in_sqrt)
    x_plus  = (-b_coeff + sqrt_disc) / (2.0 * a_coeff)
    x_minus = (-b_coeff - sqrt_disc) / (2.0 * a_coeff)

    # This is picking the positive root closest to L_target...so far seems to work, other answer is nonsensical (negative or very far from L_target)
    candidates = [(x, abs(x - L_target)) for x in (x_plus, x_minus) if x > 0]
    if not candidates:
        print(f"\n  ERROR: No positive root for L={L_target}")
        print(f"  with {KE_MeV} MeV {particle}, pitch={pitch_deg}, phi={phi_deg}.")
        print(f"  Try a different phi or lower energy.\n")
        raise SystemExit(1)

    x_initial = min(candidates, key=lambda t: t[1])[0]

# Full velocity vector at launch
vx_initial = v_perp_sim * np.cos(phi_rad)
vy_initial = vy_launch
vz_initial = v_par_sim

# Clean tiny trig values
if abs(vx_initial) < np.finfo(float).eps: vx_initial = 0.0
if abs(vy_initial) < np.finfo(float).eps: vy_initial = 0.0
if abs(vz_initial) < np.finfo(float).eps: vz_initial = 0.0

# ==========================
#   VERIFY P_phi
# ==========================
P_phi_sim = x_initial * vy_initial - charge_sign / x_initial
L_dragt   = -charge_sign / P_phi_sim

# Approximate guiding center (for reference)
def gyroradius_RE(x_RE):
    """Gyroradius in R_E at equatorial position x_RE."""
    B_local = B0 / x_RE**3
    r_g_si  = lorentz_gamma * mass * v_perp_si / (q_charge * B_local)
    return r_g_si / earth_radius

r_g_final  = gyroradius_RE(x_initial)
x_gc_approx = x_initial + charge_sign * r_g_final * np.sin(phi_rad)

# ==========================
#   DRAGT PARAMETERS
# ==========================
# Dragt normalization using L_dragt (= L_target by construction)
Gamma_d   = 1.0 / (L_dragt * earth_radius)
v_scale_d = q_charge * M_earth * Gamma_d**2 / mass

# W0^2
gamma_v = lorentz_gamma * speed
W0_sq   = (gamma_v / v_scale_d)**2

# Dragt dimensionless position and velocity
rho_dragt     = x_initial / L_dragt
v_phi_dragt   = vy_initial * L_dragt**2
rho_dot_dragt = vx_initial * L_dragt**2   # at y=0: v_rho = vx

# Canonical momentum in Dragt units (should be -1 for proton)
P_phi_dragt = rho_dragt * v_phi_dragt - charge_sign / rho_dragt

# Trapping and stability
trapped      = (charge_sign * P_phi_dragt < 0)
W0_threshold = P_phi_dragt**4 / 16.0 if trapped else None
boundary     = "CLOSED" if (trapped and W0_sq < W0_threshold) else "OPEN"

# mu^2 = sin^2(alpha_eq) = v_perp^2 / v^2
v_perp_sq_0  = vx_initial**2 + vy_initial**2
v_total_sq_0 = v_perp_sq_0 + vz_initial**2
mu_sq          = v_perp_sq_0 / v_total_sq_0 if v_total_sq_0 > 0 else 0.0
stab_threshold = 0.012 * mu_sq
orbit_char     = "REGULAR" if W0_sq < stab_threshold else "CHAOTIC"

# Physical orbit parameters
B_eq     = B0 / L_dragt**3
gyroperiod = 2.0 * np.pi * lorentz_gamma * mass / (q_charge * B_eq)
sin_alpha  = np.sin(pitch_rad)
gc_bounce  = 0.117 * (L_dragt / beta) * (1.0 - 0.4635 * sin_alpha**0.75)

################################################################################
"""Everything you could possibly need to bounce between Dragt and real units"""
###############################################################################

print(f"\n{'='*60}")
print(f"  SI -> Dragt Conversion")
print(f"{'='*60}")

print(f"\n--- Physical Inputs ---")
print(f"  Particle:       {particle}")
print(f"  L_target:       {L_target}")
print(f"  KE:             {KE_MeV} MeV")
print(f"  Pitch angle:    {pitch_deg} deg")
print(f"  Launch phi:     {phi_deg} deg")
print(f"  Lorentz gamma:  {lorentz_gamma:.6f}")
print(f"  Speed:          {speed:.4e} m/s  ({beta*100:.2f}% c)")

print(f"\n--- Launch Position ---")
print(f"  x_initial:      {x_initial:.6f} R_E")
print(f"  Gyroradius:     {r_g_final:.4f} R_E  ({r_g_final*earth_radius:.4e} m)")
print(f"  Approx GC:      {x_gc_approx:.4f} R_E")

print(f"\n--- Dragt Parameters (L_dragt = {L_dragt:.6f}) ---")
print(f"  W0^2:           {W0_sq:.8f}")
print(f"  rho:            {rho_dragt:.6f}")
print(f"  rho_dot:        {rho_dot_dragt:.6f}")
print(f"  P_phi:          {P_phi_dragt:.6f}")

print(f"\n--- Trapping & Stability ---")
print(f"  Boundary:       {boundary}", end="")
if W0_threshold is not None:
    print(f"  (W0^2={W0_sq:.6f} vs threshold={W0_threshold:.6f})")
else:
    print(f"  (no trapping barrier)")
print(f"  mu^2:           {mu_sq:.6f}  (sin^2 alpha_eq)")
print(f"  Orbit character: {orbit_char}  (W0^2={W0_sq:.6f}, threshold={stab_threshold:.6f})")

print(f"\n--- Simulator Initial Conditions (paste into testparticles) ---")
print(f"  x_initial   = npfloat({x_initial})")
print(f"  y_initial   = npfloat(0)")
print(f"  z_initial   = npfloat(0)")
print(f"  pitch_deg   = npfloat({pitch_deg})")
print(f"  phi_deg     = npfloat({phi_deg})")
print(f"  KE_particle = npfloat({KE_MeV*1e6})")

print(f"\n--- Physical Orbit Parameters ---")
print(f"  Gyroperiod:     {gyroperiod:.4e} s")
print(f"  Bounce period:  {gc_bounce:.4f} s  (Walt approx)")
print(f"  v_scale:        {v_scale_d:.4e} m/s")
print(f"  L_dragt:        {L_dragt:.6f} R_E")
print(f"{'='*60}\n")
