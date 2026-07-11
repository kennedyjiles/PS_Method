"""
Dragt-units → physical calculator.

Given Dragt dimensionless inputs (W0², ρ, ρ̇) at an L-shell, computes the
physical launch conditions: kinetic energy, speed, pitch/phi angles, gyroradius,
gyroperiod, and bounce period, plus the x_initial to paste into the simulator.
The inverse of si_to_dragt.py. Inputs are CLI-overridable (see --help);
defaults reproduce the paper cases.

Usage:
    python scripts/dragt.py
    python scripts/dragt.py --L-shell 4 --rho 1.07
    python scripts/dragt.py --particle electron --wo-squared 0.01
"""
import numpy as np
import argparse

# -----paper inputs (overridable via CLI; defaults preserved below)------
# Examples:
#   python scripts/dragt.py
#   python scripts/dragt.py --L-shell 4 --rho 1.07
#   python scripts/dragt.py --particle electron --wo-squared 0.01
_p = argparse.ArgumentParser(description="Dragt paper input calculator")
_p.add_argument("--particle",   type=str,   default="proton",
                choices=["proton", "electron"],
                help="particle species (default: proton)")
_p.add_argument("--L-shell",    type=float, default=2.0,
                dest="L_shell",
                help="L shell in R_E to center on (default: 2.0; sets physical scale only, not the dynamics)")
# Paper cases (Dragt & Finn 1976). These are dimensionless Stormer-unit inputs
# with NO intrinsic L-shell; L only sets the physical scaling (energy ~ 1/L^4).
#   (W0^2=0.005, rho=1.070):      generic trapped orbit, surface-of-section point P (Fig. 2)
#   (W0^2=0.01,  rho=1.11494632): hyperbolic fixed point h of the Poincare map,
#                                 lambda=2.49 (Fig. 11, homoclinic-point evidence)
_p.add_argument("--wo-squared", type=float, default=0.01,
                dest="wo_squared",
                help="W0^2 in Dragt units (default: 0.01; paper uses .005 and .01)")
_p.add_argument("--rho-dot",    type=float, default=0.0,
                dest="rho_dot",
                help="rho_dot in Dragt units (default: 0.0)")
_p.add_argument("--rho",        type=float, default=1.11494632,
                help="rho in Dragt units (default: 1.11494632; paper uses 1.07 and 1.11494632)")
_args = _p.parse_args()

particle    = _args.particle
L_shell     = _args.L_shell
wo_squared  = _args.wo_squared
rho_dot     = _args.rho_dot
rho         = _args.rho
x_initial   = rho * L_shell

##################################################################################
"""STOP! NOTHING DOWN HERE SHOULD BE CHANGED UNLESS YOU KNOW WHAT YOU'RE DOING."""
##################################################################################

# -----explicit paper equations------
v_phi = (1/rho) - (1/rho**2)
potential = 0.5 * (v_phi**2)             # eqn 11, assumes equatorially launched, r=rho. Assumes z=0 (equatorially launced)
z_dot = np.sqrt(wo_squared-rho_dot**2- 2*potential)   # eqn 22, assumes z=0
print(z_dot) # for wo_squared = 0.005, rho = 1.07 z_dot should be 0.0355, as reported by paper

# -----P_phi verification------
# v_phi = 1/rho - 1/rho^2 is derived from P_phi = -1 (proton), so
# Gamma = 1/(L*R_E) is exact by construction...no GCA approx.
# (The canonical v_phi = 1/rho^2 - 1/rho has opposite sign, but v_phi
# only appears squared in potential and v_perp, so the sign doesn't matter.)
#
# Cross-check: W0^2 and rho must be consistent with the same L_shell.
# Given W0^2, we can recover L independently from the energy:
#   gamma*v = sqrt(W0^2) * v_scale = sqrt(W0^2) * q*M*Gamma^2/m
# If L_shell is wrong, the KE output below will disagree with si_to_dragt.py.
charge_sign = 1 if particle == 'proton' else -1
P_phi_dragt = rho * (1/rho**2 - 1/rho) - charge_sign / rho  # uses canonical sign
print(f"   P_phi (Dragt units): {P_phi_dragt:.6f}  (should be {-charge_sign})")

# -----velocity dissection------
v_parallel = z_dot
v_perp = np.sqrt(rho_dot**2 + v_phi**2)
pitch = np.degrees(np.arctan2(v_perp, v_parallel))

"""
Convert Dragt paper v_phi (downward-B convention) to Earth's upward-B convention.
In Dragt's paper (downward-B moment), protons drift EASTWARD (v_phi > 0).
In Earth's actual field (upward-B), protons drift WESTWARD → flip sign.
Electrons drift OPPOSITE to protons, so no sign flip needed for electrons.
"""
phi_sign = -1 if particle == 'proton' else 1
phi = np.degrees(np.arctan2(phi_sign * v_phi, rho_dot))

# -----constants------
if particle=='proton':
    mass = 1.67262192595e-27           # kg
    m0c2 = 938.27208943e6              # eV
elif particle=='electron':
    mass= 9.1093837139e-31
    m0c2 = 0.51099895e6               # eV

earth_radius = 6378137.0              # m  
q_charge = 1.602176634e-19           # C (magnitude only; sign handled via phi_sign and charge_sign where needed)

B0 = 3.12e-5                         # T  (equatorial surface field, matches dipoleb_testparticles.py)
M_earth = B0 * (earth_radius**3)     # T m^3
speed_light =  299792458             # m/s
gamma_df = 1/(L_shell*earth_radius)  # Dragt's Gamma parameter (NOT Lorentz gamma). This is exact when L_shell
                                     # is the Dragt L defined by P_phi = -1 in normalized units. The GCA
                                     # approximation Gamma ~ 1/(L_gc * R_E) is only used to interpret Gamma
                                     # as an L-shell; the normalization itself requires no GCA assumption.


B_eq = B0/(L_shell**3)
v_scale = q_charge * M_earth * gamma_df**2 / mass # Dragt normalization velocity: v_scale = qM*Gamma^2/m  (m/s), converts dimensionless Dragt velocities to physical m/s


"""
From Dragt eqs 2.7, 2.15d, 2.16:  γmv = qMΓ²·W₀
Therefore W₀·v_scale = γv (relativistic momentum per rest mass), NOT physical speed.
To recover γ and v from γv:  γ = √(1 + (γv/c)²),  v = γv/γ
"""
gamma_v = np.sqrt(wo_squared) * v_scale             # γv = γβc (momentum/rest_mass)
u = gamma_v / speed_light                           # u = γβ (dimensionless)
lorentz_gamma = np.sqrt(1.0 + u**2)                 # γ = √(1 + (γβ)²)
speed = gamma_v / lorentz_gamma                      # physical speed v = γv/γ
beta_v = speed / speed_light                         # β = v/c
E_KE = (lorentz_gamma - 1.0) * m0c2                 # relativistic KE (eV)
E_total = lorentz_gamma * m0c2                       # total energy (eV)


"""
Misc crap that should be USES AS SANITY CHECKS 
v_perp above is dimensionless (Dragt normalized); physical v_perp = v_perp_dragt * v_scale / γ
(same γ correction as total speed: v_scale·v_dragt = γ·v_physical for each component)
"""

v_perp_si = v_perp * v_scale / lorentz_gamma           # physical perpendicular speed (m/s)
gyroradius = (lorentz_gamma * mass * v_perp_si)/(q_charge*B_eq)   # relativistic: r_g = gamma*m*v_perp/(q*B)
gyroperiod = 2 * np.pi * lorentz_gamma * mass / (q_charge * B_eq) # relativistic: T = 2*pi*gamma*m/(q*B)
beta = speed/speed_light
sin_alpha = np.sin(np.radians(pitch))
gc_bounce = 0.117 * (L_shell/beta) * (1-.4635 * (sin_alpha**(3/4))) # paper references numbers on the order of .2, using Walt equation to estimate

print(f"\nFor an L-shell = {L_shell} {particle}, equatorially launched:")
print(f"   Kinetic Energy: {E_KE/(1e6)} MeV")
print(f"   {(speed/speed_light)*100} % c\n")
print(f"   gyroperiod: {gyroperiod} s")
print(f"   gyroradius: {gyroradius} m")
print(f"   v_perp: {v_perp_si} m/s")
print(f"   beta: {beta}\n")
print(f"   bounce: {gc_bounce} s\n")
print(f"   Launch phi:   {phi} degrees")
print(f"   Launch pitch: {pitch} degrees") 
print(f"   Launch at x_initial: {x_initial} R_E\n")




