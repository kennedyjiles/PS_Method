"""
Round-trip test: Dragt paper ↔ dipoleB.py conversions (CORRECTED per Dragt 1965).

Key identity from paper eqs 2.7 + 2.15d + 2.16:
    γmv = qMΓ² · W₀
    γv  = v_scale · W₀     (where v_scale = qMΓ²/m, rest mass)

So W₀·v_scale = γv (relativistic momentum per rest mass), NOT physical speed.
    γ = √(1 + (W₀·v_scale/c)²)
    v = W₀·v_scale / γ

And: v_dragt = v_sim · L²  (NO gamma division)
"""
import numpy as np
import sys
sys.path.insert(0, "/sessions/elegant-brave-darwin/mnt/code/PS_Method/functions")
from functions_library_dragt import calculate_w0_squared

RE = 6378137.0; B0 = 3.12e-5; M = B0*RE**3
q = 1.602176634e-19; m_p = 1.67262192595e-27; m_e = 9.1093837139e-31
c = 299792458.0; m0c2_p = 938.27208943e6; m0c2_e = 0.51099895e6
evtoj = 1.602176634e-19

print("="*90)
print("CORRECTED ROUND-TRIP TEST (per Dragt 1965)")
print("="*90)
all_pass = True; threshold = 1e-10

proton_cases = [
    (0.005,  1.07, 0.0,  7, "Paper: L=7, W0^2=0.005"),
    (0.01,   1.07, 0.0,  7, "Paper: L=7, W0^2=0.01"),
    (0.005,  1.07, 0.0,  4, "Inner belt: L=4 (mildly rel.)"),
    (0.005,  1.07, 0.0,  3, "Deep inner: L=3 (rel.)"),
    (0.005,  1.07, 0.0,  2, "Very inner: L=2 (highly rel.)"),
    (0.005,  1.07, 0.02, 7, "Non-zero rho_dot"),
    (0.0625, 1.07, 0.0,  7, "Near boundary: W0^2=P^4/16"),
]

for i, (W0sq, rho, rho_dot, L, label) in enumerate(proton_cases):
    mass = m_p; m0c2 = m0c2_p; charge_sign = 1
    Gamma = 1.0/(L*RE)
    v_scale = q*M*Gamma**2/mass

    # --- FORWARD: Dragt → SI (corrected dragt.py) ---
    v_phi = (1/rho) - (1/rho**2)
    z_dot_sq = W0sq - rho_dot**2 - v_phi**2
    if z_dot_sq < 0:
        print(f"  A{i+1}: SKIP ({label}) — z_dot^2 < 0"); continue
    z_dot = np.sqrt(z_dot_sq)
    v_dragt_total = np.sqrt(rho_dot**2 + v_phi**2 + z_dot**2)
    v_dragt_perp = np.sqrt(rho_dot**2 + v_phi**2)

    # CORRECTED: W₀·v_scale = γv, so γ = √(1+(γv/c)²)
    gamma_v = v_dragt_total * v_scale     # = γv
    u = gamma_v / c
    gamma_correct = np.sqrt(1.0 + u**2)
    speed = gamma_v / gamma_correct       # physical v
    KE_eV = (gamma_correct - 1.0) * m0c2

    pitch_deg = np.degrees(np.arctan2(v_dragt_perp, z_dot))
    phi_deg = np.degrees(np.arctan2(-v_phi, rho_dot)) if (rho_dot != 0 or v_phi != 0) else -90.0
    x_initial = rho * L

    # --- REVERSE: SI → dipoleB.py → Dragt ---
    KE_J = KE_eV * evtoj
    gamma_sim = 1.0 + KE_J/(mass*c**2)
    v_si = c*np.sqrt(1.0 - 1.0/gamma_sim**2)
    tau_time = gamma_sim*mass/(q*B0)
    v_tau = v_si*tau_time/RE

    pitch_rad = np.radians(pitch_deg)
    phi_rad = np.radians(phi_deg)
    vx = v_tau*np.sin(pitch_rad)*np.cos(phi_rad)
    vy = v_tau*np.sin(pitch_rad)*np.sin(phi_rad)
    vz = v_tau*np.cos(pitch_rad)

    # v_dragt = v_sim · L² (NO gamma division)
    v_mag_sim = np.sqrt(vx**2 + vy**2 + vz**2)
    W0sq_rec = (v_mag_sim * L**2)**2
    rho_d = abs(x_initial)/L
    v_phi_sim = (x_initial*vy)/abs(x_initial)
    v_phi_d = v_phi_sim * L**2
    P_phi_rec = rho_d * v_phi_d - charge_sign/rho_d

    # calculate_w0_squared (takes physical speed, returns W₀² with γ correction)
    w0sq_func = calculate_w0_squared(speed, L, mass, q, M_earth=M)

    # v_perp check
    v_perp_dragt_si = v_dragt_perp * v_scale / gamma_correct  # corrected
    v_perp_dipole_si = v_si * np.sin(pitch_rad)

    # Errors
    w0_err = abs(W0sq_rec - W0sq)/W0sq
    vp_err = abs(v_perp_dragt_si - v_perp_dipole_si)/v_perp_dragt_si
    func_err = abs(w0sq_func - W0sq)/W0sq
    gamma_err = abs(gamma_sim - gamma_correct)/gamma_correct

    passed = (w0_err < threshold) and (vp_err < threshold) and (func_err < threshold) and (gamma_err < threshold)
    if not passed: all_pass = False

    print(f"  A{i+1}: {label}")
    print(f"    KE={KE_eV/1e6:.4f} MeV | γ={gamma_correct:.8f} | β={speed/c*100:.4f}%c")
    print(f"    W0^2: {W0sq} -> {W0sq_rec:.10f} (err={w0_err:.2e})")
    print(f"    γ round-trip: {gamma_correct:.10f} -> {gamma_sim:.10f} (err={gamma_err:.2e})")
    print(f"    v_perp: dragt={v_perp_dragt_si:.2f}, dipoleB={v_perp_dipole_si:.2f} (err={vp_err:.2e})")
    print(f"    calc_w0sq: {w0sq_func:.10f} (err={func_err:.2e})")
    print(f"    >>> {'PASS' if passed else 'FAIL'}")
    print()

# --- ELECTRON PATH (from SI → Dragt) ---
print("="*90)
print("ELECTRON PATH: SI -> dipoleB -> Dragt (corrected)")
print("="*90)

electron_cases = [
    (1.0e6, 59.84, -90.0, 7.49, 7, "1 MeV e-, L~7"),
    (5.0e6, 59.84, -90.0, 7.49, 7, "5 MeV e-, L~7"),
]
for i, (KE_eV, pitch_deg, phi_deg, x_init, L, label) in enumerate(electron_cases):
    mass = m_e; charge_sign = -1
    KE_J = KE_eV * evtoj
    gamma_sim = 1.0 + KE_J/(mass*c**2)
    v_si = c*np.sqrt(1.0 - 1.0/gamma_sim**2)
    tau_time = gamma_sim*mass/(q*B0)
    v_tau = v_si*tau_time/RE

    pitch_rad = np.radians(pitch_deg)
    phi_rad = np.radians(phi_deg)
    vx = v_tau*np.sin(pitch_rad)*np.cos(phi_rad)
    vy = v_tau*np.sin(pitch_rad)*np.sin(phi_rad)
    vz = v_tau*np.cos(pitch_rad)

    v_mag = np.sqrt(vx**2+vy**2+vz**2)
    W0sq = (v_mag * L**2)**2  # corrected: no gamma division
    rho_d = abs(x_init)/L
    v_phi_sim = (x_init*vy)/abs(x_init)
    v_phi_d = v_phi_sim * L**2
    P_phi = rho_d * v_phi_d - charge_sign/rho_d
    trapped = charge_sign*P_phi < 0
    W0_thresh = P_phi**4/16 if trapped else None

    # calculate_w0_squared
    w0sq_func = calculate_w0_squared(v_si, L, mass, q, M_earth=M)
    func_err = abs(w0sq_func - W0sq)/W0sq
    passed = func_err < threshold
    if not passed: all_pass = False

    print(f"  B{i+1}: {label}")
    print(f"    γ={gamma_sim:.6f} | speed={v_si/c*100:.4f}%c")
    print(f"    W0^2 (v_sim·L²)² = {W0sq:.10f}")
    print(f"    W0^2 (calc func)  = {w0sq_func:.10f} (err={func_err:.2e})")
    print(f"    P_phi={P_phi:.8f} | {'TRAPPED' if trapped else 'OPEN'}")
    if W0_thresh: print(f"    W0_thresh={W0_thresh:.8f} | {'CLOSED' if W0sq < W0_thresh else 'OPEN'}")
    print(f"    >>> {'PASS' if passed else 'FAIL'}")
    print()

print("="*90)
print("ALL TESTS PASSED" if all_pass else "SOME TESTS FAILED")
print("="*90)
