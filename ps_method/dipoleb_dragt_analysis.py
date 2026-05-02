"""
Dragt (1965) diagnostic analysis for dipole trajectories.

    conservation_monitor  — per-step W0^2 / P_phi drift monitor
    compute_adiabaticity  — epsilon = r_g |grad B| / B along a trajectory
    compute_params        — W0^2, P_phi, boundary status from initial conditions
    compute_boundary      — accessible boundary curve for Poincaré section
    compute_z_crossings   — equatorial (z=0) crossing finder via interpolation
    compute_gyrophase_mu  — gyrophase and mu at equatorial crossings
    compute_w0_squared    — W0^2 from physical speed and L-shell
    analysis_chunked      — stream h5 file for adiabaticity, meridian, crossings
    run_section           — full Dragt analysis + plots, returns dragt_log
"""

import numpy as np
import warnings
import h5py

from ps_method.constants import RE, spdlight, B_0


# ===================================================================
# === W0^2 / P_phi Conservation Monitor (per-step, no dt needed) ===
# ===================================================================
class conservation_monitor:
    """
    Monitors conservation of W0^2 and P_phi during PS integration.

    These are computed from the instantaneous state vector (positions and
    velocities) with no dependence on dt.  Any drift indicates numerical
    energy injection or removal by the integrator.

    Usage:
        mon = conservation_monitor(L_shell, charge_sign,
                           check_every=100, rtol=1e-6)
        # inside chunk loop:
        mon.check(sol_chunk)          # checks last column of chunk
        # after run:
        mon.summary()                 # prints drift report
        drift = mon.get_drift()       # returns dict with max drift info
    """

    def __init__(self, L_shell, charge_sign=1, check_every=1,
                 rtol=1e-6, halt_on_escape=False):
        """
        Parameters:
            L_shell       : L-shell used for Dragt normalization
            charge_sign   : +1 proton, -1 electron
            check_every   : check every N-th call to check() (1 = every call)
            rtol          : relative tolerance for warning
            halt_on_escape: if True, raise RuntimeError when W0^2 crosses
                            the trapping threshold
        """
        self.L = L_shell
        self.cs = charge_sign
        self.check_every = max(1, int(check_every))
        self.rtol = rtol
        self.halt_on_escape = halt_on_escape

        # Reference values (set on first call)
        self.W0sq_0 = None
        self.Pphi_0 = None
        self.W0_threshold = None

        # History (step_index, W0^2, P_phi)
        self.history = []
        self._call_count = 0

    @staticmethod
    def _compute_invariants(state, L, cs):
        """Compute W0^2 and P_phi from a 6-element state vector
        [x, y, z, vx, vy, vz] in simulator-normalized units."""
        x, y, z, vx, vy, vz = state[0], state[1], state[2], \
                                state[3], state[4], state[5]
        # W0^2 = (v_mag * L^2)^2
        v_mag = np.sqrt(vx**2 + vy**2 + vz**2)
        W0sq = (v_mag * L**2)**2

        # P_phi = rho_dragt * v_phi_dragt - charge_sign * rho_dragt^2 / r_dragt^3
        # The vector potential in dimensionless units is A_phi = -charge_sign * rho/r^3
        # (NOT -charge_sign/rho, which is only valid at the equator where r=rho).
        rho_sim = np.sqrt(x**2 + y**2)
        r_sim = np.sqrt(x**2 + y**2 + z**2)
        rho_d = rho_sim / L
        r_d = r_sim / L
        v_phi_sim = (x * vy - y * vx) / rho_sim if rho_sim > 0 else 0.0
        v_phi_d = v_phi_sim * L**2
        Pphi = rho_d * v_phi_d - cs * (rho_d**2 / r_d**3) if r_d > 0 else 0.0

        return float(W0sq), float(Pphi)

    def check(self, sol_chunk, step_index=None):
        """
        Check conservation using the LAST column of sol_chunk (shape 17 x N).
        Optionally pass the global step index for logging.

        Parameters:
            sol_chunk  : array of shape (>=6, N) — uses last column
            step_index : int or None (for logging)

        Returns:
            True if invariants are within tolerance, False if drift detected.
        """
        self._call_count += 1
        if self._call_count % self.check_every != 0:
            return True

        state = sol_chunk[0:6, -1]
        W0sq, Pphi = self._compute_invariants(state, self.L, self.cs)

        # Initialize reference on first call
        if self.W0sq_0 is None:
            self.W0sq_0 = W0sq
            self.Pphi_0 = Pphi
            trapped = (self.cs * Pphi < 0)
            self.W0_threshold = (Pphi**4 / 16) if trapped else None

        idx = step_index if step_index is not None else self._call_count
        self.history.append((idx, W0sq, Pphi))

        # Compute relative drift
        ok = True
        dW = abs(W0sq - self.W0sq_0) / max(abs(self.W0sq_0), 1e-30)
        dP = abs(Pphi - self.Pphi_0) / max(abs(self.Pphi_0), 1e-30)

        if dW > self.rtol or dP > self.rtol:
            ok = False
            warnings.warn(
                f"conservation_monitor step {idx}: W0^2 drift {dW:.2e} "
                f"(ref {self.W0sq_0:.6f} -> {W0sq:.6f}), "
                f"P_phi drift {dP:.2e} "
                f"(ref {self.Pphi_0:.6f} -> {Pphi:.6f})",
                RuntimeWarning, stacklevel=2,
            )

        # Check for escape past trapping threshold
        if (self.halt_on_escape and self.W0_threshold is not None
                and W0sq > self.W0_threshold):
            msg = (f"conservation_monitor step {idx}: W0^2 = {W0sq:.6f} "
                   f"EXCEEDED trapping threshold {self.W0_threshold:.6f}. "
                   f"Particle has numerically escaped.")
            raise RuntimeError(msg)

        return ok

    def summary(self):
        """Print a summary of conservation quality."""
        if not self.history:
            print("conservation_monitor: no data collected.")
            return

        steps = [h[0] for h in self.history]
        W0s   = np.array([h[1] for h in self.history])
        Pphis = np.array([h[2] for h in self.history])

        dW_rel = np.abs(W0s - self.W0sq_0) / max(abs(self.W0sq_0), 1e-30)
        dP_rel = np.abs(Pphis - self.Pphi_0) / max(abs(self.Pphi_0), 1e-30)

        print(f"\n{'='*60}")
        print(f"  Conservation Monitor Report")
        print(f"{'='*60}")
        print(f"  Samples:        {len(self.history)}")
        print(f"  Step range:     {steps[0]} .. {steps[-1]}")
        print(f"  W0^2 initial:   {self.W0sq_0:.8f}")
        print(f"  W0^2 final:     {W0s[-1]:.8f}")
        print(f"  W0^2 max drift: {dW_rel.max():.2e}  (step {steps[np.argmax(dW_rel)]})")
        print(f"  P_phi initial:  {self.Pphi_0:.8f}")
        print(f"  P_phi final:    {Pphis[-1]:.8f}")
        print(f"  P_phi max drift:{dP_rel.max():.2e}  (step {steps[np.argmax(dP_rel)]})")
        if self.W0_threshold is not None:
            margin = (self.W0_threshold - W0s[-1]) / self.W0_threshold
            print(f"  Trap threshold: {self.W0_threshold:.8f}")
            print(f"  Margin remaining: {margin*100:.2f}%")
        print(f"{'='*60}\n")

    def get_drift(self):
        """Return a dict with drift statistics for programmatic use."""
        if not self.history:
            return {}
        W0s   = np.array([h[1] for h in self.history])
        Pphis = np.array([h[2] for h in self.history])
        return {
            "W0sq_initial": self.W0sq_0,
            "W0sq_final":   float(W0s[-1]),
            "W0sq_max":     float(W0s.max()),
            "W0sq_min":     float(W0s.min()),
            "Pphi_initial": self.Pphi_0,
            "Pphi_final":   float(Pphis[-1]),
            "W0_threshold": self.W0_threshold,
            "history":      self.history,
        }


def compute_adiabaticity(x_arr, y_arr, z_arr, vx_arr, vy_arr, vz_arr):
    """
    Calculates the adiabaticity parameter epsilon = r_g * |grad B| / B
    along a trajectory, using the full dipole gradient (not the equatorial
    approximation).

    For a magnetic dipole, the exact result is:
        |grad B| / B  =  (3/r) * sqrt(8*sin^4(lam) + 7*sin^2(lam) + 1)
                                / (1 + 3*sin^2(lam))
    where lam is the magnetic latitude.  At the equator (lam=0) this reduces
    to the familiar 3/r.  At high latitudes the gradient is steeper.

    All inputs are in normalized units: positions in R_E, velocities in v_tau.
    In these units, the gyroradius works out to r_g [R_E] = v_perp / B_normalized,
    with no additional constants required.

    Parameters:
        x_arr, y_arr, z_arr   : position arrays (R_E)
        vx_arr, vy_arr, vz_arr: velocity arrays (v_tau, dimensionless)

    Returns:
        epsilon (array): dimensionless adiabaticity parameter at each timestep
    """
    r = np.sqrt(x_arr**2 + y_arr**2 + z_arr**2)
    r5 = r**5

    # Dipole B field components in normalized units (downward moment / upward B at equator)
    # Sign convention matches the simulator (lorentz_force): Bz = +1/r^3 at equator
    Bx = -3.0 * x_arr * z_arr / r5
    By = -3.0 * y_arr * z_arr / r5
    Bz = -(3.0 * z_arr**2 - r**2) / r5
    B_mag = np.sqrt(Bx**2 + By**2 + Bz**2)

    # B unit vector
    Bx_hat = Bx / B_mag
    By_hat = By / B_mag
    Bz_hat = Bz / B_mag

    # v_parallel (projection onto B)
    v_par = vx_arr * Bx_hat + vy_arr * By_hat + vz_arr * Bz_hat

    # v_perp (clip to zero to guard against small floating-point negatives)
    v_sq = vx_arr**2 + vy_arr**2 + vz_arr**2
    v_perp = np.sqrt(np.maximum(v_sq - v_par**2, 0.0))

    r_g = v_perp / B_mag

    # Full dipole |grad B|/B using magnetic latitude
    # sin(lam) = z/r,  sin^2(lam) = z^2/r^2
    sin2_lam = z_arr**2 / np.maximum(r**2, 1e-30)
    sin4_lam = sin2_lam**2
    # |grad B|/B = (3/r) * sqrt(8*sin^4 + 7*sin^2 + 1) / (1 + 3*sin^2)
    grad_B_over_B = (3.0 / r) * np.sqrt(8.0*sin4_lam + 7.0*sin2_lam + 1.0) \
                    / (1.0 + 3.0*sin2_lam)

    epsilon = r_g * grad_B_over_B

    return epsilon


def compute_params(x_ps, y_ps, z_ps, vx_ps, vy_ps, vz_ps, L_shell, charge_sign=1):
    """
    Computes Dragt diagnostic parameters from the initial conditions of a PS trajectory.

    Parameters:
        x_ps, y_ps, z_ps    : position arrays (R_E, normalized)
        vx_ps, vy_ps, vz_ps : velocity arrays (v_tau, normalized)
        L_shell             : L-shell used for Dragt normalization
        charge_sign         : +1 for proton, -1 for electron (default +1)

    Returns:
        dict with keys: W0_sq, rho_0_sim, rho_0_dragt, P_phi,
                        W0_threshold (None if orbit is open), boundary_status

    Notes:
        The conversion v_dragt = v_sim * L^2 has NO gamma division.
        From Dragt (1965) eqs 2.15a-d: the simulator time unit tau_0 = gamma*m/(|q|B0)
        already contains gamma, so v_sim = v_physical * gamma*m/(|q|B0*R_E).
        The Dragt dimensionless velocity is v_dragt = v_physical * gamma*m*L^2/(|q|B0*R_E)
        = v_sim * L^2.  The gamma factors cancel in the ratio of time/space scalings.
    """
    v_mag_0 = np.sqrt(vx_ps[0]**2 + vy_ps[0]**2 + vz_ps[0]**2)
    # v_dragt = v_sim * L^2 (no gamma division — gamma is already in v_sim via tau_0)
    W0_sq = (v_mag_0 * L_shell**2)**2

    rho_0_sim    = np.sqrt(x_ps[0]**2 + y_ps[0]**2)
    rho_0_dragt  = rho_0_sim / L_shell
    v_phi_0_sim  = (x_ps[0]*vy_ps[0] - y_ps[0]*vx_ps[0]) / rho_0_sim
    v_phi_0_dragt = v_phi_0_sim * L_shell**2            # same: no gamma division
    # Upward B-field convention: A_phi = -1/rho
    # P_phi = rho*v_phi - charge_sign/rho
    #   Proton  (charge_sign=+1): P_phi = rho*v_phi - 1/rho  (< 0 when trapped)
    #   Electron (charge_sign=-1): P_phi = rho*v_phi + 1/rho  (> 0 when trapped)
    # Trapped condition (both species): charge_sign * P_phi < 0
    # Note: P_phi is dominated by the 1/rho (vector potential) term; the kinetic
    # contribution (rho*v_phi) is small for trapped particles near the thalweg.
    P_phi = (rho_0_dragt * v_phi_0_dragt) - (charge_sign / rho_0_dragt)

    trapped = (charge_sign * P_phi < 0)
    W0_threshold    = P_phi**4 / 16 if trapped else None
    boundary_status = "CLOSED" if (trapped and W0_sq < W0_threshold) else "OPEN"

    # --- mu^2: equatorial pitch-angle parameter (Dragt 1965, eq 3.10) ---
    # At z=0 the dipole B is along z, so v_perp^2 = vx^2 + vy^2.
    # The L^2 conversion factor cancels in the ratio v_perp^2/v^2.
    v_perp_sq_0 = vx_ps[0]**2 + vy_ps[0]**2
    v_total_sq_0 = v_perp_sq_0 + vz_ps[0]**2
    mu_sq = v_perp_sq_0 / v_total_sq_0 if v_total_sq_0 > 0 else 0.0
    # Dragt (1965) empirical stability boundary (eq 6.1, Fig 32):
    #   W0^2 < 0.012 * mu^2   => regular (smooth Poincare section)
    #   W0^2 > 0.012 * mu^2   => chaotic (scattered)
    stability_threshold = 0.012 * mu_sq
    orbit_character = "REGULAR" if W0_sq < stability_threshold else "CHAOTIC"

    # rho_dot in Dragt units at launch: radial velocity component
    v_rho_0_sim = (x_ps[0]*vx_ps[0] + y_ps[0]*vy_ps[0]) / rho_0_sim if rho_0_sim > 0 else 0.0
    rho_dot_0_dragt = v_rho_0_sim * L_shell**2

    print(f"Dragt dimensionless energy (W0^2)     :{W0_sq:.8f}")
    print(f"Dragt dimensionless position  (rho)   :{rho_0_dragt:.6f}")
    print(f"Dragt radial velocity  (rho_dot)      :{rho_dot_0_dragt:.15f}")
    print(f"Canonical Momentum (P_phi)            :{P_phi:.6f}\n")
    if W0_threshold is not None:
        print(f"(Boundary closed if charge_sign*P_phi<0 and W0^2 < P_phi^4/16)\n   Boundary status    :{boundary_status}  (threshold={W0_threshold:.6f})")
    else:
        print(f"(Boundary closed if charge_sign*P_phi<0 and W0^2 < P_phi^4/16)\n   Boundary status:   OPEN  (no trapping barrier for this charge/momentum)")
    print(f"Pitch-angle parameter (Dragt 1965 eq 3.10 mu^2)     :{mu_sq:.6f}  (pitch_eq={np.degrees(np.arcsin(np.sqrt(mu_sq))):.2f} deg)")
    print(f"Stability (W0^2 < 0.012*mu^2 => regular, Dragt 1965 eq 6.1) character    :{orbit_character}  (W0^2={W0_sq:.6f}, threshold={stability_threshold:.6f}, ratio={W0_sq/stability_threshold:.1f}x)" if stability_threshold > 0 else f"(Stability, W0^2 < 0.012*mu^2 => regular, Dragt 1965 eq 6.1) Orbit character:\n   {orbit_character}  (mu^2=0, field-aligned)")

    return {
        "W0_sq":           W0_sq,
        "rho_0_sim":       rho_0_sim,
        "rho_0_dragt":     rho_0_dragt,
        "P_phi":           P_phi,
        "W0_threshold":    W0_threshold,
        "boundary_status": boundary_status,
        "mu_sq":           mu_sq,
        "stability_threshold": stability_threshold,
        "orbit_character": orbit_character,
    }


def compute_boundary(W0_sq, P_phi, charge_sign=1):
    """
    Computes the accessible boundary curve for the Dragt Poincaré surface of section.
    Uses a dynamic upper rho limit based on P_phi to ensure the closure is always visible.

    The V_eff maximum sits at rho = 2/|P_phi| (where charge_sign*P_phi < 0 for trapped orbits).
    Extending the plot range to 1.2x that value guarantees the inner boundary closure is captured.

    Parameters:
        W0_sq       : Dragt energy constant W0^2
        P_phi       : canonical angular momentum
        charge_sign : +1 for proton, -1 for electron (default +1)

    Returns:
        (rho_bnd, rho_dot_bnd) upper-half boundary arrays ready to plot,
        or (None, None) if no accessible region exists.
    """
    # V_eff maximum sits at rho = -2*charge_sign/P_phi = 2/|P_phi| (positive for trapped orbits)
    # V_eff = 0.5*(P_phi/rho + charge_sign/rho^2)^2  [upward B-field: A_phi = -1/rho]
    #   Proton  (charge_sign=+1): V_eff = 0.5*(P_phi/rho + 1/rho^2)^2
    #   Electron (charge_sign=-1): V_eff = 0.5*(P_phi/rho - 1/rho^2)^2
    trapped = (charge_sign * P_phi < 0)
    rho_upper = (2.0 / abs(P_phi)) * 1.2 if (trapped and abs(P_phi) > 0) else 3.0
    rho_bnd   = np.linspace(0.1, rho_upper, 5000)

    V_eff = 0.5 * (P_phi / rho_bnd + charge_sign / rho_bnd**2)**2
    valid = (W0_sq - 2.0 * V_eff) >= 0

    if not np.any(valid):
        return None, None

    return rho_bnd[valid], np.sqrt(W0_sq - 2.0 * V_eff[valid])


def compute_z_crossings(x_ps, y_ps, z_ps, vx_ps, vy_ps, L_shell):
    """
    Finds equatorial (z=0) crossings in a PS trajectory via linear interpolation.

    Parameters:
        x_ps, y_ps, z_ps : position arrays (R_E, normalized)
        vx_ps, vy_ps     : velocity arrays (v_tau, normalized)
        L_shell          : L-shell for Dragt unit conversion

    Returns:
        (rho_dragt, rho_dot_dragt, x_cross, y_cross, vx_cross, vy_cross)
        where rho_dragt/rho_dot_dragt are in Dragt units and positions/velocities
        are in sim units (needed for downstream gyrophase/mu calculations).
        Returns None if no crossings found.
    """
    mask = z_ps[1:] * z_ps[:-1] < 0
    idx  = np.where(mask)[0]

    if len(idx) == 0:
        return None

    t_frac   = (0.0 - z_ps[idx]) / (z_ps[idx+1] - z_ps[idx])
    x_cross  = x_ps[idx]  + t_frac * (x_ps[idx+1]  - x_ps[idx])
    y_cross  = y_ps[idx]  + t_frac * (y_ps[idx+1]  - y_ps[idx])
    vx_cross = vx_ps[idx] + t_frac * (vx_ps[idx+1] - vx_ps[idx])
    vy_cross = vy_ps[idx] + t_frac * (vy_ps[idx+1] - vy_ps[idx])

    rho_sim       = np.sqrt(x_cross**2 + y_cross**2)
    rho_dot_sim   = (x_cross * vx_cross + y_cross * vy_cross) / rho_sim
    rho_dragt     = rho_sim / L_shell
    rho_dot_dragt = rho_dot_sim * L_shell**2            # v_dragt = v_sim * L^2 (no gamma; see compute_params)

    return rho_dragt, rho_dot_dragt, x_cross, y_cross, vx_cross, vy_cross


def compute_gyrophase_mu(x_cross, y_cross, vx_cross, vy_cross):
    """
    Computes gyrophase and normalized magnetic moment at equatorial crossings.
    At z=0 in a dipole, B is purely in the z-direction with magnitude ~ 1/rho^3.

    Parameters:
        x_cross, y_cross   : crossing positions (R_E, sim units)
        vx_cross, vy_cross : crossing velocities (v_tau, sim units)

    Returns:
        (gyrophase [degrees, -180 to 180], mu_cross [normalized])
    """
    rho_sim = np.sqrt(x_cross**2 + y_cross**2)
    v_rho   = (x_cross * vx_cross + y_cross * vy_cross) / rho_sim
    v_phi   = (x_cross * vy_cross - y_cross * vx_cross) / rho_sim

    gyrophase = np.degrees(np.arctan2(v_rho, v_phi))

    # At z=0, dipole B is purely in +z (upward), magnitude = 1/rho^3
    # Consistent with simulator convention: downward moment, upward equatorial B
    B_z       = 1.0 / rho_sim**3
    mu_cross  = (vx_cross**2 + vy_cross**2) / (2.0 * B_z)

    return gyrophase, mu_cross


def compute_w0_squared(speed, L_shell, mass_si, q_e, M_earth=None):
    """
    Calculates Dragt's dimensionless energy constant W0^2.

    Parameters:
    speed (float): Physical speed of the particle in m/s.
    L_shell (float): The reference L-shell (dimensionless).
    mass_si (float): Particle mass in kg.
    q_e (float): Particle charge in Coulombs.
    M_earth (float): Magnetic dipole moment in T·m³.
                     Default: B_0 * RE³ from constants.py.

    Returns:
    float: The dimensionless energy constant W0^2.
    """
    if M_earth is None:
        M_earth = float(B_0) * float(RE)**3

    # 1. Calculate the reciprocal length scaling factor Gamma
    gamma_df = 1.0 / (L_shell * float(RE))

    # 2. Calculate the characteristic velocity scale
    # Dragt's unit velocity = (|q| * M * Gamma^2) / m
    v_scale = (abs(q_e) * M_earth * (gamma_df**2)) / mass_si

    # 3. Calculate W0^2 = (γv / v_scale)^2
    # From Dragt (1965) eqs 2.7 + 2.15d + 2.16:  γmv = qMΓ²·W₀
    # So W₀ = γv / v_scale, and W₀² = (γv / v_scale)²
    # The γ factor is needed because v_scale = qMΓ²/m uses rest mass,
    # while Dragt's time scaling (eq 2.15d) uses γm.
    gamma_lorentz = 1.0 / np.sqrt(1.0 - (speed / float(spdlight))**2)
    w0_sq = (gamma_lorentz * speed / v_scale)**2

    return w0_sq


# ===================================================================
# === Chunked Dragt Analysis (adiabaticity, meridian, crossings) ====
# ===================================================================
def analysis_chunked(cache_path, L_shell, ps_step, time_factor, chunk_size=1_000_000):
    """
    Stream through an h5 trajectory file in fixed-size chunks and compute:
      - adiabaticity parameter epsilon (decimated for plotting)
      - meridian plane rho/L and z/L (decimated)
      - equatorial z=0 crossings (linear interpolation)
      - epsilon statistics (initial, mean, max)

    Returns a dict with keys:
        eps_arr, t_arr, rho_arr, z_arr   — decimated arrays for plotting
        eps_initial, eps_mean, eps_max   — scalar statistics
        crossings  — tuple (rho_dragt, rho_dot_dragt, x_c, y_c, vx_c, vy_c) or None
    """
    with h5py.File(cache_path, "r") as ps_h5:
        ds = ps_h5["ps"]["y"]
        N_total = ds.shape[1]
        decimate = max(1, N_total // 500_000)  # target ~500K pts for plots
        print(f"  Dragt chunked analysis: {N_total:,} steps, "
              f"chunk={chunk_size:,}, decimate=1/{decimate}")

        eps_initial = None
        eps_sum     = 0.0
        eps_count   = 0
        eps_max     = -np.inf
        eps_dec     = []
        rho_dec     = []
        z_dec       = []
        cx_list     = []
        cy_list     = []
        cvx_list    = []
        cvy_list    = []
        prev_z      = None
        prev_state  = None

        for i0 in range(0, N_total, chunk_size):
            i1 = min(i0 + chunk_size, N_total)
            chunk = ds[:6, i0:i1]
            cx, cy, cz = chunk[0], chunk[1], chunk[2]
            cvx, cvy, cvz = chunk[3], chunk[4], chunk[5]

            # --- adiabaticity ---
            eps = compute_adiabaticity(cx, cy, cz, cvx, cvy, cvz)
            if eps_initial is None:
                eps_initial = float(eps[0])
            eps_sum   += float(np.nansum(eps))
            eps_count += int(np.sum(np.isfinite(eps)))
            cm = float(np.nanmax(eps))
            if cm > eps_max:
                eps_max = cm
            eps_dec.append(eps[::decimate])

            # --- meridian plane (decimated) ---
            rho_dec.append(np.sqrt(cx**2 + cy**2)[::decimate] / L_shell)
            z_dec.append(cz[::decimate] / L_shell)

            # --- z-crossings: chunk boundary ---
            if prev_z is not None and prev_z * cz[0] < 0:
                t_f = abs(prev_z) / (abs(prev_z) + abs(float(cz[0])))
                ps = prev_state
                cx_list.append(np.array([ps[0] + t_f * (float(cx[0]) - ps[0])]))
                cy_list.append(np.array([ps[1] + t_f * (float(cy[0]) - ps[1])]))
                cvx_list.append(np.array([ps[3] + t_f * (float(cvx[0]) - ps[3])]))
                cvy_list.append(np.array([ps[4] + t_f * (float(cvy[0]) - ps[4])]))

            # --- z-crossings: within chunk ---
            mask = cz[:-1] * cz[1:] < 0
            idx  = np.where(mask)[0]
            if len(idx) > 0:
                t_f = np.abs(cz[idx]) / (np.abs(cz[idx]) + np.abs(cz[idx+1]))
                cx_list.append(cx[idx]  + t_f * (cx[idx+1]  - cx[idx]))
                cy_list.append(cy[idx]  + t_f * (cy[idx+1]  - cy[idx]))
                cvx_list.append(cvx[idx] + t_f * (cvx[idx+1] - cvx[idx]))
                cvy_list.append(cvy[idx] + t_f * (cvy[idx+1] - cvy[idx]))

            prev_z     = float(cz[-1])
            prev_state = tuple(float(v) for v in (cx[-1], cy[-1], cz[-1],
                                                    cvx[-1], cvy[-1], cvz[-1]))
            del chunk, cx, cy, cz, cvx, cvy, cvz, eps

    # --- consolidate ---
    eps_arr = np.concatenate(eps_dec)
    rho_arr = np.concatenate(rho_dec)
    z_arr   = np.concatenate(z_dec)
    t_arr   = ps_step * np.arange(len(eps_arr), dtype=float) * decimate * time_factor

    if cx_list:
        cx_all  = np.concatenate(cx_list)
        cy_all  = np.concatenate(cy_list)
        cvx_all = np.concatenate(cvx_list)
        cvy_all = np.concatenate(cvy_list)
        rs      = np.sqrt(cx_all**2 + cy_all**2)
        rd_sim  = (cx_all * cvx_all + cy_all * cvy_all) / rs
        crossings = (rs / L_shell,
                     rd_sim * L_shell**2,
                     cx_all, cy_all, cvx_all, cvy_all)
    else:
        crossings = None

    return dict(
        eps_arr=eps_arr, t_arr=t_arr, rho_arr=rho_arr, z_arr=z_arr,
        eps_initial=eps_initial if eps_initial is not None else 0.0,
        eps_mean=eps_sum / eps_count if eps_count > 0 else 0.0,
        eps_max=eps_max,
        crossings=crossings,
    )


# ------------------------------------------------------------------
#  run_section — full Dragt analysis + plots, returns dragt_log
# ------------------------------------------------------------------
def run_section(
    # Initial conditions (normalised)
    x_initial, y_initial, z_initial,
    vx_initial, vy_initial, v_tau,
    charge_sign, gamma,
    # PS cache info
    USE_PS, cache_path, ps_step, time_factor,
    CACHE_VELOCITY_RTOL,
    # Output
    fig_folder, stem,
    # Plot functions (passed in to avoid circular imports)
    poincare_func,
    gyrophase_mu_func,
    polar_phase_space_func,
    meridian_plane_func,
    adiabaticity_func,
):
    """Run the complete Dragt analysis section.

    Computes L-shell from canonical momentum, runs Dragt parameter
    calculation, boundary analysis, adiabaticity, and generates all
    Dragt-related plots.

    Returns
    -------
    dragt_log : dict
        Populated log dictionary for the summary writer.
    L_shell_dragt : float
        L-shell used for Dragt normalisation.
    """
    dragt_log = {
        "L_eff": None, "W0_sq": None, "boundary": None,
        "mu_sq": None, "orbit_character": None,
        "eps_initial": None, "eps_mean": None, "eps_max": None,
        "hit_atmosphere": False, "hit_atm_r": None,
    }

    # --- L-shell from conserved canonical momentum ---
    _rho_init   = np.sqrt(x_initial**2 + y_initial**2)
    _v_phi_init = (x_initial * vy_initial - y_initial * vx_initial) / _rho_init
    _P_phi_code = _rho_init * _v_phi_init - charge_sign / _rho_init

    if charge_sign * _P_phi_code < 0:
        L_shell_dragt = float(-charge_sign / _P_phi_code)
    else:
        _r_init = np.sqrt(x_initial**2 + y_initial**2 + z_initial**2)
        L_shell_dragt = float(_r_init**3 / _rho_init**2)
        print("  WARNING: P_phi_code indicates open/untrapped orbit, "
              "falling back to field-line L-shell")

    print(f"\n{'='*60}")
    print(f"  Dragt Info")
    print(f"{'='*60}")
    print(f"Dragt L-shell (from conserved canonical momentum): "
          f"{L_shell_dragt:.4f} R_E")

    if not USE_PS:
        print("PS is not enabled. Cannot run Dragt Comparison.")
        return dragt_log, L_shell_dragt

    # --- Initial state from h5 ---
    with h5py.File(cache_path, "r") as _h5_init:
        _y0 = _h5_init["ps"]["y"][:6, 0].astype(float)

    # --- Cache consistency check ---
    _v_mag_ps0     = float(np.sqrt(_y0[3]**2 + _y0[4]**2 + _y0[5]**2))
    _v_tau_expected = float(v_tau)
    _v_rel_err     = abs(_v_mag_ps0 - _v_tau_expected) / _v_tau_expected
    if _v_rel_err > CACHE_VELOCITY_RTOL:
        print(f"\n  *** CACHE MISMATCH WARNING ***")
        print(f"  PS trajectory v_mag = {_v_mag_ps0:.6f}  (from cached run)")
        print(f"  Current v_tau       = {_v_tau_expected:.6f}  (from current KE_particle/mass_si)")
        print(f"  Relative error      = {_v_rel_err*100:.2f}%")
        print(f"  The cached trajectory does not match the current input parameters.")
        print(f"  Re-run the simulation (disable cache loading) to get consistent results.\n")

    # --- Dragt parameters from initial conditions ---
    _x0a  = np.array([_y0[0]]); _y0a  = np.array([_y0[1]]); _z0a  = np.array([_y0[2]])
    _vx0a = np.array([_y0[3]]); _vy0a = np.array([_y0[4]]); _vz0a = np.array([_y0[5]])
    dp = compute_params(
        _x0a, _y0a, _z0a, _vx0a, _vy0a, _vz0a,
        L_shell_dragt, charge_sign=charge_sign)
    W0_sq_calc = dp["W0_sq"]
    P_phi      = dp["P_phi"]
    rho_0_sim  = dp["rho_0_sim"]
    _rho_dot_0_sim = float((_y0[0]*_y0[3] + _y0[1]*_y0[4]) / rho_0_sim)

    # --- Analytical boundary ---
    rho_bnd, rho_dot_bnd = compute_boundary(
        W0_sq_calc, P_phi, charge_sign=charge_sign)

    # --- Chunked Dragt analysis (adiabaticity, meridian, crossings) ---
    _dragt = analysis_chunked(
        cache_path, L_shell_dragt, ps_step, time_factor)

    # --- Plots ---
    poincare_func(
        run_folder=fig_folder, L_shell_dragt=L_shell_dragt, gamma=gamma,
        rho_bnd=rho_bnd, rho_dot_bnd=rho_dot_bnd,
        rho_0_sim=rho_0_sim, rho_dot_0_sim=_rho_dot_0_sim,
        crossings=_dragt["crossings"], stem=stem,
    )

    if _dragt["crossings"] is not None:
        _rho_d, _rho_dot_d, x_c, y_c, vx_c, vy_c = _dragt["crossings"]
        gyrophase, mu_cross = compute_gyrophase_mu(x_c, y_c, vx_c, vy_c)
        gyrophase_mu_func(fig_folder, gyrophase, mu_cross, stem=stem)
        polar_phase_space_func(fig_folder, gyrophase, mu_cross, stem=stem)

    meridian_plane_func(
        fig_folder, _dragt["rho_arr"], _dragt["z_arr"], stem=stem)
    adiabaticity_func(
        fig_folder, _dragt["t_arr"], _dragt["eps_arr"],
        _dragt["eps_initial"], _dragt["eps_mean"], _dragt["eps_max"], stem=stem)

    # --- Populate Dragt log ---
    dragt_log["L_eff"]           = L_shell_dragt
    dragt_log["W0_sq"]           = W0_sq_calc
    dragt_log["boundary"]        = dp["boundary_status"]
    dragt_log["mu_sq"]           = dp["mu_sq"]
    dragt_log["orbit_character"] = dp["orbit_character"]
    dragt_log["eps_initial"]     = _dragt["eps_initial"]
    dragt_log["eps_mean"]        = _dragt["eps_mean"]
    dragt_log["eps_max"]         = _dragt["eps_max"]

    # --- Atmosphere flag from h5 ---
    try:
        with h5py.File(cache_path, "r") as _h5:
            if "ps" in _h5 and "hit_atmosphere" in _h5["ps"].attrs:
                dragt_log["hit_atmosphere"] = bool(_h5["ps"].attrs["hit_atmosphere"])
                dragt_log["hit_atm_r"]     = float(_h5["ps"].attrs["hit_atm_r"])
    except Exception:
        pass

    return dragt_log, L_shell_dragt