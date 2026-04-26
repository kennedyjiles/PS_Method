import numpy as np
import os
import json, hashlib, h5py, time, re
from numba import njit
from ps_method.universal import npfloat, maybe_njit

one = npfloat(1.0)
two = npfloat(2.0)
three = npfloat(3.0)
five = npfloat(5.0)
twopointfive = npfloat(2.5)

@maybe_njit
def lorentz_force_dipole(t, y, qoverm):
    # Unpack position and velocity
    x, y_, z, vx, vy, vz = y
    r2 = x**two + y_**two + z**two
    r5inv = r2**(-twopointfive) if r2 != 0 else 0.0

    # Magnetic field components
    Bx = -three * x * z * r5inv
    By = -three * y_ * z * r5inv
    Bz = -(three * z**two - r2) * r5inv

    # Lorentz force
    ax = qoverm * (vy * Bz - vz * By)
    ay = qoverm * (vz * Bx - vx * Bz)
    az = qoverm * (vx * By - vy * Bx)

    return np.array([vx, vy, vz, ax, ay, az], dtype=npfloat)

@maybe_njit
def PS_dipoleB(PS_order, steps_ps, initial_pos_vel, tol, qoverm, timedelta):
    n_total = 17
    final_coeff_matrix = np.zeros((n_total, steps_ps + 1), dtype=npfloat)

    # For sanity tracking of all variables
    x, y, z, vx, vy, vz = 0, 1, 2, 3, 4, 5
    r2_aux, a_aux, b_aux, c_aux, d_aux, e_aux, f_aux, g_aux = 6, 7, 8, 9, 10, 11, 12, 13
    Bx_aux, By_aux, Bz_aux = 14, 15, 16

    # set up initial dynamic variables
    final_coeff_matrix[0:6, 0] = initial_pos_vel
    x0, y0, z0 = initial_pos_vel[0], initial_pos_vel[1], initial_pos_vel[2]  # need for initilizing aux variables
    vx0, vy0, vz0 = initial_pos_vel[3], initial_pos_vel[4], initial_pos_vel[5]

    # set up initial aux variables
    r2_0 = x0**two + y0**two + z0**two
    a0 = r2_0**(-twopointfive)
    b0 = two * z0**2 - x0**2 - y0**2
    c0 = y0 * z0
    d0 = x0 * z0
    e0 = -(b0 * vy0 - three * c0 * vz0)
    f0 = -(three * d0 * vz0 - b0 * vx0)
    g0 = -(three * c0 * vx0 - three * d0 * vy0)

    final_coeff_matrix[r2_aux, 0] = r2_0
    final_coeff_matrix[a_aux, 0] = a0
    final_coeff_matrix[b_aux, 0] = b0
    final_coeff_matrix[c_aux, 0] = c0
    final_coeff_matrix[d_aux, 0] = d0
    final_coeff_matrix[e_aux, 0] = e0
    final_coeff_matrix[f_aux, 0] = f0
    final_coeff_matrix[g_aux, 0] = g0

    final_coeff_matrix[Bz_aux, 0] = -a0 * b0
    final_coeff_matrix[By_aux, 0] = -npfloat(3.0) * a0 * c0
    final_coeff_matrix[Bx_aux, 0] = -npfloat(3.0) * a0 * d0

    oip1 = one / (one + np.arange(PS_order, dtype=npfloat))
    orders_used = np.zeros(steps_ps + 1, dtype=np.int32)

    # these worked better inline
    def cauchy_sum_inline(a, b, n):
        result = 0.0
        for j in range(n + 1):
            result += a[j] * b[n - j]
        return result  

    def cauchy_divide(a, b, out, n):        #computiing zeta=a/b up to through n and stores it as out[:]
        out[0] = a[0] / b[0]
        for i in range(1, n+1):
            acc = a[i]
            for j in range(1, i + 1):
                acc -= b[j] * out[i - j]
            out[i] = acc / b[0]


    c = np.zeros((n_total, PS_order + 1), dtype=npfloat) 
    sum_terms = np.zeros(n_total, dtype=npfloat)
    zeta = np.zeros(PS_order + 1, dtype=npfloat)

    # initialize base terms outside the loop 
    c[r2_aux, 0] = final_coeff_matrix[x, 0]**two + final_coeff_matrix[y, 0]**two + final_coeff_matrix[z, 0]**two
    c[a_aux, 0] = c[r2_aux, 0]**(-twopointfive)
    zeta[0] = c[a_aux, 0] / c[r2_aux, 0]

    for j in range(1, steps_ps + 1):
        c[:, 0] = final_coeff_matrix[:, j - 1]
        sum_terms[:] = 0

        power = timedelta
        max_contrib = tol + one
        i = 0

        while max_contrib > tol and i < PS_order:
            c[x, i+1]  = c[vx, i] * oip1[i]
            c[y, i+1]  = c[vy, i] * oip1[i]
            c[z, i+1]  = c[vz, i] * oip1[i]
            c[vx, i+1] = qoverm * cauchy_sum_inline(c[a_aux], c[e_aux], i) * oip1[i]
            c[vy, i+1] = qoverm * cauchy_sum_inline(c[a_aux], c[f_aux], i) * oip1[i]
            c[vz, i+1] = qoverm * cauchy_sum_inline(c[a_aux], c[g_aux], i) * oip1[i]

            c[r2_aux, i+1] = cauchy_sum_inline(c[x], c[x], i+1) + cauchy_sum_inline(c[y], c[y], i+1) + cauchy_sum_inline(c[z], c[z], i+1)
            cauchy_divide(c[a_aux], c[r2_aux], zeta, i+1)      #This is modifying zeta in place 
            a_prime = 0.0
            for k in range(i+1):
                a_prime += (i - k + 1) * zeta[k] * c[r2_aux, i - k + 1]
            c[a_aux, i+1] = - (five / (two * (i + 1))) * a_prime

            c[b_aux, i+1] = two * cauchy_sum_inline(c[z], c[z], i+1) - cauchy_sum_inline(c[x], c[x], i+1) - cauchy_sum_inline(c[y], c[y], i+1)
            c[c_aux, i+1] = cauchy_sum_inline(c[y], c[z], i+1)
            c[d_aux, i+1] = cauchy_sum_inline(c[x], c[z], i+1)

            c[e_aux, i+1] = -(cauchy_sum_inline(c[b_aux], c[vy], i+1) - three * cauchy_sum_inline(c[c_aux], c[vz], i+1))
            c[f_aux, i+1] = -(three * cauchy_sum_inline(c[d_aux], c[vz], i+1) - cauchy_sum_inline(c[b_aux], c[vx], i+1))
            c[g_aux, i+1] = -(three * (cauchy_sum_inline(c[c_aux], c[vx], i+1) - cauchy_sum_inline(c[d_aux], c[vy], i+1)))

            c[Bx_aux, i+1] = -npfloat(3.0) * cauchy_sum_inline(c[a_aux], c[d_aux], i+1)
            c[By_aux, i+1] = -npfloat(3.0) * cauchy_sum_inline(c[a_aux], c[c_aux], i+1)
            c[Bz_aux, i+1] =        -cauchy_sum_inline(c[a_aux], c[b_aux], i+1)

            sum_terms += c[:, i+1] * power
            max_contrib = np.abs(c[:, i+1] * power).max()
            power *= timedelta
            i += 1

        final_coeff_matrix[:, j] = final_coeff_matrix[:, j - 1] + sum_terms

        x_now, y_now, z_now = final_coeff_matrix[x, j], final_coeff_matrix[y, j], final_coeff_matrix[z, j]
        vx_now, vy_now, vz_now = final_coeff_matrix[vx, j], final_coeff_matrix[vy, j], final_coeff_matrix[vz, j]

        # tethering variables
        r2_now = x_now**two + y_now**two + z_now**two
        a_now = r2_now**(-twopointfive)
        b_now = two * z_now**two - x_now**two - y_now**two
        c_now = y_now * z_now
        d_now = x_now * z_now

        e_now = (b_now * vy_now - three * c_now * vz_now)
        f_now = (three * d_now * vz_now - b_now * vx_now)
        g_now = (three * c_now * vx_now - three * d_now * vy_now)

        final_coeff_matrix[r2_aux, j] = r2_now
        final_coeff_matrix[a_aux, j] = a_now
        final_coeff_matrix[b_aux, j] = b_now
        final_coeff_matrix[c_aux, j] = c_now
        final_coeff_matrix[d_aux, j] = d_now
        final_coeff_matrix[e_aux, j] = -e_now
        final_coeff_matrix[f_aux, j] = -f_now
        final_coeff_matrix[g_aux, j] = -g_now
        final_coeff_matrix[Bx_aux, j] = -npfloat(3.0) * a_now * d_now
        final_coeff_matrix[By_aux, j] = -npfloat(3.0) * a_now * c_now
        final_coeff_matrix[Bz_aux, j] = -a_now * b_now

        orders_used[j] = i

    return final_coeff_matrix, orders_used

@maybe_njit
def compute_mu_ps(solution_ps, mass):
    x, y, z = solution_ps[0], solution_ps[1], solution_ps[2]
    vx, vy, vz = solution_ps[3], solution_ps[4], solution_ps[5]
    Bx, By, Bz = solution_ps[14], solution_ps[15], solution_ps[16]  

    mu = np.zeros_like(x)
    for i in range(len(x)):
        B = np.array([Bx[i], By[i], Bz[i]])
        B2 = np.dot(B, B)
        if B2 == 0:
            mu[i] = 0.0
            continue
        v = np.array([vx[i], vy[i], vz[i]])
        v_par = (np.dot(v, B) / B2) * B
        v_perp = v - v_par
        mu[i] = mass * np.dot(v_perp, v_perp) / (2 * np.sqrt(B2))
    return mu

@maybe_njit
def compute_mu_rk(solution_rk, mass):
    mu = np.zeros(len(solution_rk))
    for i in range(len(solution_rk)):
        x, y, z = solution_rk[i, 0:3]
        vx, vy, vz = solution_rk[i, 3:6]

        # Compute B at position
        r2 = x**2 + y**2 + z**2
        if r2 == 0:
            mu[i] = 0.0
            continue
        r5inv = r2**(-2.5)
        # Sign convention matches simulator (downward dipole moment, upward B at equator)
        B = np.array([
            -3 * x * z * r5inv,
            -3 * y * z * r5inv,
            -(3 * z**2 - r2) * r5inv
        ])

        B2 = np.dot(B, B)
        v = np.array([vx, vy, vz])
        v_par = (np.dot(v, B) / B2) * B
        v_perp = v - v_par
        mu[i] = mass * np.dot(v_perp, v_perp) / (2 * np.sqrt(B2))

    return mu


# ========================
# ==== RKG Functions ====
# ========================

@maybe_njit
def vector_potential_dipole(r):
    x, y, z = r
    r2 = x**2 + y**2 + z**2
    r3 = r2 * np.sqrt(r2)

    if r3 == 0:
        return np.zeros(3)

    Ax = y / r3
    Ay = - x / r3
    Az = 0.0

    return np.array([Ax, Ay, Az], dtype=npfloat)

@maybe_njit
def hamiltonian_rhs(t, y, qoverm):
    x, y_, z = y[0], y[1], y[2]
    px, py, pz = y[3], y[4], y[5]

    r2 = x*x + y_*y_ + z*z
    r = np.sqrt(r2)
    r3 = r2 * r
    r5 = r2 * r3

    if r5 == 0:
        return np.zeros(6, dtype=npfloat)

    # Vector potential
    Ax = y_ / r3
    Ay = -x / r3
    Az = 0.0

    # Mechanical momentum
    Pix = px - qoverm * Ax
    Piy = py - qoverm * Ay
    Piz = pz

    # dq/dt
    dxdt = Pix
    dydt = Piy
    dzdt = Piz

    # dp/dt (hardcoded)
    dpxdt = qoverm * (
        -3 * x * y_ / r5 * Pix
        - (1.0 / r3 - 3 * x * x / r5) * Piy
    )

    dpydt = qoverm * (
        (1.0 / r3 - 3 * y_ * y_ / r5) * Pix
        + 3 * x * y_ / r5 * Piy
    )

    dpzdt = qoverm * 3 * z / r5 * (-y_ * Pix + x * Piy)

    return np.array([dxdt, dydt, dzdt, dpxdt, dpydt, dpzdt], dtype=npfloat)


# OLD rkgl4_hamiltonian_step (kept for reproducibility):
# @maybe_njit
# def rkgl4_hamiltonian_step(func, y0, dt, args=(), max_iter=10, tol=1e-12, eps=1e-13):
#     sqrt3 = np.sqrt(3.0)
#     a11, a12 = 0.25, 0.25 - sqrt3 / 6.0
#     a21, a22 = 0.25 + sqrt3 / 6.0, 0.25
#     b1 = b2 = 0.5
#     dim = len(y0)
#     K = np.zeros((2, dim), dtype=npfloat)
#     K[0] = func(0.0, y0, *args)
#     K[1] = K[0].copy()
#     for n in range(max_iter):
#         Y1 = y0 + dt * (a11 * K[0] + a12 * K[1])
#         Y2 = y0 + dt * (a21 * K[0] + a22 * K[1])
#         F1 = K[0] - func(0.0, Y1, *args)
#         F2 = K[1] - func(0.0, Y2, *args)
#         F = np.concatenate((F1, F2))
#         normF = np.max(np.abs(F))
#         if normF < tol:
#             break
#         J = np.zeros((2 * dim, 2 * dim), dtype=npfloat)
#         for i in range(2):
#             for j in range(dim):
#                 dK = np.zeros((2, dim), dtype=npfloat)
#                 dK[i, j] = eps
#                 Y1_pert = y0 + dt * (a11 * (K[0] + dK[0]) + a12 * (K[1] + dK[1]))
#                 Y2_pert = y0 + dt * (a21 * (K[0] + dK[0]) + a22 * (K[1] + dK[1]))
#                 F1_pert = K[0] + dK[0] - func(0.0, Y1_pert, *args)
#                 F2_pert = K[1] + dK[1] - func(0.0, Y2_pert, *args)
#                 F_pert = np.concatenate((F1_pert, F2_pert))
#                 dF = (F_pert - F) / eps
#                 J[:, i * dim + j] = dF
#         try:
#             dK_flat = np.linalg.solve(J, -F)
#         except:
#             raise RuntimeError("Newton step failed: singular Jacobian")
#         K_flat = np.concatenate((K[0], K[1])) + dK_flat
#         K[0] = K_flat[:dim]
#         K[1] = K_flat[dim:]
#     else:
#         print("Newton did not converge")
#     return y0 + dt * (b1 * K[0] + b2 * K[1])

@maybe_njit
def rkgl4_hamiltonian_step(func, y0, dt, args=(), max_iter=10, tol=1e-12, eps=1e-13):
    # Gauss-Legendre RK4 (implicit, symplectic) — njit-friendly version.
    # Changes from old version: removed try/except (numba can't compile it,
    # forces object mode), removed np.concatenate in hot loop, pre-allocated
    # scratch arrays. Produces identical results.
    sqrt3 = np.sqrt(3.0)
    a11, a12 = 0.25, 0.25 - sqrt3 / 6.0
    a21, a22 = 0.25 + sqrt3 / 6.0, 0.25
    b1 = b2 = 0.5

    dim = len(y0)
    K = np.zeros((2, dim), dtype=npfloat)

    # Pre-allocate scratch arrays (avoids per-iteration allocation)
    F = np.zeros(2 * dim, dtype=npfloat)
    J = np.zeros((2 * dim, 2 * dim), dtype=npfloat)
    K_save = np.zeros((2, dim), dtype=npfloat)
    Y1 = np.zeros(dim, dtype=npfloat)
    Y2 = np.zeros(dim, dtype=npfloat)

    # Initial guess from explicit Euler
    K[0] = func(0.0, y0, *args)
    K[1] = K[0].copy()

    for n in range(max_iter):
        # Stage values
        for d in range(dim):
            Y1[d] = y0[d] + dt * (a11 * K[0, d] + a12 * K[1, d])
            Y2[d] = y0[d] + dt * (a21 * K[0, d] + a22 * K[1, d])

        F1 = K[0] - func(0.0, Y1, *args)
        F2 = K[1] - func(0.0, Y2, *args)
        F[:dim] = F1
        F[dim:] = F2

        # Convergence check
        normF = np.max(np.abs(F))
        if normF < tol:
            break

        # Build Jacobian by finite differences
        for d in range(dim):
            J[:, :] = 0.0
        for i in range(2):
            for j in range(dim):
                # Save, perturb, evaluate, restore
                K_save[i, j] = K[i, j]
                K[i, j] += eps

                for d in range(dim):
                    Y1[d] = y0[d] + dt * (a11 * K[0, d] + a12 * K[1, d])
                    Y2[d] = y0[d] + dt * (a21 * K[0, d] + a22 * K[1, d])

                F1_pert = K[0] - func(0.0, Y1, *args)
                F2_pert = K[1] - func(0.0, Y2, *args)

                for d in range(dim):
                    J[d,       i * dim + j] = (F1_pert[d] - F1[d]) / eps
                    J[dim + d, i * dim + j] = (F2_pert[d] - F2[d]) / eps

                K[i, j] = K_save[i, j]  # restore

        # Newton update (no try/except — let it crash if singular,
        # which would indicate a bug, not a recoverable condition)
        dK_flat = np.linalg.solve(J, -F)
        for d in range(dim):
            K[0, d] += dK_flat[d]
            K[1, d] += dK_flat[dim + d]

    result = np.zeros(dim, dtype=npfloat)
    for d in range(dim):
        result[d] = y0[d] + dt * (b1 * K[0, d] + b2 * K[1, d])
    return result

@maybe_njit
def rkgl4_hamiltonian(func, y0, dt, steps, args=()):
    d_out = np.zeros((steps + 1, len(y0)), dtype=npfloat)
    d_out[0] = y0

    for i in range(1, steps + 1):
        d_out[i] = rkgl4_hamiltonian_step(
            func, d_out[i - 1], dt, args
        )

    return d_out

# ========================
# === Mirror Functions ===
# ========================
named_indices = {"vx":3,"vy":4,"vz":5,"Bx":14,"By":15,"Bz":16}

# Compact h5 storage: only these rows are saved (pos, vel, B-field)
SAVE_ROWS = [0, 1, 2, 3, 4, 5, 14, 15, 16] #note that h5 is no longer 17 rows but for legacy purposes we keep the same indexing for the variables, just not saving the unused rows
n_save = len(SAVE_ROWS)

def expand_h5_to_full(compact_arr):
    """Expand a 9-row compact h5 array back to 17-row full layout.
    If the array already has 17 rows, return it unchanged. Protects legacy"""
    if compact_arr.shape[0] == 17:
        return compact_arr
    full = np.zeros((17, compact_arr.shape[1]), dtype=compact_arr.dtype)
    for i_new, i_old in enumerate(SAVE_ROWS):
        full[i_old, :] = compact_arr[i_new, :]
    return full

"""
Identify mirror (bounce) times by detecting zero crossings of s = v·B,
which is proportional to the parallel velocity v_parallel. A sign change
in v·B indicates reversal of motion along the magnetic field line and thus
a mirror point. Small-|s| values are excluded to suppress numerical jitter,
and a minimum index separation is enforced to avoid multiple detections
near a single mirror. Optional linear interpolation provides sub-step
estimates of the mirror times.
s_eps: magnitude threshold to ignore tiny jitters (units of v·B).
"""

def mirror_times_from_PS(final_coeff_matrix, dt, idx_map=None, interp=True, min_gap=15,
                         s_eps=1e-18):

    idx = named_indices if idx_map is None else idx_map
    vx = final_coeff_matrix[idx["vx"], :]
    vy = final_coeff_matrix[idx["vy"], :]
    vz = final_coeff_matrix[idx["vz"], :]
    Bx = final_coeff_matrix[idx["Bx"], :]
    By = final_coeff_matrix[idx["By"], :]
    Bz = final_coeff_matrix[idx["Bz"], :]

    s = vx*Bx + vy*By + vz*Bz  # proxy for v_parallel*|B|
    crossings_idx, crossings_tau = [], []
    last_i = -10**9

    for i in range(1, s.size):
        s0, s1 = s[i-1], s[i]

        # require both samples to be "significant" to avoid micro-jitter
        if abs(s0) < s_eps or abs(s1) < s_eps:
            continue

        if s0 * s1 < 0.0 and (i - last_i) >= min_gap:
            if interp:
                denom = (s1 - s0)
                if denom == 0.0:
                    tc = i * dt
                else:
                    t0, t1 = (i-1)*dt, i*dt
                    tc = t0 + (t1 - t0) * (-s0) / denom
            else:
                tc = i * dt
            crossings_idx.append(i)
            crossings_tau.append(tc)
            last_i = i

    return np.asarray(crossings_idx, dtype=int), np.asarray(crossings_tau, dtype=float)

def init_bounce_stream_state():
    return {
        "last_s": None,
        "last_t": None,
        "last_y": None,
        "last_cross_t": -np.inf,
        "crossing_times": [],
    }

def init_drift_stream_state():
    return {
        "t_samples": [],
        "phi_samples": [],
        "last_phi": None,
    }

def record_drift_sample_at_time(y_prev, y_curr, t_prev, t_curr, tc, state):
    """
    Linearly interpolate (x,y) at mirror time tc,
    compute phi, unwrap incrementally, and store.
    """
    # interpolation weight
    if t_curr == t_prev:
        w = 0.0
    else:
        w = (tc - t_prev) / (t_curr - t_prev)

    x = (1 - w) * y_prev[0] + w * y_curr[0]
    y = (1 - w) * y_prev[1] + w * y_curr[1]

    phi = np.arctan2(y, x)

    # incremental unwrap
    if state["last_phi"] is not None:
        dphi = phi - state["last_phi"]
        phi -= 2.0 * np.pi * np.round(dphi / (2.0 * np.pi))

    state["last_phi"] = phi
    state["t_samples"].append(tc)
    state["phi_samples"].append(phi)


def process_bounce_and_drift_chunk(
    y_chunk,
    t_chunk,
    bounce_state,
    drift_state,
    min_gap_tau,
    s_eps,
    idx_map=None,
    interp=True,
):
    idx = named_indices if idx_map is None else idx_map

    vx = y_chunk[idx["vx"], :]
    vy = y_chunk[idx["vy"], :]
    vz = y_chunk[idx["vz"], :]
    Bx = y_chunk[idx["Bx"], :]
    By = y_chunk[idx["By"], :]
    Bz = y_chunk[idx["Bz"], :]

    s = vx*Bx + vy*By + vz*Bz

    for i in range(len(s)):
        si = s[i]
        ti = t_chunk[i]

        if bounce_state["last_s"] is not None:
            s0, s1 = bounce_state["last_s"], si

            if abs(s0) >= s_eps and abs(s1) >= s_eps:
                if s0 * s1 < 0.0 and (ti - bounce_state["last_cross_t"]) >= min_gap_tau:

                    # --- interpolate mirror time ---
                    if interp:
                        denom = (s1 - s0)
                        if denom == 0.0:
                            tc = ti
                        else:
                            tc = bounce_state["last_t"] + (ti - bounce_state["last_t"]) * (-s0) / denom
                    else:
                        tc = ti

                    bounce_state["crossing_times"].append(tc)
                    bounce_state["last_cross_t"] = tc

                    # --- NEW: drift sample at mirror ---
                    record_drift_sample_at_time_npunwrap_semantics(
                        y_prev=bounce_state["last_y"],
                        y_curr=y_chunk[:, i],
                        t_prev=bounce_state["last_t"],
                        t_curr=ti,
                        tc=tc,
                        state=drift_state,
                    )

        bounce_state["last_s"] = si
        bounce_state["last_t"] = ti
        bounce_state["last_y"] = y_chunk[:, i]


def finalize_bounce_stream(state, time_scale_sec=None):
    return bounce_summary(state["crossing_times"], time_scale_sec=time_scale_sec)

def finalize_drift_stream(
    drift_state,
    time_scale_sec=None,
    min_phase_rad=1.0,
):
    t = np.asarray(drift_state["t_samples"], dtype=float)
    phi = np.asarray(drift_state["phi_samples"], dtype=float)

    if t.size < 2:
        return {
            "period_tau_mean": None,
            "period_tau_fit": None,
            "period_s_mean": None,
            "period_s_fit": None,
            "direction": +1,
        }

    # slope-based estimate
    a, b = np.polyfit(t, phi, 1)
    dphi_span = phi.max() - phi.min()

    period_tau_fit = None
    if a != 0.0 and dphi_span >= min_phase_rad:
        period_tau_fit = (2.0 * np.pi) / abs(a)

    # direction
    direction = +1 if (phi[-1] - phi[0]) >= 0 else -1

    result = {
        "period_tau_fit": period_tau_fit,
        "period_s_fit": (period_tau_fit * time_scale_sec)
                          if (period_tau_fit is not None and time_scale_sec is not None)
                          else None,
        "direction": direction,
    }
    return result


def bounce_summary(crossing_times_tau, time_scale_sec=None):
    import numpy as np
    c = np.asarray(crossing_times_tau, dtype=float)
    half_tau = np.diff(c) if c.size >= 2 else np.array([], float)
    full_tau = (c[2:] - c[:-2]) if c.size >= 3 else np.array([], float)

    out = {
        "n_crossings": int(c.size),
        "half_tau": half_tau,
        "half_mean_tau": float(np.mean(half_tau)) if half_tau.size else None,
        "full_tau": full_tau,
        "full_mean_tau": float(np.mean(full_tau)) if full_tau.size else None,
        "bounce_frequency_per_tau": (1.0/float(np.mean(full_tau))) if full_tau.size else None,
    }

    if time_scale_sec is not None:
        half_s = half_tau * time_scale_sec
        full_s = full_tau * time_scale_sec
        out.update({
            "half_s": half_s,
            "half_mean_s": float(np.mean(half_s)) if half_s.size else None,
            "full_s": full_s,
            "full_mean_s": float(np.mean(full_s)) if full_s.size else None,
            "bounce_frequency_hz": (1.0/float(np.mean(full_s))) if full_s.size else None,
        })
    return out


# ========================
# === Drift Functions ===
# ========================

def record_drift_sample_at_time_npunwrap_semantics(
    y_prev, y_curr,
    t_prev, t_curr, tc,
    state,
):
    # interpolation weight
    w = 0.0 if t_curr == t_prev else (tc - t_prev) / (t_curr - t_prev)

    # raw phi at endpoints
    phi0 = np.arctan2(y_prev[1], y_prev[0])
    phi1 = np.arctan2(y_curr[1], y_curr[0])

    # --- LOCAL unwrap: allow multi-2π jumps ---
    d = phi1 - phi0
    if abs(d) > np.pi:
        phi1 -= 2.0 * np.pi * np.round(d / (2.0 * np.pi))

    # interpolate phi (NOT x,y)
    phi_tc = (1.0 - w) * phi0 + w * phi1

    # --- GLOBAL unwrap: allow multi-2π jumps relative to last stored sample ---
    if state.get("last_phi", None) is not None:
        d2 = phi_tc - state["last_phi"]
        if abs(d2) > np.pi:
            phi_tc -= 2.0 * np.pi * np.round(d2 / (2.0 * np.pi))

    state["last_phi"] = phi_tc
    state["t_samples"].append(tc)
    state["phi_samples"].append(phi_tc)



def _unwrap_phi_from_PS(final_coeff_matrix):
    """
    Build unwrapped cylindrical azimuth phi = atan2(y,x) from PS output.
    Returns phi_unwrapped (radians), shape (N,)
    """
    x = final_coeff_matrix[0, :]
    y = final_coeff_matrix[1, :]
    phi = np.arctan2(y, x)               # [-pi, pi]
    phi_unwrapped = np.unwrap(phi)       # continuous, removes 2π jumps
    return phi_unwrapped

def _pick_samples(t_tau, phi_unwrapped, sample='raw', mirror_times_tau=None):
    """
    Choose which samples to use for drift estimation.
    - 'raw'      : use all points (most jitter from gyro)
    - 'mirrors'  : resample phi(t) at mirror times only (needs mirror_times_tau)
    """
    if sample == 'raw' or mirror_times_tau is None or len(mirror_times_tau) == 0:
        return t_tau, phi_unwrapped

    # Interpolate phi at mirror times to suppress gyro-scale noise
    t_all = t_tau
    phi_all = phi_unwrapped
    # t_m = np.asarray(mirror_times_tau, dtype=float)

    t_all = np.asarray(t_all, dtype=np.float64)
    phi_all = np.asarray(phi_all, dtype=np.float64)
    t_m = np.asarray(mirror_times_tau, dtype=np.float64)
    # guard: require t_all increasing and within bounds
    t_m = t_m[(t_m >= t_all[0]) & (t_m <= t_all[-1])]
    if t_m.size == 0:
        return t_all, phi_all
    phi_m = np.interp(t_m, t_all, phi_all)
    return t_m, phi_m

def drift_period_from_PS(final_coeff_matrix, dt_tau,
                         mirror_times_tau=None,
                         sample='mirrors',
                         return_details=False,
                         time_scale_sec=None,
                         min_phase_rad=1.0):

    # Build time array in τ
    N = final_coeff_matrix.shape[1]
    t_tau = dt_tau * np.arange(N, dtype=float)

    phi_unwrapped = _unwrap_phi_from_PS(final_coeff_matrix)

    # Choose sampling (raw or at mirrors)
    t_used, phi_used = _pick_samples(t_tau, phi_unwrapped, sample=sample,
                                     mirror_times_tau=mirror_times_tau)

    # ---- Robust slope fit (gated by min unwrapped phase) ----
    period_tau_fit = None
    if t_used.size >= 2:
        a, b = np.polyfit(t_used.astype(np.float64), phi_used.astype(np.float64), 1) # linear fit: phi ≈ a t + b, cast as flot 64, does not like 128 otherwise 
        dphi_span = float(np.max(phi_used) - np.min(phi_used))
        if a != 0.0 and dphi_span >= float(min_phase_rad):
            period_tau_fit = (2.0 * np.pi) / abs(a)
    else:
        a = 0.0

    # ---- Crossing-based period without assuming monotonic dphi ----
    drift_turn_times = []
    if t_used.size >= 2:
        phi0 = phi_used[0]
        dphi = phi_used - phi0
        # Determine net direction from overall advance (more robust than slope)
        net = dphi[-1] - dphi[0]
        direction = +1 if net >= 0 else -1
        step = 2.0 * np.pi * direction

        dphi_min, dphi_max = float(dphi.min()), float(dphi.max())
        if direction > 0:
            levels = np.arange(2.0*np.pi, dphi_max + 1e-12, 2.0*np.pi)
        else:
            levels = np.arange(-2.0*np.pi, dphi_min - 1e-12, -2.0*np.pi)

        # Scan each level and find the FIRST sign-change interval after the previous crossing
        last_i = 0
        for L in levels:
            # a[k] = dphi[k] - L
            a0 = dphi[:-1] - L
            a1 = dphi[1:]  - L
            # look for sign change or exact hit
            candidates = np.where(a0[last_i:] * a1[last_i:] <= 0)[0]
            if candidates.size == 0:
                continue
            i = int(candidates[0] + last_i)
            # linear interpolation within [i, i+1]
            t0, t1 = t_used[i], t_used[i+1]
            y0, y1 = a0[i], a1[i]
            denom = (y1 - y0)
            tc = t1 if denom == 0 else t0 + (t1 - t0) * (-y0) / denom
            drift_turn_times.append(tc)
            last_i = i + 1  # move forward so next level finds the next crossing

    drift_turn_times = np.asarray(drift_turn_times, dtype=float)
    drift_intervals_tau = np.diff(drift_turn_times) if drift_turn_times.size >= 2 else np.array([], float)
    period_tau_mean = float(np.mean(drift_intervals_tau)) if drift_intervals_tau.size else None

    # If we never set direction above (e.g., too few samples), fall back to slope sign or +1
    if t_used.size >= 2:
        net_adv = (phi_used[-1] - phi_used[0])
        direction_out = +1 if net_adv >= 0 else -1
    else:
        direction_out = +1

    result = {
        "period_tau_mean": period_tau_mean,
        "period_tau_fit": period_tau_fit,
        "period_s_mean": (period_tau_mean * time_scale_sec) if (time_scale_sec is not None and period_tau_mean is not None) else None,
        "period_s_fit":  (period_tau_fit  * time_scale_sec) if (time_scale_sec is not None and period_tau_fit  is not None) else None,
        "direction": int(direction_out),
    }

    if return_details:
        result.update({
            "t_tau": t_tau,
            "phi_unwrapped": phi_unwrapped,
            "t_used_tau": t_used,
            "phi_used": phi_used,
            "drift_turn_times_tau": drift_turn_times,
            "drift_intervals_tau": drift_intervals_tau,
        })
    return result

# ========================
# === Write Functions ===
# ========================

def _to_serializable(x):
    if isinstance(x, (np.floating, np.float32, np.float64)):
        return float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.ndarray,)):
        return x.tolist()
    return x

def get_run_params(USE_RK45, USE_RK4, USE_RKG, USE_PS, decimate, PS_CHUNKING,
                   mass_si, q_e, B_0, gamma, user_min_phase,
                   x_initial, y_initial, z_initial,
                   pitch_deg, phi_deg,
                   norm_time, ps_step, rk4_step, rkg_step,
                   PS_order, tol, qoverm, rtol_rk45, atol_rk45):

    """Collect all (hopefully) knobs that define a unique run."""
    return {
        # toggles
        "USE_RK45": bool(USE_RK45),
        "USE_RK4":  bool(USE_RK4),
        "USE_RKG":  bool(USE_RKG),
        "USE_PS":  bool(USE_PS),
        "PS_CHUNKING":  bool(PS_CHUNKING),


        # physics & normalization
        "decimate": _to_serializable(decimate),
        "mass_si": _to_serializable(mass_si),
        "q_e": _to_serializable(q_e),
        "B_0": _to_serializable(B_0),
        "gamma": _to_serializable(gamma),
        "user_min_phase": _to_serializable(user_min_phase),

        # initial conditions (positions in RE units and velocity setup)
        "x_initial": _to_serializable(x_initial),
        "y_initial": _to_serializable(y_initial),
        "z_initial": _to_serializable(z_initial),
        "pitch_deg": _to_serializable(pitch_deg),
        "phi_deg": _to_serializable(phi_deg),

        # times / steps
        "norm_time": _to_serializable(norm_time),
        "ps_step": _to_serializable(ps_step),
        "rk4_step": _to_serializable(rk4_step),
        "rkg_step": _to_serializable(rkg_step),

        # PS & solver knobs
        "PS_order": int(PS_order),
        "tol": _to_serializable(tol),
        "rtol_rk45": _to_serializable(rtol_rk45),
        "atol_rk45": _to_serializable(atol_rk45),
  
        # charge/mass normalization used in RHS
        "qoverm": _to_serializable(qoverm),
    }

def run_hash(params: dict) -> str:
    j = json.dumps(params, sort_keys=True, default=_to_serializable, separators=(",",":"))
    return hashlib.sha1(j.encode("utf-8")).hexdigest()[:16]

def h5_path_for(params, output_folder):
    return os.path.join(output_folder, f"run_{run_hash(params)}.h5")

def save_results_h5(h5_path, results, summary):
    with h5py.File(h5_path, "w") as f:

        # # --- params ---
        # f.attrs["params_json"] = json.dumps(
        #     params, sort_keys=True, default=_to_serializable
        # )

        f.attrs["summary_json"] = json.dumps(summary)

        # --- solver groups ---
        for k in ("ps", "rk4", "rk45", "rkg"):
            if k not in results or results[k] is None:
                continue

            grp = f.create_group(k)

            for name, val in results[k].items():
                if val is None:
                    continue

                if isinstance(val, np.ndarray):
                    grp.create_dataset(
                        name,
                        data=val,
                        compression="gzip",
                        compression_opts=2
                    )
                else:
                    grp.attrs[name] = val

        # --- meta ---
        meta = results.get("meta", {})
        gmeta = f.create_group("meta")

        for mk, mv in meta.get("timing", {}).items():
            gmeta.attrs[f"timing_{mk}"] = float(mv)

        for sk in ("physical_time", "norm_time", "percent_c", "particle_label"):
            if sk in meta:
                gmeta.attrs[sk] = meta[sk]


def load_results_h5(h5_path):
    with h5py.File(h5_path, "r") as f:
        loaded = {"meta": {"timing": {}}}
        # loaded["params"] = json.loads(f.attrs["params_json"])

        if "params_json" in f.attrs:
            loaded["params"] = json.loads(f.attrs["params_json"])
        else:
            loaded["params"] = None   # streaming-only file


        def _read_grp(name):
            if name not in f:
                return None

            g = f[name]
            out = {}

            # datasets
            for ds in g:
                out[ds] = g[ds][...]

            # attributes (dt, steps, t0, etc.)
            for k, v in g.attrs.items():
                out[k] = v

            return out
        for k in ("ps","rk4","rk45","rkg"):
            loaded[k] = _read_grp(k)

        gmeta = f["meta"]
        for a in gmeta.attrs:
            if a.startswith("timing_"):
                loaded["meta"]["timing"][a.replace("timing_","")] = gmeta.attrs[a]
            else:
                loaded["meta"][a] = gmeta.attrs[a]

        return loaded

def append_results_h5(h5_path, results, summary):
    """
    Append non-PS solver results and metadata to an existing HDF5 file.
    Ensures dictionary is written exactly once (for streaming PS files).
    """

    with h5py.File(h5_path, "a") as f:

        # Root-level metadata (FINALIZE STREAMED FILE)
        if "summary_json" not in f.attrs:
            f.attrs["summary_json"] = json.dumps(summary)

        # Meta group
        if "meta" not in f:
            gmeta = f.create_group("meta")
        else:
            gmeta = f["meta"]

        # Timing
        for mk, mv in results["meta"]["timing"].items():
            gmeta.attrs[f"timing_{mk}"] = float(mv)

        # Other meta
        for sk in ("physical_time", "norm_time", "percent_c", "particle_label"):
            if sk in results["meta"]:
                gmeta.attrs[sk] = results["meta"][sk]

        # -------------------------------------------------
        # RK solvers
        # -------------------------------------------------
        for k in ("rk4", "rk45", "rkg"):
            if results.get(k) is None:
                continue

            if k in f:
                del f[k]

            grp = f.create_group(k)
            for name, val in results[k].items():
                if isinstance(val, np.ndarray):
                    grp.create_dataset(
                        name,
                        data=val,
                        compression="gzip",
                        compression_opts=2,
                    )
                else:
                    grp.attrs[name] = val


def summarize_error(label, err, f):
    mean_val = np.mean(err)
    max_val  = np.max(np.abs(err))
    rms_val  = np.sqrt(np.mean(err**2))
    f.write(
        f"  {label:<8}: "
        f"mean = {mean_val:.2e}, "
        f"max = {max_val:.2e}, "
        f"rms = {rms_val:.2e}\n"
    )

def summarize(err):
    return {
        "mean": np.mean(err),
        "max":  np.max(np.abs(err)),
        "rms":  np.sqrt(np.mean(err**2))
    }

def load_legacy_file(h5_path):
    """
    loader for legacy HDF5 files
    """

    f = h5py.File(h5_path, "r") 

    # -------------------------------------------------
    # Params (legacy)
    # -------------------------------------------------
    params = {}
    if "params_json" in f.attrs:
        params = json.loads(f.attrs["params_json"])

    # -------------------------------------------------
    # Meta
    # -------------------------------------------------
    meta = f["meta"]
    timing = {
        k.replace("timing_", ""): float(v)
        for k, v in meta.attrs.items()
        if k.startswith("timing_")
    }

    particle_label = meta.attrs.get("particle_label", "")
    label_l = particle_label.lower()

    # Reconstruct Particle type
    if "proton" in label_l:
        particle = "Proton"
    elif "electron" in label_l:
        particle = "Electron"
    else:
        particle = "Unknown"

    # Reconstruct Energy (eV)
    m = re.search(r"([0-9.+\-eE]+)\s*ev", label_l)
    if m:
        KE_particle = float(m.group(1))
    else:
        raise RuntimeError(f"Could not parse energy from particle_label: '{particle_label}'")
    
    # Reconstruct gyroperiods from legacy files
    x_initial = params["x_initial"]
    ps_step = params.get("ps_step",0)
    rk4_step = params.get("rk4_step", 0)
    rkg_step = params.get("rkg_step", 0)
    norm_time = params["norm_time"]
    T_gyro = 2.0 * np.pi * (x_initial**3)  
    gyroperiods= norm_time / T_gyro
    npfloat= np.float64 

    summary = {
        "meta": {
            "stem": h5_path.split("/")[-1].replace(".h5", ""),
            "legacy": True,
            "particle": particle,
            "mass_si": params["mass_si"],
            "q_e": params["q_e"],
            "energy_eV": npfloat(KE_particle),   
            "pitch_deg": params["pitch_deg"],
            "phi_deg": params["phi_deg"],
            "x0": params["x_initial"],
            "y0": params["y_initial"],
            "z0": params["z_initial"],
            "B0_T": params["B_0"],
            "gyroperiods": gyroperiods,
            "norm_time": float(meta.attrs.get("norm_time")),
            "physical_time": float(meta.attrs.get("physical_time")),
            "percent_c": float(meta.attrs.get("percent_c")),
            "qoverm": params["qoverm"],
            "dtype": npfloat.__name__,  
            "timing": timing,

        },
        "ps": {"enabled": False},
        "rk4": {"enabled": False},
        "rk45": {"enabled": False},
        "rkg": {"enabled": False},
    }

    datasets = {}

    # ======= PS ========
    if "ps" in f:
        g = f["ps"]
        streaming = bool(g.attrs.get("streaming", False))

        # IMPORTANT: dataset handles only (lazy access)
        datasets["ps_y"] = g["y"] if "y" in g else None
        datasets["ps_orders"] = g["orders"] if "orders" in g else None

        # Authoritative max order (safe for large files)
        max_ps_used = (
            int(g.attrs["max_ps"])
            if "max_ps" in g.attrs
            else None
        )

        summary["ps"].update({
            "enabled": True,
            "dt": float(g.attrs.get("dt", ps_step)),
            "steps": int(g.attrs.get("steps", int(norm_time / ps_step))),
            "streaming": streaming,
            "ordercap": params["PS_order"],
            "max_ps": max_ps_used,              
            "decimate": int(g.attrs.get("decimate", 1)),
            "numberstepspergyro": int(np.round(T_gyro / g.attrs.get("dt", ps_step)) ),
            "E0": float(g.attrs.get("E0")),
            "mu0": float(g.attrs.get("mu0")),
            "tol": params["tol"],
        })

   
   
    # ======= RK4 ========
    if "rk4" in f:
        g = f["rk4"]
        summary["rk4"].update({
            "enabled": True,
            "dt": g.attrs.get("dt", rk4_step),
            "steps": g.attrs.get("steps", int(norm_time/rk4_step)),
            "numberstepspergyro": int(np.round(T_gyro/g.attrs.get("dt", rk4_step)))
        })
        datasets["rk4_y"] = g["y"]

    # ======= RK45 ========
    if "rk45" in f:
        summary["rk45"].update({
                    "enabled": True,
                    "atol": params["atol_rk45"],
                    "rtol": params["rtol_rk45"],
                })
        datasets["rk45_t"] = f["rk45"]["t"]
        datasets["rk45_y"] = f["rk45"]["y"]

    # ======= RKG ========
    if "rkg" in f:
        g = f["rkg"]
        summary["rkg"].update({
            "enabled": True,
            "dt": g.attrs.get("dt", rkg_step),
            "steps": g.attrs.get("steps", int(norm_time/rkg_step)),
            "numberstepspergyro": int(np.round(T_gyro/g.attrs.get("dt", rkg_step)))

        })
        datasets["rkg_y"] = g["y"]

    return summary, datasets, params, f


def write_dict(f, d, indent=0):
    pad = " " * indent
    for k, v in d.items():
        if isinstance(v, dict):
            f.write(f"{pad}{k}:\n")
            write_dict(f, v, indent + 2)
        else:
            f.write(f"{pad}{k} = {v}\n")

# ===================================
# === Decimate/Chunking Functions ===
# ===================================
def run_ps_streaming_with_decimation(
    initial_pos_vel_ps,
    steps_ps,
    ps_step,
    PS_order,
    tol,
    qoverm,
    E0_ps,
    mu0_ps,
    cache_path,
    write_data,
    chunk_steps,
    decimate,
    N_STEPS_PER_GYRO_ps,
    user_min_phase,
    dragt_monitor=None,
):
    start_time_ps = time.time()

    n_state = 17
    # SAVE_ROWS and n_save defined at module level

    cur_state = initial_pos_vel_ps.copy()
    remaining = steps_ps
    global_index = 0
    max_ps = 0
    hit_atmosphere = False
    hit_atm_step   = -1
    hit_atm_r      = 0.0
    R_ATMOSPHERE   = 1.0   # in R_E; change to 1.0157 for ~100 km altitude

    if write_data:
        f = h5py.File(cache_path, "w")
        ps_grp = f.create_group("ps")
        ps_grp.attrs["ordercap"] = PS_order
        ps_grp.attrs["numberstepspergyro"] = int(N_STEPS_PER_GYRO_ps)
        ps_grp.attrs["dt"]        = npfloat(ps_step)
        ps_grp.attrs["steps"]    = int(steps_ps)
        ps_grp.attrs["streaming"] = True
        ps_grp.attrs["chunksize"]= int(chunk_steps)
        ps_grp.attrs["decimate"] = int(decimate)
        ps_grp.attrs["tol"] = npfloat(tol)
        ps_grp.attrs["minphase"] = npfloat(user_min_phase)
        ps_grp.attrs["E0"]       = float(E0_ps)
        ps_grp.attrs["mu0"]      = float(mu0_ps)
        ps_grp.attrs["t0"]       = 0.0
        # Row layout: [x,y,z, vx,vy,vz, Bx,By,Bz]
        ps_grp.attrs["save_rows"] = "pos_vel_B"
        ps_grp.attrs["n_save"]    = n_save

        dset_y = ps_grp.create_dataset(
            "y",
            shape=(n_save, 0),
            maxshape=(n_save, None),
            dtype=npfloat,
            chunks=(n_save, min(chunk_steps, steps_ps + 1)),
            compression="gzip",
            compression_opts=2,
        )

        dset_orders = ps_grp.create_dataset(
            "orders",
            shape=(0,),
            maxshape=(None,),
            dtype=np.int16,
            compression="gzip",
            compression_opts=2,
        )
    else:
        f = None
        dset_y = dset_orders = None

    try:
        while remaining > 0:
            this_chunk = min(chunk_steps, remaining)

            sol_chunk, orders_chunk = PS_dipoleB(
                PS_order, this_chunk, cur_state, tol, qoverm, ps_step
            )

            idx_chunk = np.arange(global_index, global_index + this_chunk + 1)

            if global_index == 0:
                sol_eff = sol_chunk
                orders_eff = orders_chunk
                idx_eff = idx_chunk
            else:
                sol_eff = sol_chunk[:, 1:]
                orders_eff = orders_chunk[1:]
                idx_eff = idx_chunk[1:]

            if decimate <= 1:
                keep = np.ones_like(idx_eff, dtype=bool)
            else:
                keep = (idx_eff % decimate == 0) | (idx_eff == steps_ps)

            sol_keep = sol_eff[:, keep]
            orders_keep = orders_eff[keep]

            if sol_keep.shape[1] > 0:
                max_ps = max(max_ps, int(orders_keep.max()))

                if write_data:
                    old_len = dset_y.shape[1]
                    new_len = old_len + sol_keep.shape[1]

                    dset_y.resize((n_save, new_len))
                    dset_orders.resize((new_len,))

                    dset_y[:, old_len:new_len] = sol_keep[SAVE_ROWS, :]
                    dset_orders[old_len:new_len] = orders_keep

            # ---- atmospheric impact check (diagnostic only, does not halt) ----
            r_sq = sol_chunk[0]**2 + sol_chunk[1]**2 + sol_chunk[2]**2
            below = np.where(r_sq < R_ATMOSPHERE**2)[0]
            if len(below) > 0:
                r_min_chunk = float(np.sqrt(r_sq[below].min()))
                hit_atm_r = min(hit_atm_r, r_min_chunk) if hit_atmosphere else r_min_chunk
                if not hit_atmosphere:
                    hit_atmosphere = True
                    hit_atm_step = global_index + below[0]

            cur_state = sol_chunk[0:6, -1].copy()
            global_index += this_chunk
            remaining -= this_chunk

            # --- W0^2 / P_phi conservation check ---
            if dragt_monitor is not None:
                dragt_monitor.check(sol_chunk, step_index=global_index)

        if write_data:
            ps_grp.attrs["max_ps"] = max_ps
            ps_grp.attrs["hit_atmosphere"] = hit_atmosphere
            ps_grp.attrs["hit_atm_step"]   = hit_atm_step
            ps_grp.attrs["hit_atm_r"]      = hit_atm_r

        elapsed_ps = time.time() - start_time_ps

        if hit_atmosphere:
            print(f"\n    *** ATMOSPHERE FLAG: particle crossed r < {R_ATMOSPHERE} R_E "
                  f"at step {hit_atm_step:,} (r_min = {hit_atm_r:.4f} R_E) ***\n")

        return max_ps, elapsed_ps

    finally:
        if f is not None:
            f.close()


def slice_solution(t, sol, window_duration, norm_time, mode="last"):

    if mode == "last":
        t_end = norm_time
        t_start = max(t[0], t_end - window_duration)
    elif mode == "first":
        t_start = t[0]
        t_end = min(t[-1], t_start + window_duration)
    else:
        raise ValueError(f"Unknown slice mode: {mode}")

    idx = np.where((t >= t_start) & (t <= t_end))[0]

    if sol is None:
        return idx

    if sol.shape[0] <= sol.shape[1]:
        arr = sol
    else:
        arr = sol.T

    x = arr[0, idx]
    y = arr[1, idx]
    z = arr[2, idx]

    return x, y, z


def compute_energy_ps_chunked(
    ps_y_h5,
    E0_ps,
    dt_ps_store,
    chunk_cols=200000,
    stride=1,
    dtype=np.float64,
    return_plot_data=True,
):
    """
    Computes relative kinetic energy drift in a memory-efficient, chunked manner.
    Optionally returns decimated (stride-sampled) plot arrays only.
    """
    n_store = ps_y_h5.shape[1]

    if return_plot_data:
        # Estimate length of final array with stride
        n_points = (n_store + stride - 1) // stride
        t_plot = np.empty(n_points, dtype=npfloat)
        drift_plot = np.empty(n_points, dtype=npfloat)
        k = 0

    j_global = 0
    for j0 in range(0, n_store, chunk_cols):
        j1 = min(j0 + chunk_cols, n_store)
        v = ps_y_h5[3:6, j0:j1].astype(dtype, copy=False)

        E = 0.5 * np.sum(v * v, axis=0)
        rel = np.abs(E - E0_ps) / E0_ps

        if return_plot_data:
            for j_local in range(j1 - j0):
                if j_global % stride == 0:
                    t_plot[k] = j_global * dt_ps_store
                    drift_plot[k] = rel[j_local]
                    k += 1
                j_global += 1
        else:
            # If keeping full rel array
            raise NotImplementedError("Full array return not yet implemented in memory-saving mode")

    if return_plot_data:
        return t_plot[:k], drift_plot[:k]

def build_run_stem(summary, stem):
    r = summary["meta"]
    ps = summary["ps"]

    parts = [
        stem,
        "DipoleB_",
        r["particle"],
        f"{r['energy_eV']:.1e}eV",
        f"pitch{r['pitch_deg']}",
        f"phi{r['phi_deg']}",
        f"{r['norm_time']:.2e}s",
        r["dtype"],
    ]

    if ps["enabled"]:
        parts.insert(4, f"{ps['dt']}step_PS{ps['max_ps']}")

    return "_".join(parts)

def build_figure_filename(
    summary,
    output_folder,
    stem,
    figure_tag,
    ext="png"
):
    run_stem = build_run_stem(summary, stem)

    return (
        f"{output_folder}/{run_stem}_{figure_tag}.{ext}"
    )

# ===================================
# ======= Debug/Sanity Check =======
# ===================================

def check_time_grids(norm_time, ps_step=None, steps_ps=None,
                     rk4_step=None, steps_rk4=None,
                     rkg_step=None, steps_rkg=None,
                     rk45_t=None):

    lines = []

    def _report(label, step, steps):
        final_t = step * steps
        lines.append(
            f"{label}: step={step:.3e}, steps={steps}, final_time={final_t:.3e}"
        )

    if ps_step is not None and steps_ps is not None:
        _report("PS", ps_step, steps_ps)
    if rk4_step is not None and steps_rk4 is not None:
        _report("RK4", rk4_step, steps_rk4)
    if rkg_step is not None and steps_rkg is not None:
        _report("RKG", rkg_step, steps_rkg)
    if rk45_t is not None:
        lines.append(f"RK45: final time = {rk45_t[-1]:.3e}")

    return "\n".join(lines)


# ===================================================================
# Compute Canonical Angular Momentum (in normalized units, not Dragt)
# ===================================================================

def compute_pphi_error_chunked(
    cache_path, y0, charge_sign, ps_step, time_factor,
    chunk_cols=1_000_000, max_plot_points=500_000,
):
    """
    Compute the relative (or absolute) error of canonical angular momentum
    P_phi from chunked PS h5 data.

    Parameters
    ----------
    cache_path : str
        Path to the PS h5 file.
    y0 : array_like
        Initial 17-element state vector (compact h5 rows expanded).
        Needs positions (0-2) and velocities (3-5).
    charge_sign : float
        +1 for proton, -1 for electron.
    ps_step : float
        PS step size (normalized time units).
    time_factor : float
        Conversion factor from normalized time to gyroperiods (1/T_gyro).
    chunk_cols : int
        Number of columns to read per h5 chunk.
    max_plot_points : int
        Target max points for the decimated output arrays.

    Returns
    -------
    dict with keys:
        "t_gyro"        : 1D array, time in gyroperiods (decimated)
        "rel_error_log" : 1D array, error with zeros replaced by 1e-16
        "max_err"       : float, global max error
        "P_phi_initial" : float, initial canonical angular momentum
        "ylabel"        : str, appropriate axis label
    """
    # --- Initial P_phi ---
    rho0  = np.sqrt(y0[0]**2 + y0[1]**2)
    r0    = np.sqrt(y0[0]**2 + y0[1]**2 + y0[2]**2)
    vphi0 = (y0[0]*y0[4] - y0[1]*y0[3]) / rho0
    P_phi_initial = (rho0 * vphi0) - charge_sign * (rho0**2 / r0**3)

    # --- Chunked read ---
    with h5py.File(cache_path, "r") as h5:
        ds = h5["ps"]["y"]
        N = ds.shape[1]
        dec = max(1, N // max_plot_points)
        err_dec = []
        max_err = 0.0

        for i0 in range(0, N, chunk_cols):
            i1 = min(i0 + chunk_cols, N)
            ch = ds[:6, i0:i1]
            rho = np.sqrt(ch[0]**2 + ch[1]**2)
            r   = np.sqrt(ch[0]**2 + ch[1]**2 + ch[2]**2)
            vp  = (ch[0]*ch[4] - ch[1]*ch[3]) / rho
            pp  = (rho * vp) - charge_sign * (rho**2 / r**3)

            if P_phi_initial == 0:
                err = np.abs(pp)
            else:
                err = np.abs((pp - P_phi_initial) / P_phi_initial)

            cm = float(np.max(err))
            if cm > max_err:
                max_err = cm

            err_dec.append(err[::dec])
            del ch, rho, r, vp, pp, err

    rel_error = np.concatenate(err_dec)
    rel_error_log = np.where(rel_error == 0, 1e-16, rel_error)
    t_gyro = ps_step * np.arange(len(rel_error_log), dtype=np.float64) * dec * time_factor

    ylabel = (r"Absolute Error $|\Delta P_\phi|$" if P_phi_initial == 0
              else r"Relative Error $|(P_\phi - P_{\phi,0}) / P_{\phi,0}|$")

    return {
        "t_gyro":        t_gyro,
        "rel_error_log": rel_error_log,
        "max_err":       max_err,
        "P_phi_initial": P_phi_initial,
        "ylabel":        ylabel,
    }


# ===================================================================
# ============ Gyro-window index helper =============================
# ===================================================================
def _gyro_window_indices(gyro_window, total_steps, window_steps):
    """Return (i0, i1) slice indices for first/last/all gyro window."""
    if gyro_window == "last":
        i1 = total_steps
        i0 = max(0, i1 - window_steps)
    elif gyro_window == "first":
        i0 = 0
        i1 = min(window_steps, total_steps)
    elif gyro_window == "all":
        i0 = 0
        i1 = total_steps
    else:
        raise ValueError("gyro_window must be 'first', 'last', or 'all'")
    return i0, i1


# ===================================================================
# ============ Mu deviation — RK solvers (in-memory) ================
# ===================================================================
def compute_mu_deviation_rk(
    solution, steps, dt, N_GYRO, N_STEPS_PER_GYRO,
    mass, gyro_window, time_factor,
    solver_type="rk4",
):
    """
    Compute magnetic moment deviation for an in-memory RK solver solution.

    Parameters
    ----------
    solution : ndarray
        For RK4/RK45: shape (6, N) — columns are time steps.
        For RKG: shape (N, 6) — rows are time steps (Hamiltonian format).
    steps : int
        Total number of integration steps.
    dt : float
        Step size (normalized time).
    N_GYRO : int
        Number of gyroperiods in the analysis window.
    N_STEPS_PER_GYRO : float
        Steps per gyroperiod for this solver.
    mass : float
        Relativistic mass (gamma * m_si).
    gyro_window : str
        "first", "last", or "all".
    time_factor : float
        Conversion from normalized time to gyroperiods (1/T_gyro).
    solver_type : str
        "rk4", "rk45", or "rkg". Controls data layout and mu computation.

    Returns
    -------
    dict with keys:
        "t"       : 1D time array in gyroperiods
        "mudrift" : 1D relative mu deviation array
        "mu0"     : float, initial magnetic moment
    """
    window_steps = int(round(N_GYRO * N_STEPS_PER_GYRO))
    i0, i1 = _gyro_window_indices(gyro_window, steps, window_steps)

    if solver_type == "rkg":
        # RKG stores (N, 6) with canonical momentum — need to convert to velocity
        r0 = solution[0, 0:3]
        p0 = solution[0, 3:6]
        A0 = vector_potential_dipole(r0)
        v0 = p0 - A0
        state0 = np.hstack((r0, v0))[None, :]
        mu0 = compute_mu_rk(state0, mass)[0]

        r_win = solution[i0:i1, 0:3]
        p_win = solution[i0:i1, 3:6]
        A_win = np.empty_like(r_win)
        for i in range(len(r_win)):
            A_win[i] = vector_potential_dipole(r_win[i])
        v_win = p_win - A_win
        state_win = np.hstack((r_win, v_win))
        mu_win = compute_mu_rk(state_win, mass)
    else:
        # RK4 and RK45: shape (6, N) — columns are time steps
        mu0 = compute_mu_rk(solution[:, 0:1].T, mass)[0]
        mu_win = compute_mu_rk(solution[:, i0:i1].T, mass)

    mudrift = np.abs(mu_win - mu0) / mu0
    t = (i0 + np.arange(mudrift.size, dtype=np.float64)) * dt * time_factor

    return {"t": t, "mudrift": mudrift, "mu0": mu0}


# ===================================================================
# ============ Mu deviation — PS (chunked h5) =======================
# ===================================================================
def compute_mu_deviation_ps(
    cache_path, steps_ps, ps_step, PS_decimate,
    N_GYRO, N_STEPS_PER_GYRO, mass, mu0_ps,
    gyro_window, time_factor,
    max_plot_points=1_000_000,
):
    """
    Compute magnetic moment deviation for PS data from chunked h5.

    Parameters
    ----------
    cache_path : str
        Path to the PS h5 file.
    steps_ps : int
        Total physical integration steps.
    ps_step : float
        PS step size (normalized time).
    PS_decimate : int
        Decimation factor used during streaming.
    N_GYRO : int
        Number of gyroperiods in the analysis window.
    N_STEPS_PER_GYRO : float
        Steps per gyroperiod (on the PS grid).
    mass : float
        Relativistic mass.
    mu0_ps : float
        Initial magnetic moment (PS).
    gyro_window : str
        "first", "last", or "all".
    time_factor : float
        Conversion from normalized time to gyroperiods.
    max_plot_points : int
        Cap on output array length for plotting.

    Returns
    -------
    dict with keys:
        "t"              : 1D time array in gyroperiods (decimated for plotting)
        "mudrift"        : 1D full mu deviation array
        "mudrift_plot"   : 1D decimated mu deviation for plotting
        "ps_order_label" : int, max PS order from h5 attrs
    """
    window_steps = N_GYRO * N_STEPS_PER_GYRO
    i0_phys, i1_phys = _gyro_window_indices(gyro_window, steps_ps, window_steps)

    ps_store_stride = PS_decimate if (PS_decimate > 1) else 1
    j0 = int(np.floor(i0_phys / ps_store_stride))
    j1 = int(np.ceil(i1_phys / ps_store_stride))

    with h5py.File(cache_path, "r") as ps_h5:
        ps_grp = ps_h5["ps"]
        ps_y = ps_grp["y"]
        ps_order_label = int(ps_grp.attrs["max_ps"])
        n_store = ps_y.shape[1]

        j0 = max(0, min(j0, n_store))
        j1 = max(0, min(j1, n_store))

        if j1 <= j0:
            raise RuntimeError("Empty PS mu window (chunked)")

        y_ps_win = expand_h5_to_full(ps_y[:, j0:j1])

    mu_ps = compute_mu_ps(y_ps_win, mass)
    mudrift = np.abs(mu_ps - mu0_ps) / mu0_ps

    dt_ps_store = ps_step * ps_store_stride
    t_store = np.arange(j0, j1, dtype=np.float64) * dt_ps_store
    moment_stride = max(1, round(len(mu_ps) // max_plot_points))
    t_plot = t_store[::moment_stride] * time_factor
    mudrift_plot = mudrift[::moment_stride]

    return {
        "t":              t_plot,
        "mudrift":        mudrift,
        "mudrift_plot":   mudrift_plot,
        "ps_order_label": ps_order_label,
    }
