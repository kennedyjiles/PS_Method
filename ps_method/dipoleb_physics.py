"""
Physics kernels and analysis for charged particle motion in a magnetic dipole.

Core solvers:
    lorentz_force                — dipole Lorentz force (numba-compiled)
    ps_integrate                 — power series integrator (chunked, streamed to h5)
    hamiltonian_rhs              — Hamilton's equations for symplectic integrator
    rkgl4_hamiltonian_step       — single implicit Gauss-Legendre RK4 step
    rkgl4_hamiltonian            — full symplectic integration loop

Vector potential:
    vector_potential             — A_phi for canonical momentum

Streaming / decimation:
    run_ps_streaming_with_decimation — full PS run with decimated output


"""

import numpy as np
import os
import json, hashlib, h5py, time, re
from numba import njit
from ps_method.utils import npfloat, maybe_njit
from ps_method.writers import SAVE_ROWS, n_save, expand_h5_to_full, build_filename

one = npfloat(1.0)
two = npfloat(2.0)
three = npfloat(3.0)
five = npfloat(5.0)
twopointfive = npfloat(2.5)

@maybe_njit
def lorentz_force(t, y, qoverm):
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
def ps_integrate(PS_order, steps_ps, initial_pos_vel, tol, qoverm, timedelta):
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

# ========================
# ==== RKG Functions ====
# ========================

@maybe_njit
def vector_potential(r):
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

@maybe_njit
def rkgl4_hamiltonian_step(func, y0, dt, args=(), max_iter=10, tol=1e-12, eps=1e-13):
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

                K[i, j] = K_save[i, j]  

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
            compression_opts=1,
            shuffle=True,
        )

        dset_orders = ps_grp.create_dataset(
            "orders",
            shape=(0,),
            maxshape=(None,),
            dtype=np.int16,
            compression="gzip",
            compression_opts=1,
            shuffle=True,
        )
    else:
        f = None
        dset_y = dset_orders = None

    try:
        while remaining > 0:
            this_chunk = min(chunk_steps, remaining)

            sol_chunk, orders_chunk = ps_integrate(
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
