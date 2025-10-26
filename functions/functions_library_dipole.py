import numpy as np
import builtins
import os
import json, hashlib, h5py
from numba import njit
from functions.functions_library_universal import rk4_fixed_step, extract_v, compute_energy_drift, npfloat, maybe_njit

one = npfloat(1.0)
two = npfloat(2.0)
three = npfloat(3.0)
five = npfloat(5.0)
twopointfive = npfloat(2.5)
# if z is oriented in the positive direction uncomment these top two and comment out the others
# @((lambda f: f) if npfloat == np.float128 else njit)
# def lorentz_force_dipole(t, y, qoverm):
#     # Unpack position and velocity
#     x, y_, z, vx, vy, vz = y
#     r2 = x**two + y_**two + z**two
#     r5inv = r2**(-twopointfive) if r2 != 0 else 0.0

#     # Magnetic field components
#     Bx = three * x * z * r5inv
#     By = three * y_ * z * r5inv
#     Bz = (three * z**two - r2) * r5inv

#     # Lorentz force
#     ax = qoverm * (vy * Bz - vz * By)
#     ay = qoverm * (vz * Bx - vx * Bz)
#     az = qoverm * (vx * By - vy * Bx)

#     return np.array([vx, vy, vz, ax, ay, az], dtype=npfloat)

# @((lambda f: f) if npfloat == np.float128 else njit)
# def PS_dipoleB(PS_order, steps_ps, initial_pos_vel, tol, qoverm, timedelta):
#     n_total = 17
#     final_coeff_matrix = np.zeros((n_total, steps_ps + 1), dtype=npfloat)

#     x, y, z, vx, vy, vz = 0, 1, 2, 3, 4, 5
#     r2_aux, a_aux, b_aux, c_aux, d_aux, e_aux, f_aux, g_aux = 6, 7, 8, 9, 10, 11, 12, 13
#     Bx_aux, By_aux, Bz_aux = 14, 15, 16

#     final_coeff_matrix[0:6, 0] = initial_pos_vel
#     x0, y0, z0 = initial_pos_vel[0], initial_pos_vel[1], initial_pos_vel[2]
#     vx0, vy0, vz0 = initial_pos_vel[3], initial_pos_vel[4], initial_pos_vel[5]

#     r2_0 = x0**two + y0**two + z0**two
#     a0 = r2_0**(-twopointfive)
#     b0 = two * z0**2 - x0**2 - y0**2
#     c0 = y0 * z0
#     d0 = x0 * z0
#     e0 = b0 * vy0 - three * c0 * vz0
#     f0 = three * d0 * vz0 - b0 * vx0
#     g0 = three * c0 * vx0 - three * d0 * vy0

#     final_coeff_matrix[r2_aux, 0] = r2_0
#     final_coeff_matrix[a_aux, 0] = a0
#     final_coeff_matrix[b_aux, 0] = b0
#     final_coeff_matrix[c_aux, 0] = c0
#     final_coeff_matrix[d_aux, 0] = d0
#     final_coeff_matrix[e_aux, 0] = e0
#     final_coeff_matrix[f_aux, 0] = f0
#     final_coeff_matrix[g_aux, 0] = g0

#     final_coeff_matrix[Bz_aux, 0] = a0 * b0
#     final_coeff_matrix[By_aux, 0] = npfloat(3.0) * a0 * c0
#     final_coeff_matrix[Bx_aux, 0] = npfloat(3.0) * a0 * d0

#     oip1 = one / (one + np.arange(PS_order, dtype=npfloat))
#     orders_used = np.zeros(steps_ps + 1, dtype=np.int32)

#     def cauchy_sum_inline(a, b, n):
#         result = 0.0
#         for j in range(n + 1):
#             result += a[j] * b[n - j]
#         return result  

#     def cauchy_divide(a, b, out, n):        #computiing zeta=a/b up to through n and stores it as out[:]
#         out[0] = a[0] / b[0]
#         for i in range(1, n+1):
#             acc = a[i]
#             for j in range(1, i + 1):
#                 acc -= b[j] * out[i - j]
#             out[i] = acc / b[0]


#     c = np.zeros((n_total, PS_order + 1), dtype=npfloat)
#     sum_terms = np.zeros(n_total, dtype=npfloat)
#     zeta = np.zeros(PS_order + 1, dtype=npfloat)

#     # initialize base terms outside the loop
#     c[r2_aux, 0] = final_coeff_matrix[x, 0]**two + final_coeff_matrix[y, 0]**two + final_coeff_matrix[z, 0]**two
#     c[a_aux, 0] = c[r2_aux, 0]**(-twopointfive)
#     zeta[0] = c[a_aux, 0] / c[r2_aux, 0]

#     for j in range(1, steps_ps + 1):
#         c[:, 0] = final_coeff_matrix[:, j - 1]
#         sum_terms[:] = 0

#         power = timedelta
#         max_contrib = tol + one
#         i = 0

#         while max_contrib > tol and i < PS_order:
#             c[x, i+1]  = c[vx, i] * oip1[i]
#             c[y, i+1]  = c[vy, i] * oip1[i]
#             c[z, i+1]  = c[vz, i] * oip1[i]
#             c[vx, i+1] = qoverm * cauchy_sum_inline(c[a_aux], c[e_aux], i) * oip1[i]
#             c[vy, i+1] = qoverm * cauchy_sum_inline(c[a_aux], c[f_aux], i) * oip1[i]
#             c[vz, i+1] = qoverm * cauchy_sum_inline(c[a_aux], c[g_aux], i) * oip1[i]

#             c[r2_aux, i+1] = cauchy_sum_inline(c[x], c[x], i+1) + cauchy_sum_inline(c[y], c[y], i+1) + cauchy_sum_inline(c[z], c[z], i+1)
#             cauchy_divide(c[a_aux], c[r2_aux], zeta, i+1)      #This is modifying zeta in place 
#             a_prime = 0.0
#             for k in range(i+1):
#                 a_prime += (i - k + 1) * zeta[k] * c[r2_aux, i - k + 1]
#             c[a_aux, i+1] = - (five / (two * (i + 1))) * a_prime

#             c[b_aux, i+1] = two * cauchy_sum_inline(c[z], c[z], i+1) - cauchy_sum_inline(c[x], c[x], i+1) - cauchy_sum_inline(c[y], c[y], i+1)
#             c[c_aux, i+1] = cauchy_sum_inline(c[y], c[z], i+1)
#             c[d_aux, i+1] = cauchy_sum_inline(c[x], c[z], i+1)

#             c[e_aux, i+1] = cauchy_sum_inline(c[b_aux], c[vy], i+1) - three * cauchy_sum_inline(c[c_aux], c[vz], i+1)
#             c[f_aux, i+1] = three * cauchy_sum_inline(c[d_aux], c[vz], i+1) - cauchy_sum_inline(c[b_aux], c[vx], i+1)
#             c[g_aux, i+1] = three * (cauchy_sum_inline(c[c_aux], c[vx], i+1) - cauchy_sum_inline(c[d_aux], c[vy], i+1))

#             c[Bx_aux, i+1] = npfloat(3.0) * cauchy_sum_inline(c[a_aux], c[d_aux], i+1)
#             c[By_aux, i+1] = npfloat(3.0) * cauchy_sum_inline(c[a_aux], c[c_aux], i+1)
#             c[Bz_aux, i+1] =        cauchy_sum_inline(c[a_aux], c[b_aux], i+1)

#             sum_terms += c[:, i+1] * power
#             max_contrib = np.abs(c[:, i+1]).max()
#             power *= timedelta
#             i += 1

#         final_coeff_matrix[:, j] = final_coeff_matrix[:, j - 1] + sum_terms

#         x_now, y_now, z_now = final_coeff_matrix[x, j], final_coeff_matrix[y, j], final_coeff_matrix[z, j]
#         vx_now, vy_now, vz_now = final_coeff_matrix[vx, j], final_coeff_matrix[vy, j], final_coeff_matrix[vz, j]

#         r2_now = x_now**two + y_now**two + z_now**two
#         a_now = r2_now**(-twopointfive)
#         b_now = two * z_now**two - x_now**two - y_now**two
#         c_now = y_now * z_now
#         d_now = x_now * z_now

#         e_now = b_now * vy_now - three * c_now * vz_now
#         f_now = three * d_now * vz_now - b_now * vx_now
#         g_now = three * c_now * vx_now - three * d_now * vy_now

#         final_coeff_matrix[r2_aux, j] = r2_now
#         final_coeff_matrix[a_aux, j] = a_now
#         final_coeff_matrix[b_aux, j] = b_now
#         final_coeff_matrix[c_aux, j] = c_now
#         final_coeff_matrix[d_aux, j] = d_now
#         final_coeff_matrix[e_aux, j] = e_now
#         final_coeff_matrix[f_aux, j] = f_now
#         final_coeff_matrix[g_aux, j] = g_now
#         final_coeff_matrix[Bx_aux, j] = npfloat(3.0) * a_now * d_now
#         final_coeff_matrix[By_aux, j] = npfloat(3.0) * a_now * c_now
#         final_coeff_matrix[Bz_aux, j] = a_now * b_now

#         orders_used[j] = i

#     return final_coeff_matrix, orders_used

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
        # while i < PS_order:
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
            max_contrib = np.abs(c[:, i+1]).max()
            # max_contrib = np.abs(c[:, i+1] * power).max()
            power *= timedelta
            i += 1

        final_coeff_matrix[:, j] = final_coeff_matrix[:, j - 1] + sum_terms

        x_now, y_now, z_now = final_coeff_matrix[x, j], final_coeff_matrix[y, j], final_coeff_matrix[z, j]
        vx_now, vy_now, vz_now = final_coeff_matrix[vx, j], final_coeff_matrix[vy, j], final_coeff_matrix[vz, j]

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
        B = np.array([
            3 * x * z * r5inv,
            3 * y * z * r5inv,
            (3 * z**2 - r2) * r5inv
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


# @((lambda f: f) if npfloat == np.float128 else njit)
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
def rkgl4_hamiltonian_step(func, y0, dt, args=(), max_iter=10, tol=1e-12, eps=1e-13):  # when using with the Lorentz force I used tol=1e-12 and eps=1e-12
    sqrt3 = np.sqrt(3.0)
    a11, a12 = 0.25, 0.25 - sqrt3 / 6.0
    a21, a22 = 0.25 + sqrt3 / 6.0, 0.25
    b1 = b2 = 0.5

    dim = len(y0)
    K = np.zeros((2, dim), dtype=npfloat)

    # initial guess from explicit Euler
    K[0] = func(0.0, y0, *args)
    K[1] = K[0].copy()

    for n in range(max_iter):
        # stage values
        Y1 = y0 + dt * (a11 * K[0] + a12 * K[1])
        Y2 = y0 + dt * (a21 * K[0] + a22 * K[1])

        F1 = K[0] - func(0.0, Y1, *args)
        F2 = K[1] - func(0.0, Y2, *args)
        F = np.concatenate((F1, F2))

        # convergence check
        normF = np.max(np.abs(F))
        if normF < tol:
            break

        # finite-difference Jacobian J (block 2x2)
        J = np.zeros((2 * dim, 2 * dim), dtype=npfloat)
        for i in range(2):
            for j in range(dim):
                dK = np.zeros((2, dim), dtype=npfloat)
                dK[i, j] = eps
                Y1_pert = y0 + dt * (a11 * (K[0] + dK[0]) + a12 * (K[1] + dK[1]))
                Y2_pert = y0 + dt * (a21 * (K[0] + dK[0]) + a22 * (K[1] + dK[1]))

                F1_pert = K[0] + dK[0] - func(0.0, Y1_pert, *args)
                F2_pert = K[1] + dK[1] - func(0.0, Y2_pert, *args)
                F_pert = np.concatenate((F1_pert, F2_pert))

                dF = (F_pert - F) / eps
                J[:, i * dim + j] = dF
        try:
            dK_flat = np.linalg.solve(J, -F)
        except:
            raise RuntimeError("Newton step failed: singular Jacobian")

        K_flat = np.concatenate((K[0], K[1])) + dK_flat
        K[0] = K_flat[:dim]
        K[1] = K_flat[dim:]

    else:
        print("⚠️ Newton did not converge")

    return y0 + dt * (b1 * K[0] + b2 * K[1])


@maybe_njit
def rkgl4_hamiltonian(func, y0, t_eval, args=()):
    nt = len(t_eval)
    d_out = np.zeros((nt, len(y0)), dtype=np.float64)
    d_out[0] = y0
    for i in range(1, nt):
        dt = t_eval[i] - t_eval[i - 1]
        d_out[i] = rkgl4_hamiltonian_step(func, d_out[i - 1], dt, args)
    return d_out

# ========================
# === Mirror Functions ===
# ========================
named_indices = {"vx":3,"vy":4,"vz":5,"Bx":14,"By":15,"Bz":16}

"""
Find mirror crossings (v·B=0) in tau units.
s_eps: magnitude threshold to ignore tiny jitters (units of v·B in your norm).
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

def get_run_params(USE_RK45, USE_RK4, USE_RKG,
                   mass_si, q_e, B_0, gamma,
                   x_initial, y_initial, z_initial,
                   pitch_deg, phi_deg,
                   norm_time, ps_step, rk4_step, rkg_step,
                   PS_order, tol, qoverm):
    """Collect all knobs that define a run."""
    return {
        # toggles
        "USE_RK45": bool(USE_RK45),
        "USE_RK4":  bool(USE_RK4),
        "USE_RKG":  bool(USE_RKG),

        # physics & normalization
        "mass_si": _to_serializable(mass_si),
        "q_e": _to_serializable(q_e),
        "B_0": _to_serializable(B_0),
        "gamma": _to_serializable(gamma),

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
        "rtol_rk45": 1e-8,
        "atol_rk45": 1e-10,

        # charge/mass normalization used in RHS
        "qoverm": _to_serializable(qoverm),
    }

def run_hash(params: dict) -> str:
    j = json.dumps(params, sort_keys=True, default=_to_serializable, separators=(",",":"))
    return hashlib.sha1(j.encode("utf-8")).hexdigest()[:16]

def h5_path_for(params, output_folder):
    return os.path.join(output_folder, f"run_{run_hash(params)}.h5")

def save_results_h5(h5_path, params, results):
    """
    results expects keys like:
      'ps': {'t': t_eval_ps, 'y': solution_ps_rescaled, 'orders': orders_used}
      'rk4': {'t': t_eval_rk4, 'y': solution_rk4}
      'rk45': {'t': solution_rk45.t, 'y': solution_rk45.y}
      'rkg': {'t': t_eval_rkg, 'y': solution_rkg}
      'meta': {'timing': {...}, 'physical_time': physical_time, ...}
    """
    with h5py.File(h5_path, "w") as f:
        # store params as a single JSON attribute on root
        f.attrs["params_json"] = json.dumps(params, sort_keys=True, default=_to_serializable)

        for k in ("ps","rk4","rk45","rkg"):
            if k in results and results[k] is not None:
                grp = f.create_group(k)
                for name, arr in results[k].items():
                    if arr is None: 
                        continue
                    grp.create_dataset(name, data=arr, compression="gzip", compression_opts=2)

        # meta info
        meta = results.get("meta", {})
        gmeta = f.create_group("meta")
        # store timing dict as attrs
        for mk, mv in meta.get("timing", {}).items():
            gmeta.attrs[f"timing_{mk}"] = float(mv)
        # scalar attrs
        for sk in ("physical_time","norm_time","percent_c","particle_label"):
            if sk in meta:
                gmeta.attrs[sk] = meta[sk]

def load_results_h5(h5_path):
    with h5py.File(h5_path, "r") as f:
        loaded = {"meta": {"timing": {}}}
        # params
        loaded["params"] = json.loads(f.attrs["params_json"])

        # helper to pull groups
        def _read_grp(name):
            if name not in f: return None
            g = f[name]
            out = {}
            for ds in g:
                out[ds] = g[ds][...]
            return out

        for k in ("ps","rk4","rk45","rkg"):
            loaded[k] = _read_grp(k)

        # meta attrs
        gmeta = f["meta"]
        for a in gmeta.attrs:
            if a.startswith("timing_"):
                loaded["meta"]["timing"][a.replace("timing_","")] = gmeta.attrs[a]
            else:
                loaded["meta"][a] = gmeta.attrs[a]

        return loaded
