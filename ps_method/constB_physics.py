"""
constB_physics.py — Physics kernels for charged particle motion in a
                    uniform (constant) magnetic field.

Three solvers are provided:
  • PS_constB           – Power-series integrator with adaptive truncation order.
  • analytical_constantB – Exact solution to system of equations.
  • lorentz_force_constB – RHS function for scipy / RK4 integrators (dp/dt = qv × B).

All functions are compiled with @maybe_njit (skipped when float128 is active).
"""

import numpy as np
from numba import njit
import os
from ps_method.universal import maybe_njit, npfloat

one = npfloat(1.0)


@maybe_njit
def PS_constB(order_max, steps, initial_pos_vel, timedelta, Bfield, qoverm, tol):
    """Advance a charged particle through a uniform B field using a power-series method.

    At each time step the coefficients c[n] of position and velocity are
    built recursively:
        x_{n+1} = (1/(n+1)) * vx_n          (position from velocity)
        vx_{n+1} = (1/(n+1)) * q/m (v × B)_n  (velocity from Lorentz force)
    The series is summed until the largest coefficient drops below `tol` or
    `order_max` terms have been used.

    Returns
    -------
    final_coeff_matrix : (6, steps+1) array — full trajectory [x,y,z,vx,vy,vz].
    orders_used        : (steps+1,) int array — PS order actually used per step.
    """
    n_total = 6
    final_coeff_matrix = np.zeros((n_total, steps + 1), dtype=npfloat)
    final_coeff_matrix[:, 0] = initial_pos_vel
    oip1 = one / (one + np.arange(order_max))       # 1/(i+1) lookup table
    orders_used = np.zeros(steps + 1, dtype=np.int32)

    # Named indices for readability
    x, y, z = 0, 1, 2
    vx, vy, vz = 3, 4, 5

    for j in range(1, steps + 1):
        c = np.zeros((n_total, order_max + 1), dtype=npfloat)
        c[:, 0] = final_coeff_matrix[:, j - 1]     # seed with end of previous step
        power = timedelta
        sum_terms = np.zeros(n_total, dtype=npfloat)
        max_contrib = tol + one
        i = 0

        while max_contrib > tol and i < order_max:
            # Position coefficients: dx/dt = v  →  x_{n+1} = v_n / (n+1)
            c[x, i+1] = oip1[i] * c[vx, i]
            c[y, i+1] = oip1[i] * c[vy, i]
            c[z, i+1] = oip1[i] * c[vz, i]

            # Velocity coefficients: dv/dt = (q/m)(v × B)  →  cross product at order n
            c[vx, i+1] = oip1[i] * qoverm * (Bfield[2]*c[vy, i] - Bfield[1]*c[vz, i])
            c[vy, i+1] = oip1[i] * qoverm * (Bfield[0]*c[vz, i] - Bfield[2]*c[vx, i])
            c[vz, i+1] = oip1[i] * qoverm * (Bfield[1]*c[vx, i] - Bfield[0]*c[vy, i])

            sum_terms += c[:, i+1] * power        
            max_contrib = np.abs(c[:, i+1]).max()  
            power *= timedelta
            i += 1

        final_coeff_matrix[:, j] = final_coeff_matrix[:, j - 1] + sum_terms
        orders_used[j] = i

    return final_coeff_matrix, orders_used


@maybe_njit
def analytical_constantB(tau, d, qoverm):
    """Exact closed-form trajectory in a uniform magnetic field.
    In normalized coordinates (ω_c = |qB/m| = 1) 

    Parameters
    ----------
    tau     : 1-D array of normalized times.
    d       : length-6 initial state [x0, y0, z0, vx0, vy0, vz0].
    qoverm  : sign of q/m (+1 proton, −1 electron in a +z field).

    Returns
    -------
    (6, len(tau)) array — [x, y, z, vx, vy, vz] at each time.
    """
    x0, y0, z0, vx0, vy0, vz0 = d

    s = np.sign(qoverm)

    sin_t = np.sin(s * tau)
    cos_t = np.cos(s * tau)

    x_t = x0 + s * (vy0 * (1 - cos_t) + vx0 * sin_t)
    y_t = y0 + s * (-vx0 * (1 - cos_t) + vy0 * sin_t)
    z_t = z0 + vz0 * tau

    vx_t = vx0 * cos_t - vy0 * sin_t
    vy_t = vy0 * cos_t + vx0 * sin_t
    vz_t = vz0 * np.ones_like(tau)

    return np.vstack((x_t, y_t, z_t, vx_t, vy_t, vz_t))


@maybe_njit
def lorentz_force_constB(t, d, Bfield, qoverm):
    """Right-hand side for the Lorentz equation of motion in a uniform B field.

    Returns d/dt [x, y, z, vx, vy, vz] 
    Used as the RHS call for scipy.integrate.solve_ivp (RK45) and rk4_fixed_step.
    """
    x, y, z, vx, vy, vz = d
    dvx = qoverm * (vy * Bfield[2] - vz * Bfield[1])
    dvy = qoverm * (vz * Bfield[0] - vx * Bfield[2])
    dvz = qoverm * (vx * Bfield[1] - vy * Bfield[0])

    return np.array([vx, vy, vz, dvx, dvy, dvz])

