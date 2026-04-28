import numpy as np
import builtins
import os
import json
from numba import njit
from .universal import cauchy_sum, maybe_njit, npfloat

@maybe_njit
def lorentz_force_hyperB(t, y, gamma, qoverm):
    y = y.astype(npfloat)
    gamma = npfloat(gamma)

    Bz = np.tanh(gamma * y[1])
    ax = qoverm * y[4] * Bz
    ay = - qoverm * y[3] * Bz
    az = npfloat(0.0)

    return np.array([y[3], y[4], y[5], ax, ay, az], dtype=npfloat)

@maybe_njit
def PS_hyperB(PS_order, steps_ps, initial_pos_vel, timedelta, gamma, qoverm, tol):
    n_total = 9        # x, y, z, v_x, v_y, v_z, sinh, cosh, Bz 
    final_coeff_matrix = np.zeros((n_total, steps_ps + 1), dtype=npfloat)
    
    # Labeling indices for sanity tracking
    x, y, z = 0, 1, 2
    vx, vy, vz = 3, 4, 5
    sinh_aux, cosh_aux = 6, 7
    Bz_aux  = 8

    # Setting up Initial conditions
    final_coeff_matrix[0:6, 0] = initial_pos_vel
    y0 = initial_pos_vel[1]
    
    final_coeff_matrix[sinh_aux, 0] = np.sinh(gamma * y0)
    final_coeff_matrix[cosh_aux, 0] = np.cosh(gamma * y0)
    final_coeff_matrix[Bz_aux, 0]   = np.tanh(gamma * y0)

    Bz_series   = np.zeros(PS_order, dtype=npfloat)
    orders_used = np.zeros(steps_ps + 1, dtype=np.int32)
    oip1        = 1.0 / (1.0 + np.arange(PS_order))

    for j in range(1, steps_ps + 1):
        c = np.zeros((n_total, PS_order + 1), dtype=npfloat)
        c[:, 0] = final_coeff_matrix[:, j - 1]

        sum_terms = np.zeros(n_total, dtype=npfloat)
        power     = timedelta
        i         = 0
        max_contrib = tol + npfloat(1.0)

        while max_contrib > tol and i < PS_order:

            # --- Core Recurrences ---
            if i == 0:
                Bz_series[0] = c[sinh_aux, 0] / c[cosh_aux, 0]
            else:
                s = c[sinh_aux, i]
                for k in range(1, i+1):
                    s -= c[cosh_aux, k] * Bz_series[i-k]
                Bz_series[i] = s / c[cosh_aux, 0]

            vyBz = cauchy_sum(c[vy], Bz_series, i)
            vxBz = cauchy_sum(c[vx], Bz_series, i)

            c[x, i+1]  = oip1[i] * c[vx, i]
            c[y, i+1]  = oip1[i] * c[vy, i]
            c[z, i+1]  = oip1[i] * c[vz, i]
            c[vx, i+1] = qoverm * oip1[i] * vyBz
            c[vy, i+1] = -qoverm * oip1[i] * vxBz
            c[vz, i+1] = 0.0

            c[sinh_aux, i+1] = oip1[i] * gamma * cauchy_sum(c[cosh_aux], c[vy], i)
            c[cosh_aux, i+1] = oip1[i] * gamma * cauchy_sum(c[sinh_aux], c[vy], i)
            c[Bz_aux,   i+1] = Bz_series[i]

            if not np.isfinite(c[:, i+1]).all(): # internal overflow → stop series immediately
                break

            sum_terms += c[:, i+1] * power
            max_contrib = np.abs(c[:, i+1]).max()
            power *= timedelta
            i += 1

        final_coeff_matrix[:, j] = final_coeff_matrix[:, j-1] + sum_terms
        orders_used[j] = i

        # --- tethering ---
        y_now = final_coeff_matrix[y, j]
        sinh_now = np.sinh(gamma * y_now)
        cosh_now = np.cosh(gamma * y_now)
        Bz_now = sinh_now / cosh_now

        final_coeff_matrix[sinh_aux, j] = sinh_now
        final_coeff_matrix[cosh_aux, j] = cosh_now
        final_coeff_matrix[Bz_aux, j] = Bz_now

    return final_coeff_matrix, orders_used

# I/O functions (save/load h5, run params, hashing) have moved to ps_method/writers.py
