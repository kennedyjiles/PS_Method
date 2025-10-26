import builtins
import numpy as np
from numba import njit
from functions.functions_library_universal import maybe_njit, npfloat

one = npfloat(1.0)

@maybe_njit
def PS_constantB_adaptive(order_max, steps, initial_pos_vel, timedelta, Bfield, qoverm, tol):
    n_total = 6        # x, y, z, v_x, v_y, v_z
    final_coeff_matrix = np.zeros((n_total, steps + 1), dtype=npfloat)  # array to store everything
    final_coeff_matrix[:, 0] = initial_pos_vel          # initialize with intial position velocity
    oip1 = one / (one + np.arange(order_max))
    orders_used = np.zeros(steps + 1, dtype=np.int32)
    
    #Labeling indices to visually track more easily
    x, y, z = 0, 1, 2
    vx, vy, vz = 3, 4, 5
    
    for j in range(1, steps + 1):
        c = np.zeros((n_total, order_max + 1), dtype=npfloat)    # array storage for loop
        c[:, 0] = final_coeff_matrix[:, j - 1]
        power = timedelta
        sum_terms = np.zeros(n_total, dtype=npfloat)
        max_contrib = tol + one
        i = 0

        while max_contrib > tol and i < order_max:
            # Position derivatives from velocity
            c[x, i+1] = oip1[i] * c[vx,i]
            c[y, i+1] = oip1[i] * c[vy,i]
            c[z, i+1] = oip1[i] * c[vz,i]

            # Velocity derivatives from lorentz force
            c[vx, i+1] = oip1[i]*qoverm*(Bfield[2]*c[vy,i]-Bfield[1]*c[vz,i])
            c[vy, i+1] = oip1[i]*qoverm*(Bfield[0]*c[vz,i]-Bfield[2]*c[vx,i])
            c[vz, i+1] = oip1[i]*qoverm*(Bfield[1]*c[vx,i]-Bfield[0]*c[vy,i])

            sum_terms += c[:, i+1]* power # just keeps adding these on until PS prder is reached, final added to permanent matrix below
            max_contrib = np.abs(c[:, i+1]).max()
            power *= timedelta
            i += 1

        final_coeff_matrix[:, j] = final_coeff_matrix[:, j - 1] + sum_terms        
        orders_used[j] = i

    return final_coeff_matrix, orders_used

def analytical_constantB(t, d, Bfield, qoverm):
    x0, y0, z0, vx0, vy0, vz0 = d
    omega = qoverm * Bfield[2]  # Cyclotron frequency
    
    sin_ot = np.sin(omega * t)
    cos_ot = np.cos(omega * t)
    
    x_t = x0 + (vy0 / omega) * (1 - cos_ot) + (vx0 / omega) * sin_ot
    y_t = y0 - (vx0 / omega) * (1 - cos_ot) + (vy0 / omega) * sin_ot
    z_t = z0 + vz0 * t

    vx_t = vx0 * cos_ot - vy0 * sin_ot
    vy_t = vy0 * cos_ot + vx0 * sin_ot
    vz_t = vz0 * np.ones_like(t)

    return np.vstack((x_t, y_t, z_t, vx_t, vy_t, vz_t))

@maybe_njit
def lorentz_force_constB(t, d, Bfield, qoverm):
    x, y, z, vx, vy, vz = d  
    dvx = qoverm * (vy * Bfield[2] - vz * Bfield[1])
    dvy = qoverm * (vz * Bfield[0] - vx * Bfield[2])
    dvz = qoverm * (vx * Bfield[1] - vy * Bfield[0])

    return np.array([vx, vy, vz, dvx, dvy, dvz])

def vector_error(num_sol, ana_sol, is_rk45=False):
    # extract numerical trajectory
    eps = 1e-15
    if is_rk45:
        x, y, z = num_sol.y[0], num_sol.y[1], num_sol.y[2]
    else:
        x, y, z = num_sol[0], num_sol[1], num_sol[2]
    # analytical trajectory
    xa, ya, za = ana_sol[0], ana_sol[1], ana_sol[2]

    abs_err = np.sqrt((x - xa)**2 + (y - ya)**2 + (z - za)**2)

    r_ana = np.sqrt(xa**2 + ya**2 + za**2)
    rel_err = abs_err / np.maximum(r_ana, eps)
    return abs_err, rel_err