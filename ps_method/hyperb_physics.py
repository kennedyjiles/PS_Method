"""
hyperb_physics.py — Physics kernels for charged particle motion in a
                   hyperbolic-tangent magnetic field.

Two solvers are provided:
  • ps_integrate  – Power-series integrator with adaptive truncationorder.
  • lorentz_force – RHS function for scipy / RK4 integrators.

Because Bz = tanh(γy) is nonlinear, the PS method must track auxiliary
series for sinh(γy) and cosh(γy) alongside the physical state.  The Taylor
coefficients of Bz are recovered at each order via a division recurrence
(Cauchy product inversion of cosh into sinh).  After each full step the
auxiliary variables are "tethered" — recomputed from the exact functions —
to prevent drift.

All functions are compiled with @maybe_njit (skipped when float128 is active).
"""

import numpy as np
from .utils import maybe_njit, npfloat


@maybe_njit
def cauchy_sum(a, b, n):
    """Cauchy product: n-th coefficient of the product of two power series."""
    result = 0.0
    for j in range(n + 1):
        result += a[j] * b[n - j]
    return result


@maybe_njit
def ps_integrate(PS_order, steps_ps, initial_pos_vel, timedelta, gamma, qoverm, tol):
    """Advance a charged particle through B = ẑ tanh(γy) using power-series method.

    The state vector has 9 components: the 6 physical variables [x, y, z, vx,
    vy, vz] plus 3 auxiliary series [sinh(γy), cosh(γy), Bz = tanh(γy)] needed
    to express the nonlinear field as a Taylor series.

    After each step, sinh, cosh, and Bz are tethered (recomputed from exact
    functions of the updated y position, see paper) to prevent auxiliary drift.

    Returns
    -------
    final_coeff_matrix : (9, steps_ps+1) array — trajectory + auxiliaries.
    orders_used        : (steps_ps+1,) int array — PS order used per step.
    """
    n_total = 9
    final_coeff_matrix = np.zeros((n_total, steps_ps + 1), dtype=npfloat)

    # Named indices for readability
    x, y, z = 0, 1, 2
    vx, vy, vz = 3, 4, 5
    sinh_aux, cosh_aux = 6, 7
    Bz_aux = 8

    # Initial conditions: physical state + auxiliary evaluations at y0
    final_coeff_matrix[0:6, 0] = initial_pos_vel
    y0 = initial_pos_vel[1]

    final_coeff_matrix[sinh_aux, 0] = np.sinh(gamma * y0)
    final_coeff_matrix[cosh_aux, 0] = np.cosh(gamma * y0)
    final_coeff_matrix[Bz_aux, 0]   = np.tanh(gamma * y0)

    Bz_series   = np.zeros(PS_order, dtype=npfloat)
    orders_used = np.zeros(steps_ps + 1, dtype=np.int32)
    oip1        = 1.0 / (1.0 + np.arange(PS_order))    # 1/(i+1) lookup table

    for j in range(1, steps_ps + 1):
        c = np.zeros((n_total, PS_order + 1), dtype=npfloat)
        c[:, 0] = final_coeff_matrix[:, j - 1]         # seed with end of previous step

        sum_terms = np.zeros(n_total, dtype=npfloat)
        power     = timedelta
        i         = 0
        max_contrib = tol + npfloat(1.0)

        while max_contrib > tol and i < PS_order:

            # --- Bz coefficient via division recurrence ---
            if i == 0:
                Bz_series[0] = c[sinh_aux, 0] / c[cosh_aux, 0]
            else:
                s = c[sinh_aux, i]
                for k in range(1, i + 1):
                    s -= c[cosh_aux, k] * Bz_series[i - k]
                Bz_series[i] = s / c[cosh_aux, 0]

            # Velocity–field products via Cauchy (discrete convolution)
            vyBz = cauchy_sum(c[vy], Bz_series, i)
            vxBz = cauchy_sum(c[vx], Bz_series, i)

            # Position coefficients: dx/dt = v  →  x_{n+1} = v_n / (n+1)
            c[x, i+1]  = oip1[i] * c[vx, i]
            c[y, i+1]  = oip1[i] * c[vy, i]
            c[z, i+1]  = oip1[i] * c[vz, i]

            # Velocity coefficients: dv/dt = (q/m)(v × B)
            c[vx, i+1] =  qoverm * oip1[i] * vyBz
            c[vy, i+1] = -qoverm * oip1[i] * vxBz
            c[vz, i+1] = 0.0

            # Auxiliary coefficients: d/dt sinh(γy) = γ cosh(γy) vy  (chain rule)
            c[sinh_aux, i+1] = oip1[i] * gamma * cauchy_sum(c[cosh_aux], c[vy], i)
            c[cosh_aux, i+1] = oip1[i] * gamma * cauchy_sum(c[sinh_aux], c[vy], i)
            c[Bz_aux,   i+1] = Bz_series[i]

            if not np.isfinite(c[:, i+1]).all():
                break

            sum_terms += c[:, i+1] * power             
            max_contrib = np.abs(c[:, i+1]).max()      
            power *= timedelta
            i += 1

        final_coeff_matrix[:, j] = final_coeff_matrix[:, j - 1] + sum_terms
        orders_used[j] = i

        # --- Tethering: recompute auxiliaries from exact functions to prevent drift ---
        y_now = final_coeff_matrix[y, j]
        sinh_now = np.sinh(gamma * y_now)
        cosh_now = np.cosh(gamma * y_now)
        Bz_now = sinh_now / cosh_now

        final_coeff_matrix[sinh_aux, j] = sinh_now
        final_coeff_matrix[cosh_aux, j] = cosh_now
        final_coeff_matrix[Bz_aux, j] = Bz_now

    return final_coeff_matrix, orders_used

@maybe_njit
def lorentz_force(t, y, gamma, qoverm):
    """Right-hand side for the Lorentz equation in a tanh magnetic field.

    B = ẑ tanh(γy), so the force is (q/m)(v × B):
        dvx/dt =  (q/m) vy Bz
        dvy/dt = -(q/m) vx Bz
        dvz/dt =  0

    Returns d/dt [x, y, z, vx, vy, vz].
    Used as the RHS callback for scipy.integrate.solve_ivp (RK45) and rk4_fixed_step.
    """
    y = y.astype(npfloat)
    gamma = npfloat(gamma)

    Bz = np.tanh(gamma * y[1])
    ax = qoverm * y[4] * Bz
    ay = -qoverm * y[3] * Bz
    az = npfloat(0.0)

    return np.array([y[3], y[4], y[5], ax, ay, az], dtype=npfloat)




