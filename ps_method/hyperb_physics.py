"""
hyperb_physics.py — Physics functions for charged particle motion in a
                   hyperbolic-tangent magnetic field.

Three functions are provided:
  • cauchy_sum    – Cauchy product helper for power-series coefficient multiplication.
  • ps_integrate  – Power-series integrator with adaptive truncation order.
  • lorentz_force – RHS function for scipy / RK4 integrators.

All functions are compiled with @maybe_njit (skipped when float128 is active).
"""

import numpy as np
from . import utils as ul


@ul.maybe_njit
def cauchy_sum(a, b, n):
    """Cauchy product: n-th coefficient of the product of two power series."""
    result = ul.npfloat(0.0)
    for j in range(n + 1):
        result += a[j] * b[n - j]
    return result


@ul.maybe_njit
def ps_integrate(PS_order, steps_ps, initial_pos_vel, timedelta, gamma, charge_sign, tol):
    """Advance a charged particle through B = ẑ tanh(γy) using power-series method.

    The state vector has 9 components: the 6 physical variables [x, y, z, vx,
    vy, vz] plus 3 auxiliary series [sinh(γy), cosh(γy), Bz = tanh(γy)] needed
    to express the nonlinear field as a power series.

    After each step, sinh, cosh, and Bz are tethered (recomputed from exact
    functions of the updated y position, see paper) to prevent auxiliary drift.

    Returns
    -------
    state_history : (9, steps_ps+1) array — trajectory + auxiliaries at each step.
    orders_used   : (steps_ps+1,) int array — PS order used per step.
    """
    n_total = 9
    state_history = np.zeros((n_total, steps_ps + 1), dtype=ul.npfloat)

    # Named indices for readability
    x, y, z = 0, 1, 2
    vx, vy, vz = 3, 4, 5
    sinh_aux, cosh_aux = 6, 7
    Bz_aux = 8

    # Initial conditions: physical state + auxiliary evaluations at y0
    state_history[0:6, 0] = initial_pos_vel
    y0 = initial_pos_vel[1]

    state_history[sinh_aux, 0] = np.sinh(gamma * y0)
    state_history[cosh_aux, 0] = np.cosh(gamma * y0)
    state_history[Bz_aux, 0]   = np.tanh(gamma * y0)

    Bz_series   = np.zeros(PS_order, dtype=ul.npfloat)
    orders_used = np.zeros(steps_ps + 1, dtype=np.int32)
    _one        = ul.npfloat(1.0)
    oip1        = _one / (_one + np.arange(PS_order))   # 1/(i+1) lookup table

    c = np.zeros((n_total, PS_order + 1), dtype=ul.npfloat)
    sum_terms = np.zeros(n_total, dtype=ul.npfloat)

    for j in range(1, steps_ps + 1):
        c[:, 0] = state_history[:, j - 1]              # seed with end of previous step
        sum_terms[:] = 0                               # reset accumulator
        power     = timedelta
        i         = 0
        max_contrib = tol + ul.npfloat(1.0)

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

            # Position coefficients
            c[x, i+1]  = oip1[i] * c[vx, i]
            c[y, i+1]  = oip1[i] * c[vy, i]
            c[z, i+1]  = oip1[i] * c[vz, i]

            # Velocity coefficients
            c[vx, i+1] =  charge_sign * oip1[i] * vyBz
            c[vy, i+1] = -charge_sign * oip1[i] * vxBz
            c[vz, i+1] = 0.0

            # Auxiliary coefficients
            c[sinh_aux, i+1] = oip1[i] * gamma * cauchy_sum(c[cosh_aux], c[vy], i)
            c[cosh_aux, i+1] = oip1[i] * gamma * cauchy_sum(c[sinh_aux], c[vy], i)
            c[Bz_aux,   i+1] = Bz_series[i]

            if not np.isfinite(c[:, i+1]).all():
                break

            new_term = c[:, i+1] * power
            sum_terms += new_term
            # Convergence test: per-component relative test — stop when
            # |new_term[k]| < |sum_terms[k]| * tol for every component.
            # Components with |sum_terms[k]| ≤ tol are treated as already
            # converged (avoids divide-by-zero on inactive axes).
            max_contrib = ul.npfloat(0.0)
            for k in range(n_total):
                ref = abs(sum_terms[k])
                if ref > tol:
                    ratio = abs(new_term[k]) / ref
                    if ratio > max_contrib:
                        max_contrib = ratio
            # Old (paper version):
            # max_contrib = np.abs(c[:, i+1]).max()
            power *= timedelta
            i += 1

        state_history[:, j] = state_history[:, j - 1] + sum_terms
        orders_used[j] = i

        # --- Tethering: recompute auxiliaries from exact functions to minimize drift ---
        y_now = state_history[y, j]
        sinh_now = np.sinh(gamma * y_now)
        cosh_now = np.cosh(gamma * y_now)
        Bz_now = sinh_now / cosh_now

        state_history[sinh_aux, j] = sinh_now
        state_history[cosh_aux, j] = cosh_now
        state_history[Bz_aux, j] = Bz_now

    return state_history, orders_used

@ul.maybe_njit
def lorentz_force(t, d, gamma, charge_sign):
    """Right-hand side for the Lorentz equation in a tanh magnetic field.

    B = ẑ tanh(γy), so the force is (q/m)(v × B):
        dvx/dt =  (q/m) vy Bz
        dvy/dt = -(q/m) vx Bz
        dvz/dt =  0

    Returns d/dt [x, y, z, vx, vy, vz].
    Used as the RHS callback for scipy.integrate.solve_ivp (RK45) and rk4_fixed_step.
    """
    # t is required by the solver's RHS call signature (solve_ivp /
    # rk4_fixed_step); unused here.
    d = d.astype(ul.npfloat)
    gamma = ul.npfloat(gamma)

    x, y, z, vx, vy, vz = d

    Bz = np.tanh(gamma * y)
    ax =  charge_sign * vy * Bz
    ay = -charge_sign * vx * Bz
    az = ul.npfloat(0.0)

    return np.array([vx, vy, vz, ax, ay, az], dtype=ul.npfloat)

