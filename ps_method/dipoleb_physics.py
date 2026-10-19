"""
Physics kernels for charged-particle motion in a magnetic dipole (dipoleb).

Solvers (all numba-compiled via maybe_njit):
    lorentz_force_inplace        — dipole Lorentz force, dy/dt written into a buffer (RK4 / RK45 kernels)
    ps_integrate                 — power series integrator over a block of fixed steps
    hamiltonian_rhs_inplace      — Hamilton's equations in canonical (q, p) coordinates (RKG kernel)
    _rkgl4_step_fp_inplace       — one 2-stage Gauss–Legendre step, fixed-point stage solver
    rkgl4_hamiltonian            — symplectic RKG integration loop

Vector potential:
    vector_potential             — A = (y, -x, 0)/r^3 for the canonical momentum

Streaming:
    run_ps_streaming_with_decimation — fixed-step PS run streamed to h5 in chunks
"""

import numpy as np
import h5py
import time
from . import utils as ul
from . import writers as wr

one = ul.npfloat(1.0)
two = ul.npfloat(2.0)
three = ul.npfloat(3.0)
five = ul.npfloat(5.0)
twopointfive = ul.npfloat(2.5)

@ul.maybe_njit
def lorentz_force_inplace(d, charge_sign, out):
    """Dipole Lorentz force: writes dy/dt = (v, charge_sign * v x B) into ``out``.

    B = -(3xz, 3yz, 3z^2 - r^2) / r^5 in normalised units. In-place so the RK
    kernels allocate nothing per evaluation.
    """
    x, y, z, vx, vy, vz = d[0], d[1], d[2], d[3], d[4], d[5]
    r2 = x**two + y**two + z**two
    r5inv = r2**(-twopointfive) if r2 != 0 else ul.npfloat(0.0)

    Bx = -three * x * z * r5inv
    By = -three * y * z * r5inv
    Bz = -(three * z**two - r2) * r5inv

    out[0] = vx
    out[1] = vy
    out[2] = vz
    out[3] = charge_sign * (vy * Bz - vz * By)
    out[4] = charge_sign * (vz * Bx - vx * Bz)
    out[5] = charge_sign * (vx * By - vy * Bx)


@ul.maybe_njit
def ps_integrate(ps_order, steps_ps, initial_pos_vel, tol, charge_sign, timedelta,
                 warn=True, horner=False):
    """Integrate ``steps_ps`` fixed steps of size ``timedelta`` by power series.

    Each step builds Taylor coefficients up to ``ps_order`` (stopping early
    once the largest relative contribution drops below ``tol``) for the 6
    state rows plus 11 auxiliary rows (r^2, r^-5, and the products that make
    the B-field recurrence polynomial).

    Returns (state_history (17, steps_ps + 1), orders_used (steps_ps + 1,)).
    """
    n_total = 17
    state_history = np.zeros((n_total, steps_ps + 1), dtype=ul.npfloat)

    # row indices of the 17-row PS state
    x, y, z, vx, vy, vz = 0, 1, 2, 3, 4, 5
    r2_aux, a_aux, b_aux, c_aux, d_aux, e_aux, f_aux, g_aux = 6, 7, 8, 9, 10, 11, 12, 13
    Bx_aux, By_aux, Bz_aux = 14, 15, 16

    # set up initial dynamic variables
    state_history[0:6, 0] = initial_pos_vel
    x0, y0, z0 = initial_pos_vel[0], initial_pos_vel[1], initial_pos_vel[2]
    vx0, vy0, vz0 = initial_pos_vel[3], initial_pos_vel[4], initial_pos_vel[5]

    # set up initial aux variables
    r2_0 = x0**two + y0**two + z0**two
    a0 = r2_0**(-twopointfive)
    b0 = two * z0**two - x0**two - y0**two
    c0 = y0 * z0
    d0 = x0 * z0
    e0 = -(b0 * vy0 - three * c0 * vz0)
    f0 = -(three * d0 * vz0 - b0 * vx0)
    g0 = -(three * c0 * vx0 - three * d0 * vy0)

    state_history[r2_aux, 0] = r2_0
    state_history[a_aux, 0] = a0
    state_history[b_aux, 0] = b0
    state_history[c_aux, 0] = c0
    state_history[d_aux, 0] = d0
    state_history[e_aux, 0] = e0
    state_history[f_aux, 0] = f0
    state_history[g_aux, 0] = g0
    state_history[Bz_aux, 0] = -a0 * b0
    state_history[By_aux, 0] = -three * a0 * c0
    state_history[Bx_aux, 0] = -three * a0 * d0

    oip1 = one / (one + np.arange(ps_order, dtype=ul.npfloat))
    orders_used = np.zeros(steps_ps + 1, dtype=np.int32)
    n_unconverged = 0

    def cauchy_sum_inline(a, b, n):
        result = 0.0
        for jj in range(n + 1):
            result += a[jj] * b[n - jj]
        return result

    c = np.zeros((n_total, ps_order + 1), dtype=ul.npfloat)
    sum_terms = np.zeros(n_total, dtype=ul.npfloat)
    zeta = np.zeros(ps_order + 1, dtype=ul.npfloat)

    for j in range(1, steps_ps + 1):
        c[:, 0] = state_history[:, j - 1]
        sum_terms[:] = 0

        # base division coefficient for this step's coefficients
        zeta[0] = c[a_aux, 0] / c[r2_aux, 0]

        power = timedelta
        max_contrib = tol + one
        i = 0

        while max_contrib > tol and i < ps_order:
            c[x, i+1]  = c[vx, i] * oip1[i]
            c[y, i+1]  = c[vy, i] * oip1[i]
            c[z, i+1]  = c[vz, i] * oip1[i]
            c[vx, i+1] = charge_sign * cauchy_sum_inline(c[a_aux], c[e_aux], i) * oip1[i]
            c[vy, i+1] = charge_sign * cauchy_sum_inline(c[a_aux], c[f_aux], i) * oip1[i]
            c[vz, i+1] = charge_sign * cauchy_sum_inline(c[a_aux], c[g_aux], i) * oip1[i]

            # self-convolutions shared by r2 and b; operation order matches the
            # original inline expressions so results are bit-identical.
            xx = cauchy_sum_inline(c[x], c[x], i+1)
            yy = cauchy_sum_inline(c[y], c[y], i+1)
            zz = cauchy_sum_inline(c[z], c[z], i+1)
            c[r2_aux, i+1] = xx + yy + zz

            # Series division a = zeta * r2: only the new zeta[i] is needed,
            # earlier entries persist across orders.
            if i >= 1:
                acc = c[a_aux, i]
                for jj in range(1, i + 1):
                    acc -= c[r2_aux, jj] * zeta[i - jj]
                zeta[i] = acc / c[r2_aux, 0]

            a_prime = 0.0
            for k in range(i + 1):
                a_prime += (i - k + 1) * zeta[k] * c[r2_aux, i - k + 1]
            c[a_aux, i+1] = - (five / (two * (i + 1))) * a_prime

            c[b_aux, i+1] = two * zz - xx - yy
            c[c_aux, i+1] = cauchy_sum_inline(c[y], c[z], i+1)
            c[d_aux, i+1] = cauchy_sum_inline(c[x], c[z], i+1)

            c[e_aux, i+1] = -(cauchy_sum_inline(c[b_aux], c[vy], i+1) - three * cauchy_sum_inline(c[c_aux], c[vz], i+1))
            c[f_aux, i+1] = -(three * cauchy_sum_inline(c[d_aux], c[vz], i+1) - cauchy_sum_inline(c[b_aux], c[vx], i+1))
            c[g_aux, i+1] = -(three * (cauchy_sum_inline(c[c_aux], c[vx], i+1) - cauchy_sum_inline(c[d_aux], c[vy], i+1)))

            # Bail out if the r^-5 series blows up.
            if not np.isfinite(c[a_aux, i+1]):
                break

            # Accumulate the increment; convergence is judged on the 6 state rows.
            max_contrib = ul.npfloat(0.0)
            for k in range(6):
                nt = c[k, i+1] * power
                sum_terms[k] += nt
                ref = abs(sum_terms[k])
                ant = abs(nt)
                if ref > tol:
                    ratio = ant / ref
                elif ant > tol:
                    ratio = ant / tol
                else:
                    ratio = ul.npfloat(0.0)
                if ratio > max_contrib:
                    max_contrib = ratio

            power *= timedelta
            i += 1

        if horner:
            # Re-evaluate the converged increment with Horner's rule (highest
            # order first, numpy.polyval's scheme). Same coefficients and the
            # same convergence decision as the running sum above; only the
            # summation order changes. Combining the small high-order terms
            # first avoids re-rounding the dominant c_1*dt term once per order,
            # which lowers the round-off floor of the energy error (~10x at
            # 65 steps/gyro, no gain at ~20). Not bit-identical to horner=False.
            for k in range(6):
                acc = ul.npfloat(0.0)
                for q in range(i, 0, -1):
                    acc = acc * timedelta + c[k, q]
                sum_terms[k] = acc * timedelta

        state_history[0:6, j] = state_history[0:6, j - 1] + sum_terms[0:6]

        x_now, y_now, z_now = state_history[x, j], state_history[y, j], state_history[z, j]
        vx_now, vy_now, vz_now = state_history[vx, j], state_history[vy, j], state_history[vz, j]

        r2_now = x_now**two + y_now**two + z_now**two
        a_now = r2_now**(-twopointfive)
        b_now = two * z_now**two - x_now**two - y_now**two
        c_now = y_now * z_now
        d_now = x_now * z_now
        e_now = -(b_now * vy_now - three * c_now * vz_now)
        f_now = -(three * d_now * vz_now - b_now * vx_now)
        g_now = -(three * c_now * vx_now - three * d_now * vy_now)

        state_history[r2_aux, j] = r2_now
        state_history[a_aux, j] = a_now
        state_history[b_aux, j] = b_now
        state_history[c_aux, j] = c_now
        state_history[d_aux, j] = d_now
        state_history[e_aux, j] = e_now
        state_history[f_aux, j] = f_now
        state_history[g_aux, j] = g_now
        state_history[Bx_aux, j] = -three * a_now * d_now
        state_history[By_aux, j] = -three * a_now * c_now
        state_history[Bz_aux, j] = -a_now * b_now

        orders_used[j] = i
        if max_contrib > tol:
            n_unconverged += 1

    # In fixed-step runs unconverged steps enter the output, so this is a real
    # accuracy alarm. Adaptive probe calls pass warn=False: their unconverged
    # steps are rejected by the quality mask (orders >= cap) and never reach
    # the output — rejections are reported in the adaptive summary instead.
    if warn and n_unconverged > 0:
        print("  [dipoleb ps_integrate] WARNING:", n_unconverged, "/", steps_ps,
              "steps hit ps_order=", ps_order, "without reaching tol=", tol)

    return state_history, orders_used

# ========================
# ==== RKG Functions ====
# ========================

@ul.maybe_njit
def vector_potential(r):
    """A = (y, -x, 0) / r^3, the normalised dipole vector potential."""
    x, y, z = r
    r2 = x**2 + y**2 + z**2
    r3 = r2 * np.sqrt(r2)

    if r3 == 0:
        return np.zeros(3, dtype=ul.npfloat)

    Ax = y / r3
    Ay = - x / r3
    Az = 0.0

    return np.array([Ax, Ay, Az], dtype=ul.npfloat)

@ul.maybe_njit
def hamiltonian_rhs_inplace(d, charge_sign, out):
    """Hamilton's equations for the dipole in canonical coordinates; writes into ``out``.

    d = (x, y, z, px, py, pz) with p = v + charge_sign * A and A = (y, -x, 0)/r^3.
    """
    x, y, z = d[0], d[1], d[2]
    px, py, pz = d[3], d[4], d[5]

    r2 = x*x + y*y + z*z
    r = np.sqrt(r2)
    r3 = r2 * r
    r5 = r2 * r3

    if r5 == 0:
        for i in range(6):
            out[i] = 0.0
        return

    Ax = y / r3
    Ay = -x / r3

    Pix = px - charge_sign * Ax
    Piy = py - charge_sign * Ay
    Piz = pz

    out[0] = Pix
    out[1] = Piy
    out[2] = Piz
    out[3] = charge_sign * (
        -three * x * y / r5 * Pix
        - (one / r3 - three * x * x / r5) * Piy
    )
    out[4] = charge_sign * (
        (one / r3 - three * y * y / r5) * Pix
        + three * x * y / r5 * Piy
    )
    out[5] = charge_sign * three * z / r5 * (-y * Pix + x * Piy)

@ul.maybe_njit
def _rkgl4_step_fp_inplace(func, y0, dt, charge_sign, max_iter, tol,
                           Y1, Y2, K0, K1, K0n, K1n, out):
    """One 2-stage Gauss–Legendre step (order 4) by fixed-point iteration on
    the stage derivatives, believed to match Yugo & Iyemori (2007) with the
    Calvo et al. (2003) explicit-Euler starting guess.

    ``func`` writes in place and the stage buffers are passed in, so nothing
    is allocated per step. Writes the new state into ``out``; returns
    (converged, iters).
    """
    sqrt3 = np.sqrt(3.0)
    a11, a12 = 0.25, 0.25 - sqrt3 / 6.0
    a21, a22 = 0.25 + sqrt3 / 6.0, 0.25
    b1 = b2 = 0.5
    dim = y0.shape[0]

    # Initial guess from explicit Euler (K1 = copy of K0)
    func(y0, charge_sign, K0)
    for d in range(dim):
        K1[d] = K0[d]

    converged = False
    iters = 0
    for n in range(max_iter):
        for d in range(dim):
            Y1[d] = y0[d] + dt * (a11 * K0[d] + a12 * K1[d])
            Y2[d] = y0[d] + dt * (a21 * K0[d] + a22 * K1[d])

        func(Y1, charge_sign, K0n)
        func(Y2, charge_sign, K1n)

        diff = 0.0
        for d in range(dim):
            c0 = abs(K0n[d] - K0[d])
            c1 = abs(K1n[d] - K1[d])
            if c0 > diff:
                diff = c0
            if c1 > diff:
                diff = c1

        # K0 <- K0n, K1 <- K1n (swap buffers instead of copying)
        K0, K0n = K0n, K0
        K1, K1n = K1n, K1
        iters = n + 1
        if diff < tol:
            converged = True
            break

    for d in range(dim):
        out[d] = y0[d] + dt * (b1 * K0[d] + b2 * K1[d])
    return converged, iters


@ul.maybe_njit
def rkgl4_hamiltonian(func, y0, dt, steps, charge_sign):
    """Symplectic RKG (Gauss–Legendre s=2) integration loop.

    ``func`` is an in-place right-hand side, func(y, charge_sign, out)
    (hamiltonian_rhs_inplace). Stage solve: fixed-point, tol 1e-15, cap 100.

    Returns (trajectory (steps + 1, dim), n_failed, avg_iters, max_iters):
        n_failed   steps that hit the sweep cap without converging
        avg_iters  mean fixed-point sweeps per step
        max_iters  worst-case sweeps over the run
    """
    dim = y0.shape[0]
    d_out = np.zeros((steps + 1, dim), dtype=ul.npfloat)
    d_out[0] = y0
    n_failed = 0
    total_iters = 0
    max_iters = 0
    Y1 = np.empty(dim, dtype=ul.npfloat); Y2 = np.empty(dim, dtype=ul.npfloat)
    K0 = np.empty(dim, dtype=ul.npfloat); K1 = np.empty(dim, dtype=ul.npfloat)
    K0n = np.empty(dim, dtype=ul.npfloat); K1n = np.empty(dim, dtype=ul.npfloat)

    for i in range(1, steps + 1):
        converged, iters = _rkgl4_step_fp_inplace(
            func, d_out[i - 1], dt, charge_sign, 100, 1e-15,
            Y1, Y2, K0, K1, K0n, K1n, d_out[i])
        if not converged:
            n_failed += 1
        total_iters += iters
        if iters > max_iters:
            max_iters = iters

    avg_iters = total_iters / steps
    return d_out, n_failed, avg_iters, max_iters

# ===================================
# === Decimate/Chunking Functions ===
# ===================================
def run_ps_streaming_with_decimation(
    initial_pos_vel_ps,
    steps_ps,
    ps_step,
    ps_order,
    tol,
    charge_sign,
    e0_ps,
    mu0_ps,
    cache_path,
    write_data,
    chunk_steps,
    decimate,
    n_steps_per_gyro_ps,
    user_min_phase,
    dragt_monitor=None,
    r_atmosphere=1.0,
    global_index_start=0,
    total_steps=None,
    segment_index=None,
    horner=False,
):
    """Fixed-step PS run streamed to ``cache_path`` in chunks of ``chunk_steps``.

    Keeps every ``decimate``-th global step (plus the last). In a segmented
    run this integrates one segment: ``global_index_start`` is its first
    global step and ``total_steps`` the whole run's count. Returns
    (max_ps, t_integ), where t_integ is kernel time only.
    """
    start_time_ps = time.time()
    t_integ = 0.0      # time spent inside the PS kernel only (no h5 write, monitors, or JIT)

    if total_steps is None:
        total_steps = steps_ps

    cur_state = initial_pos_vel_ps.copy()
    remaining = steps_ps
    global_index = global_index_start
    max_ps = 0
    sum_orders = 0      # for mean over kept (output-grid) orders
    count_orders = 0
    hit_atmosphere = False
    hit_atm_step   = -1
    hit_atm_r      = 0.0
    # r_atmosphere is in R_E; configurable via yaml (default 1.0 = surface)

    if write_data:
        f = h5py.File(cache_path, "w")
        ps_grp = f.create_group("ps")
        ps_grp.attrs["ordercap"] = ps_order
        ps_grp.attrs["numberstepspergyro"] = int(n_steps_per_gyro_ps)
        ps_grp.attrs["dt"]        = ul.npfloat(ps_step)
        ps_grp.attrs["steps"]    = int(steps_ps)
        ps_grp.attrs["streaming"] = True
        ps_grp.attrs["chunksize"]= int(chunk_steps)
        ps_grp.attrs["decimate"] = int(decimate)
        ps_grp.attrs["horner"]   = bool(horner)
        ps_grp.attrs["tol"] = ul.npfloat(tol)
        ps_grp.attrs["minphase"] = ul.npfloat(user_min_phase)
        ps_grp.attrs["E0"]       = float(e0_ps)
        ps_grp.attrs["mu0"]      = float(mu0_ps)
        # t0 is derived from the global step clock: dt * first-step index.
        # For a single run global_index_start == 0, so this stays 0.0.
        ps_grp.attrs["t0"]       = ul.npfloat(ps_step) * np.int64(global_index_start)
        # Row layout: [x,y,z, vx,vy,vz, Bx,By,Bz]
        ps_grp.attrs["save_rows"] = "pos_vel_B"
        ps_grp.attrs["n_save"]    = wr.n_save

        dset_y = ps_grp.create_dataset(
            "y",
            shape=(wr.n_save, 0),
            maxshape=(wr.n_save, None),
            dtype=ul.npfloat,
            chunks=(wr.n_save, min(chunk_steps, steps_ps + 1)),
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

            _t0 = time.perf_counter()
            sol_chunk, orders_chunk = ps_integrate(
                ps_order, this_chunk, cur_state, tol, charge_sign, ps_step,
                horner=horner,
            )
            t_integ += time.perf_counter() - _t0

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
                keep = (idx_eff % decimate == 0) | (idx_eff == total_steps)

            sol_keep = sol_eff[:, keep]
            orders_keep = orders_eff[keep]

            if sol_keep.shape[1] > 0:
                max_ps = max(max_ps, int(orders_keep.max()))
                sum_orders   += int(orders_keep.sum())
                count_orders += orders_keep.size

                if write_data:
                    old_len = dset_y.shape[1]
                    new_len = old_len + sol_keep.shape[1]

                    dset_y.resize((wr.n_save, new_len))
                    dset_orders.resize((new_len,))

                    dset_y[:, old_len:new_len] = sol_keep[wr.SAVE_ROWS, :]
                    dset_orders[old_len:new_len] = orders_keep

            # ---- atmospheric impact check (diagnostic only, does not halt) ----
            r_sq = sol_chunk[0]**2 + sol_chunk[1]**2 + sol_chunk[2]**2
            below = np.where(r_sq < r_atmosphere**2)[0]
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
            ps_grp.attrs["mean_ps"] = (sum_orders / count_orders) if count_orders > 0 else 0.0
            ps_grp.attrs["integ_time"] = float(t_integ)                      # kernel only
            ps_grp.attrs["wall_time_incl_h5_write"] = float(time.time() - start_time_ps)  # NOT an integrator timing
            # sum/count kept so the VDS can aggregate a true cross-segment mean.
            ps_grp.attrs["sum_orders"]   = int(sum_orders)
            ps_grp.attrs["count_orders"] = int(count_orders)
            ps_grp.attrs["hit_atmosphere"] = hit_atmosphere
            ps_grp.attrs["hit_atm_step"]   = hit_atm_step
            ps_grp.attrs["hit_atm_r"]      = hit_atm_r

            # --- checkpoint handoff -------------------------------------------
            # The only thing needed to continue the (autonomous) trajectory is
            # the boundary state; the global step index is the exact clock.
            # start/end states are stored as full-precision datasets (not attrs)
            # so a restarted segment continues bit-for-bit.
            ps_grp.attrs["start_global_index"] = int(global_index_start)
            ps_grp.attrs["end_global_index"]   = int(global_index)
            ps_grp.attrs["total_steps"]        = int(total_steps)
            if segment_index is not None:
                ps_grp.attrs["segment_index"]  = int(segment_index)
            ps_grp.create_dataset(
                "start_state", data=np.asarray(initial_pos_vel_ps, dtype=ul.npfloat))
            ps_grp.create_dataset(
                "end_state", data=np.asarray(cur_state, dtype=ul.npfloat))

        if hit_atmosphere:
            print(f"\n    *** ATMOSPHERE FLAG: particle crossed r < {r_atmosphere} R_E "
                  f"at step {hit_atm_step:,} (r_min = {hit_atm_r:.4f} R_E) ***\n")

        # Returned time is INTEGRATION ONLY (kernel calls), so it is comparable
        # with the RK timers, which never include output I/O. Wall time is in
        # the h5 attrs as wall_time_incl_h5_write.
        return max_ps, t_integ

    finally:
        if f is not None:
            f.close()
