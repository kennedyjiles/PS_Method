"""
Adaptive Power Series stepping for dipole B field.
Hybrid approach: uses the original fast PS_dipoleB for easy chunks,
falls back to per-step adaptive mode only when the series needs it.

All other functions are imported unchanged from functions_library_dipole.
"""

import numpy as np
import h5py
import time
import warnings
from functions.functions_library_dipole import *          # everything passes through
from functions.functions_library_universal_chunk import npfloat

# ---- constants (match original) ----
_one   = npfloat(1.0)
_two   = npfloat(2.0)
_three = npfloat(3.0)
_five  = npfloat(5.0)
_two5  = npfloat(2.5)


# ===========================================================
#   Module-level Cauchy helpers (defined once at import time)
# ===========================================================
def _cauchy_sum(a, b, n):
    s = 0.0
    for j in range(n + 1):
        s += a[j] * b[n - j]
    return s

def _cauchy_divide(a, b, out, n):
    out[0] = a[0] / b[0]
    for i in range(1, n + 1):
        acc = a[i]
        for j in range(1, i + 1):
            acc -= b[j] * out[i - j]
        out[i] = acc / b[0]


# ===========================================================
#                  Helper: tether aux vars
# ===========================================================
def _tether_aux(state):
    """Recompute auxiliary variables [6:17] from position/velocity [0:6] in place."""
    xv, yv, zv   = state[0], state[1], state[2]
    vxv, vyv, vzv = state[3], state[4], state[5]

    r2 = xv**_two + yv**_two + zv**_two
    a  = r2**(-_two5)
    b  = _two * zv**_two - xv**_two - yv**_two
    cv = yv * zv
    d  = xv * zv
    e  = b * vyv - _three * cv * vzv
    f  = _three * d * vzv  - b * vxv
    g  = _three * cv * vxv - _three * d * vyv

    state[6]  = r2
    state[7]  = a
    state[8]  = b
    state[9]  = cv
    state[10] = d
    state[11] = -e
    state[12] = -f
    state[13] = -g
    state[14] = -npfloat(3.0) * a * d
    state[15] = -npfloat(3.0) * a * cv
    state[16] = -a * b


# ===========================================================
#   Per-step adaptive chunk (SLOW PATH — only for hard regions)
#   Uses LOCAL GYROPERIOD to set dt proactively from |B|.
# ===========================================================
_TWO_PI = npfloat(2.0 * np.pi)

def _local_dt_from_B(state, steps_per_local_gyro, dt_min, dt_max):
    """Compute dt from local |B| so we take a fixed number of steps
    per local gyroperiod.  tau_local = 2*pi / |B|  (normalised units
    where q/m = 1), so dt = tau_local / N."""
    Bmag2 = state[14]**2 + state[15]**2 + state[16]**2
    if Bmag2 <= 0.0 or not np.isfinite(Bmag2):
        return dt_max
    Bmag  = np.sqrt(Bmag2)
    tau_local = _TWO_PI / Bmag
    dt = tau_local / steps_per_local_gyro
    return max(dt_min, min(dt, dt_max))


def _ps_adaptive_chunk(
    PS_order, n_output, cur_state, tol, qoverm, ps_step,
    dt_internal, dt_min, dt_max,
    order_low, order_high, grow_factor, shrink_factor, max_retries,
    c, zeta, oip1, sum_terms,
    t_internal,
    halt_on_failure=True,
    steps_per_local_gyro=200,
):
    """
    Process n_output grid points with adaptive substeps.
    BATCHED: for each output point, compute n_sub from local |B| and call
    PS_dipoleB once.  If a step diverges mid-batch, accept the good prefix,
    recompute dt_B at the new position, and retry the remainder.

    Key design rules to prevent hangs:
      - n_sub is capped at MAX_SUB to prevent memory/time explosions
      - After accepting a good prefix (first_bad > 0), retries resets to 0
        so dt_B is recomputed fresh at the new position
      - Only first_bad == 0 (zero progress) counts as a retry toward max_retries
    """
    n_state = 17
    MAX_SUB = 2000          # cap: never ask PS_dipoleB for more than this

    sol_chunk    = np.zeros((n_state, n_output + 1), dtype=npfloat)
    orders_chunk = np.zeros(n_output + 1, dtype=np.int32)
    sol_chunk[:, 0] = cur_state
    max_ps     = 0
    substeps   = 0
    rejections = 0

    for jj in range(1, n_output + 1):
        t_remaining = ps_step
        last_order  = 0
        retries     = 0      # consecutive zero-progress failures
        dt_override = 0.0    # >0 means use this instead of dt_B (after a zero-progress fail)

        while t_remaining > dt_min * 0.01:
            # ---- choose dt ----
            dt_B = _local_dt_from_B(cur_state, steps_per_local_gyro,
                                     dt_min, dt_max)
            if dt_override > 0.0:
                # dt_override may be from shrinking (after zero-progress failure)
                # or from growing (after easy batch).  Either way, cap at dt_B
                # so we never exceed what the local field suggests.
                dt_use = min(dt_override, dt_B)
            else:
                dt_use = dt_B              # fresh from local |B|

            # ---- compute batch size (capped) ----
            n_sub = max(1, int(np.ceil(t_remaining / dt_use)))
            if n_sub > MAX_SUB:
                n_sub = MAX_SUB
                dt_actual = npfloat(dt_use)   # keep the desired dt; don't cover all t_remaining
            else:
                dt_actual = npfloat(t_remaining / n_sub)   # exact coverage

            # ---- ONE Numba call ----
            sol_batch, orders_batch = PS_dipoleB(
                PS_order, n_sub, cur_state[:6].copy(),
                tol, qoverm, dt_actual,
            )

            # ---- find first bad step ----
            # Same quality bar as the fast path: reject if order > order_high
            batch_orders = orders_batch[1:n_sub + 1]
            first_bad = -1                          # -1 means all good
            for kk in range(n_sub):
                if (batch_orders[kk] >= PS_order or
                        batch_orders[kk] > order_high or
                        not np.all(np.isfinite(sol_batch[:6, kk + 1]))):
                    first_bad = kk
                    break

            if first_bad >= 0:
                # ---- some steps failed ----
                if first_bad > 0:
                    # accept the good prefix
                    cur_state[:] = sol_batch[:, first_bad]
                    _tether_aux(cur_state)
                    # safety: if prefix endpoint is NaN/Inf, back up one more step
                    if not np.all(np.isfinite(cur_state)):
                        if first_bad > 1:
                            cur_state[:] = sol_batch[:, first_bad - 1]
                            _tether_aux(cur_state)
                            first_bad -= 1
                        else:
                            # even the first step is bad — treat as zero progress
                            cur_state[:] = sol_batch[:, 0]
                            _tether_aux(cur_state)
                            dt_override = max(float(dt_actual) * shrink_factor, dt_min)
                            retries += 1
                            rejections += 1
                            continue
                    t_remaining -= first_bad * float(dt_actual)
                    substeps    += first_bad
                    good_max = int(batch_orders[:first_bad].max())
                    if good_max > max_ps:
                        max_ps = good_max
                    last_order = int(batch_orders[first_bad - 1])
                    # progress was made → fresh dt_B next time
                    retries     = 0
                    dt_override = 0.0
                    rejections += 1
                    continue        # re-enter loop: dt_B recomputed at new position

                else:
                    # first_bad == 0: zero progress
                    retries    += 1
                    rejections += 1
                    if retries > max_retries:
                        if halt_on_failure:
                            warnings.warn(
                                f"Adaptive PS: {max_retries} retries with "
                                f"zero progress (order={int(batch_orders[0])}) "
                                f"at t={float(t_internal + ps_step - t_remaining):.4f}, "
                                f"dt={float(dt_actual):.6e}. HALTING.",
                                RuntimeWarning, stacklevel=2,
                            )
                            for kk in range(jj, n_output + 1):
                                sol_chunk[:, kk] = cur_state
                                orders_chunk[kk] = -1
                            t_internal += (ps_step - t_remaining)
                            return (sol_chunk, orders_chunk, cur_state,
                                    dt_use, t_internal, max_ps,
                                    substeps, rejections, True)
                        else:
                            # force-accept the bad step
                            cur_state[:] = sol_batch[:, 1]
                            _tether_aux(cur_state)
                            t_remaining -= float(dt_actual)
                            substeps    += 1
                            retries      = 0
                            dt_override  = 0.0
                            continue
                    # shrink dt and retry at same position
                    dt_override = max(float(dt_actual) * shrink_factor, dt_min)
                    continue

            # ---- whole batch accepted ----
            cur_state[:] = sol_batch[:, -1]
            _tether_aux(cur_state)

            # safety: if state went NaN/Inf (near-origin pass), shrink and retry
            if not np.all(np.isfinite(cur_state)):
                # revert to the last known good state (start of this batch)
                cur_state[:] = sol_batch[:, 0]
                _tether_aux(cur_state)
                dt_override = max(float(dt_actual) * shrink_factor, dt_min)
                rejections += 1
                continue

            batch_max = int(batch_orders.max())
            if batch_max > max_ps:
                max_ps = batch_max
            last_order   = int(batch_orders[-1])
            substeps    += n_sub
            t_remaining -= n_sub * float(dt_actual)    # may not be zero if cap was active
            if t_remaining < dt_min * 0.01:
                t_remaining = 0.0                      # clean exit
            retries      = 0

            # ---- growth logic: if batch was easy, try a larger dt next time ----
            # If the highest order in the batch was below order_low, the series
            # converged quickly and we can afford a bigger step.  Apply grow_factor
            # but cap at dt_B (the local-B suggestion) so we don't overshoot.
            if batch_max < order_low:
                grown_dt = min(float(dt_actual) * grow_factor, float(dt_max))
                dt_override = grown_dt   # use this on next iteration instead of dt_B
            else:
                dt_override  = 0.0       # reset: let dt_B decide

        t_internal += ps_step
        sol_chunk[:, jj] = cur_state
        orders_chunk[jj] = last_order

    return (sol_chunk, orders_chunk, cur_state,
            dt_B if n_output > 0 else dt_internal,
            t_internal, max_ps, substeps, rejections,
            False)


# ===========================================================
#    Hybrid adaptive streaming integrator
# ===========================================================
def run_ps_streaming_adaptive(
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
    # ---- adaptive control ----
    order_low=50,
    order_high=300,
    grow_factor=1.5,
    shrink_factor=0.5,
    max_retries=20,
    steps_per_local_gyro=200,
    min_fast_path_N=10,
):
    """
    Hybrid PS integration: uses the original fast PS_dipoleB when the
    orbit is easy, drops to per-step adaptive mode only in hard regions.

    Output format is identical to run_ps_streaming_with_decimation.
    """
    start_time_ps = time.time()
    n_state = 17
    # SAVE_ROWS and n_save imported from functions_library_dipole

    # --- build initial 17-element state ---
    cur_state = np.zeros(n_state, dtype=npfloat)
    cur_state[0:6] = initial_pos_vel_ps
    _tether_aux(cur_state)

    remaining    = steps_ps
    global_index = 0
    max_ps_global = 0

    # --- adaptive bookkeeping ---
    dt_internal     = npfloat(ps_step)
    dt_min          = npfloat(ps_step * 1e-8)
    dt_max          = npfloat(ps_step * 5.0)
    total_substeps  = 0
    total_rejections = 0
    fast_chunks     = 0
    adaptive_chunks = 0
    hit_atmosphere  = False
    hit_atm_step    = -1
    hit_atm_r       = 0.0
    R_ATMOSPHERE    = 1.0   # in R_E; change to 1.0157 for ~100 km altitude

    # --- pre-allocate scratch for adaptive path ---
    c_scratch    = np.zeros((n_state, PS_order + 1), dtype=npfloat)
    zeta_scratch = np.zeros(PS_order + 1, dtype=npfloat)
    oip1         = _one / (_one + np.arange(PS_order, dtype=npfloat))
    sum_scratch  = np.zeros(n_state, dtype=npfloat)

    # --- internal time ---
    t_internal = npfloat(0.0)

    # --- h5 setup (identical to original) ---
    if write_data:
        f = h5py.File(cache_path, "w")
        ps_grp = f.create_group("ps")
        ps_grp.attrs["ordercap"]          = PS_order
        ps_grp.attrs["numberstepspergyro"]= int(N_STEPS_PER_GYRO_ps)
        ps_grp.attrs["dt"]               = npfloat(ps_step)
        ps_grp.attrs["steps"]            = int(steps_ps)
        ps_grp.attrs["streaming"]        = True
        ps_grp.attrs["chunksize"]        = int(chunk_steps)
        ps_grp.attrs["decimate"]         = int(decimate)
        ps_grp.attrs["tol"]              = npfloat(tol)
        ps_grp.attrs["minphase"]         = npfloat(user_min_phase)
        ps_grp.attrs["E0"]              = float(E0_ps)
        ps_grp.attrs["mu0"]             = float(mu0_ps)
        ps_grp.attrs["t0"]              = 0.0
        ps_grp.attrs["adaptive"]         = True
        ps_grp.attrs["order_low"]        = order_low
        ps_grp.attrs["order_high"]       = order_high
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

    def _use_fast_path(state, ps_step, min_N):
        """Return True if ps_step is small enough for the local B field.
        Check: how many ps_steps fit in one local gyroperiod?
        If enough (>= min_N), the fast Numba path can handle it."""
        Bmag2 = state[14]**2 + state[15]**2 + state[16]**2
        if Bmag2 <= 0.0 or not np.isfinite(Bmag2):
            return True  # weak/zero field — fast path fine
        Bmag = np.sqrt(Bmag2)
        tau_local = _TWO_PI / Bmag
        effective_N = tau_local / ps_step
        return effective_N >= min_N

    try:
        force_adaptive = False   # set True when fast path diverges
        while remaining > 0:
            this_chunk = min(chunk_steps, remaining)
            halted = False

            if not force_adaptive and _use_fast_path(cur_state, ps_step, min_fast_path_N):
                # =============================================
                #  FAST PATH: use original PS_dipoleB batch
                #  (ps_step is small enough for the local field)
                # =============================================
                sol_chunk, orders_chunk = PS_dipoleB(
                    PS_order, this_chunk, cur_state[:6].copy(),
                    tol, qoverm, ps_step,
                )

                chunk_max = int(orders_chunk[1:].max()) if this_chunk > 0 else 0

                # check if any step hit the cap, was too hard, or produced NaN
                has_nan = not np.all(np.isfinite(sol_chunk[:6, -1]))
                if chunk_max >= PS_order or chunk_max > order_high or has_nan:
                    # REDO this chunk in adaptive mode
                    force_adaptive = True
                    dt_internal = _local_dt_from_B(
                        cur_state, steps_per_local_gyro, dt_min, dt_max)
                    continue

                # fast path accepted — count it and clear the flag
                force_adaptive = False
                total_substeps += this_chunk
                fast_chunks += 1

                # update live state from batch output
                cur_state[:6] = sol_chunk[:6, -1].copy()
                _tether_aux(cur_state)
                cur_state[:] = sol_chunk[:, -1]

                # safety: if _tether_aux produced NaN/Inf (near-origin pass),
                # revert to pre-chunk state and redo in adaptive mode
                if not np.all(np.isfinite(cur_state)):
                    cur_state[:6] = sol_chunk[:6, 0].copy()
                    _tether_aux(cur_state)
                    force_adaptive = True
                    dt_internal = _local_dt_from_B(
                        cur_state, steps_per_local_gyro, dt_min, dt_max)
                    continue

                t_internal += this_chunk * ps_step
                max_ps_global = max(max_ps_global, chunk_max)

            else:
                # =============================================
                #  ADAPTIVE PATH: local B says ps_step is too
                #  large — subdivide with B-based dt
                # =============================================
                dt_internal = _local_dt_from_B(
                    cur_state, steps_per_local_gyro, dt_min, dt_max)

                (sol_chunk, orders_chunk, cur_state, dt_internal,
                 t_internal, chunk_max_ps, chunk_substeps, chunk_rejections,
                 halted
                ) = _ps_adaptive_chunk(
                    PS_order, this_chunk, cur_state, tol, qoverm, ps_step,
                    dt_internal, dt_min, dt_max,
                    order_low, order_high, grow_factor, shrink_factor, max_retries,
                    c_scratch, zeta_scratch, oip1, sum_scratch,
                    t_internal,
                    steps_per_local_gyro=steps_per_local_gyro,
                )

                max_ps_global    = max(max_ps_global, chunk_max_ps)
                total_substeps  += chunk_substeps
                total_rejections += chunk_rejections
                adaptive_chunks += 1
                force_adaptive = False   # let next chunk try fast path again

            # ---- atmospheric impact check (diagnostic only, does not halt) ----
            r_sq = sol_chunk[0]**2 + sol_chunk[1]**2 + sol_chunk[2]**2
            below = np.where(r_sq < R_ATMOSPHERE**2)[0]
            if len(below) > 0:
                r_min_chunk = float(np.sqrt(r_sq[below].min()))
                hit_atm_r = min(hit_atm_r, r_min_chunk) if hit_atmosphere else r_min_chunk
                if not hit_atmosphere:
                    hit_atmosphere = True
                    hit_atm_step = global_index + below[0]

            # ---- write chunk to h5 (same for both paths) ----
            idx_chunk = np.arange(global_index, global_index + this_chunk + 1)

            if global_index == 0:
                sol_eff    = sol_chunk
                orders_eff = orders_chunk
                idx_eff    = idx_chunk
            else:
                sol_eff    = sol_chunk[:, 1:]
                orders_eff = orders_chunk[1:]
                idx_eff    = idx_chunk[1:]

            if decimate <= 1:
                keep = np.ones_like(idx_eff, dtype=bool)
            else:
                keep = (idx_eff % decimate == 0) | (idx_eff == steps_ps)

            sol_keep    = sol_eff[:, keep]
            orders_keep = orders_eff[keep]

            if sol_keep.shape[1] > 0:
                max_ps_global = max(max_ps_global, int(orders_keep.max()))

                if write_data:
                    old_len = dset_y.shape[1]
                    new_len = old_len + sol_keep.shape[1]
                    dset_y.resize((n_save, new_len))
                    dset_orders.resize((new_len,))
                    dset_y[:, old_len:new_len]   = sol_keep[SAVE_ROWS, :]
                    dset_orders[old_len:new_len] = orders_keep

            global_index += this_chunk
            remaining    -= this_chunk

            # --- Dragt conservation check ---
            if dragt_monitor is not None:
                dragt_monitor.check(sol_chunk, step_index=global_index)

            # --- halt if adaptive path gave up ---
            if halted:
                print(f"\n  *** Integration halted at t={float(t_internal):.2f} "
                      f"(step {global_index}/{steps_ps}) — "
                      f"PS series cannot converge at dt_min.\n"
                      f"  Good trajectory data saved up to this point.\n")
                break

        # --- finalise ---
        if write_data:
            ps_grp.attrs["max_ps"]           = max_ps_global
            ps_grp.attrs["total_substeps"]   = total_substeps
            ps_grp.attrs["total_rejections"] = total_rejections
            ps_grp.attrs["hit_atmosphere"]   = hit_atmosphere
            ps_grp.attrs["hit_atm_step"]     = hit_atm_step
            ps_grp.attrs["hit_atm_r"]        = hit_atm_r

        completed_steps = global_index
        elapsed_ps = time.time() - start_time_ps
        status = "HALTED" if completed_steps < steps_ps else "complete"
        print(f"\n  Adaptive PS {status}:"
              f"\n    completed       = {completed_steps:,} / {steps_ps:,} steps"
              f"\n    fast chunks     = {fast_chunks}"
              f"\n    adaptive chunks = {adaptive_chunks}"
              f"\n    substeps        = {total_substeps:,}"
              f"  ({total_rejections:,} rejected)"
              f"\n    max order       = {max_ps_global}"
              f"\n    final dt        = {float(dt_internal):.4f}"
              f"  (nominal = {float(ps_step):.4f})"
              f"\n    wall time       = {elapsed_ps:.1f} s")
        if hit_atmosphere:
            print(f"    *** ATMOSPHERE FLAG: particle crossed r < {R_ATMOSPHERE} R_E "
                  f"at step {hit_atm_step:,} (r_min = {hit_atm_r:.4f} R_E) ***")
        print()

        return max_ps_global, elapsed_ps

    finally:
        if f is not None:
            f.close()
