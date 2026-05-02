"""
Bounce and drift period analysis for dipole trajectories.

Bounce detection:
    mirror_times_from_PS             — mirror points from PS coefficient matrix
    init_bounce_stream_state         — initialize streaming bounce detector
    process_bounce_and_drift_chunk   — process one chunk for bounce/drift
    finalize_bounce_stream           — compute bounce period from collected crossings
    bounce_summary                   — formatted bounce period statistics

Drift detection:
    init_drift_stream_state          — initialize streaming drift detector
    record_drift_sample_at_time      — record one drift sample
    record_drift_sample_at_time_npunwrap_semantics — improved unwrap variant
    finalize_drift_stream            — compute drift period from collected samples
    drift_period_from_PS             — drift period from PS coefficients

Internal helpers:
    _unwrap_phi_from_PS              — unwrap azimuthal angle from PS coefficients
    _pick_samples                    — sample selection for drift analysis
"""

import numpy as np

# Default row indices for the 17-row PS coefficient matrix
named_indices = {"vx": 3, "vy": 4, "vz": 5, "Bx": 14, "By": 15, "Bz": 16}


# ========================
# === Mirror Functions ===
# ========================

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
