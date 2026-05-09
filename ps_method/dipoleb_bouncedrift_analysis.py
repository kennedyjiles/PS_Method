"""
Bounce and drift period analysis for dipole trajectories (streaming pipeline).

Bounce detection identifies mirror points by sign changes of s = v·B
(proportional to the parallel velocity v_parallel). Small-|s| values are
suppressed to avoid numerical jitter, and a minimum time gap is enforced
to avoid double-counting. Linear interpolation gives sub-step mirror times.

Drift detection records azimuthal phase φ = atan2(y, x) at each mirror
crossing, with double unwrapping (local at the crossing, global across
crossings) to handle multi-2π jumps.

Streaming API (consumed by dipoleb.py):
    init_bounce_stream_state         — initialize bounce detector state
    init_drift_stream_state          — initialize drift detector state
    process_bounce_and_drift_chunk   — process one chunk, update both states
    bounce_summary                   — formatted bounce period statistics
    finalize_drift_stream            — compute drift period from collected samples

Internal helpers:
    record_drift_sample              — interpolate (x,y) → φ → unwrap → store
"""

import numpy as np

# Default row indices for the 17-row PS coefficient matrix
named_indices = {"vx": 3, "vy": 4, "vz": 5, "Bx": 14, "By": 15, "Bz": 16}


# ===================================================================
# === Streaming state initialisers ==================================
# ===================================================================
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


# ===================================================================
# === Per-chunk update ==============================================
# ===================================================================
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

    s = vx*Bx + vy*By + vz*Bz   # proxy for v_parallel * |B|

    for i in range(len(s)):
        si = s[i]
        ti = t_chunk[i]

        if bounce_state["last_s"] is not None:
            s0, s1 = bounce_state["last_s"], si

            # require both samples to be "significant" to avoid micro-jitter
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

                    # --- record drift sample at this mirror ---
                    record_drift_sample(
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


# ===================================================================
# === Internal helper: record one drift sample ======================
# ===================================================================
def record_drift_sample(
    y_prev, y_curr,
    t_prev, t_curr, tc,
    state,
):
    """
    Linearly interpolate (x, y) at mirror time tc, compute φ = atan2(y, x),
    then unwrap multi-2π jumps locally (between endpoints) and globally
    (relative to the last stored sample).
    """
    # interpolation weight
    w = 0.0 if t_curr == t_prev else (tc - t_prev) / (t_curr - t_prev)

    # raw phi at endpoints
    phi0 = np.arctan2(y_prev[1], y_prev[0])
    phi1 = np.arctan2(y_curr[1], y_curr[0])

    # --- LOCAL unwrap: allow multi-2π jumps between adjacent samples ---
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


# ===================================================================
# === Finalisation ==================================================
# ===================================================================
def bounce_summary(crossing_times_tau, time_scale_sec=None):
    c = np.asarray(crossing_times_tau, dtype=float)
    full_tau = (c[2:] - c[:-2]) if c.size >= 3 else np.array([], float)

    out = {
        "n_crossings": int(c.size),
        "full_tau": full_tau,
        "full_mean_tau": float(np.mean(full_tau)) if full_tau.size else None,
    }

    if time_scale_sec is not None:
        full_s = full_tau * time_scale_sec
        out.update({
            "full_s": full_s,
            "full_mean_s": float(np.mean(full_s)) if full_s.size else None,
            "bounce_frequency_hz": (1.0/float(np.mean(full_s))) if full_s.size else None,
        })
    return out


def finalize_drift_stream(
    drift_state,
    time_scale_sec=None,
    min_phase_rad=1.0,
):
    t = np.asarray(drift_state["t_samples"], dtype=float)
    phi = np.asarray(drift_state["phi_samples"], dtype=float)

    if t.size < 2:
        return {
            "period_tau_fit": None,
            "period_s_fit": None,
            "direction": +1,
        }

    # slope-based estimate
    a, _ = np.polyfit(t.astype(np.float64), phi.astype(np.float64), 1)
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
