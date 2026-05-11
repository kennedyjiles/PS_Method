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

Internal:
    _process_chunk_kernel            — JIT'd inner loop: scans chunk for
                                        sign changes and emits crossings +
                                        drift samples (no Python in hot path)
"""

import numpy as np
from . import utils as ul

# Default row indices for the 17-row PS coefficient matrix.
# Position rows (x, y) are needed by the drift-φ interpolation; velocity
# and B rows are needed by the bounce sign-change detection.
named_indices = {"x": 0, "y": 1, "z": 2,
                 "vx": 3, "vy": 4, "vz": 5,
                 "Bx": 14, "By": 15, "Bz": 16}


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
@ul.maybe_njit
def _process_chunk_kernel(
    y_chunk, t_chunk,
    last_s, last_t, last_y, last_cross_t, last_phi,
    has_prev, has_phi_prev,
    min_gap_tau, s_eps,
    idx_x, idx_y,
    idx_vx, idx_vy, idx_vz, idx_Bx, idx_By, idx_Bz,
):
    """JIT'd inner loop: scan one chunk for v·B sign changes, emit
    interpolated crossing times + corresponding drift φ samples.

    Returns (state scalars + counts + worst-case-sized output buffers).
    The Python wrapper trims the buffers to the populated prefix.
    """
    n = y_chunk.shape[1]
    cross_times = np.zeros(n, dtype=ul.npfloat)
    sample_ts   = np.zeros(n, dtype=ul.npfloat)
    sample_phis = np.zeros(n, dtype=ul.npfloat)
    n_cross   = 0
    n_samples = 0

    for i in range(n):
        ti = t_chunk[i]
        si = (y_chunk[idx_vx, i] * y_chunk[idx_Bx, i]
              + y_chunk[idx_vy, i] * y_chunk[idx_By, i]
              + y_chunk[idx_vz, i] * y_chunk[idx_Bz, i])

        if has_prev:
            # require both samples to be "significant" to avoid micro-jitter
            if abs(last_s) >= s_eps and abs(si) >= s_eps:
                if last_s * si < 0.0 and (ti - last_cross_t) >= min_gap_tau:
                    # --- interpolate mirror time ---
                    denom = si - last_s
                    if denom == 0.0:
                        tc = ti
                    else:
                        tc = last_t + (ti - last_t) * (-last_s) / denom

                    cross_times[n_cross] = tc
                    n_cross += 1
                    last_cross_t = tc

                    # --- inline record_drift_sample ---
                    w = 0.0 if ti == last_t else (tc - last_t) / (ti - last_t)
                    phi0 = np.arctan2(last_y[idx_y], last_y[idx_x])
                    phi1 = np.arctan2(y_chunk[idx_y, i], y_chunk[idx_x, i])
                    # LOCAL unwrap between adjacent samples
                    d = phi1 - phi0
                    if abs(d) > np.pi:
                        phi1 -= 2.0 * np.pi * np.round(d / (2.0 * np.pi))
                    phi_tc = (1.0 - w) * phi0 + w * phi1
                    # GLOBAL unwrap relative to last stored sample
                    if has_phi_prev:
                        d2 = phi_tc - last_phi
                        if abs(d2) > np.pi:
                            phi_tc -= 2.0 * np.pi * np.round(d2 / (2.0 * np.pi))
                    last_phi = phi_tc
                    has_phi_prev = True

                    sample_ts[n_samples] = tc
                    sample_phis[n_samples] = phi_tc
                    n_samples += 1

        last_s = si
        last_t = ti
        last_y = y_chunk[:, i].copy()
        has_prev = True

    return (last_s, last_t, last_y, last_cross_t, last_phi,
            has_phi_prev,
            n_cross, cross_times, n_samples, sample_ts, sample_phis)


def process_bounce_and_drift_chunk(
    y_chunk,
    t_chunk,
    bounce_state,
    drift_state,
    min_gap_tau,
    s_eps,
):
    """Public entry point. Marshals dict-state into the JIT'd kernel and
    appends emitted crossings / drift samples back into the dict state.
    Always uses linear interpolation for sub-step mirror times.
    """
    idx = named_indices

    has_prev = bounce_state["last_s"] is not None
    last_s = ul.npfloat(bounce_state["last_s"]) if has_prev else ul.npfloat(0.0)
    last_t = ul.npfloat(bounce_state["last_t"]) if has_prev else ul.npfloat(0.0)
    last_y = (bounce_state["last_y"]
              if has_prev else np.zeros(y_chunk.shape[0], dtype=y_chunk.dtype))
    last_cross_t = ul.npfloat(bounce_state["last_cross_t"])

    has_phi_prev = drift_state["last_phi"] is not None
    last_phi = ul.npfloat(drift_state["last_phi"]) if has_phi_prev else ul.npfloat(0.0)

    (last_s, last_t, last_y, last_cross_t, last_phi,
     has_phi_prev,
     n_cross, cross_times,
     n_samples, sample_ts, sample_phis) = _process_chunk_kernel(
        y_chunk, t_chunk,
        last_s, last_t, last_y, last_cross_t, last_phi,
        has_prev, has_phi_prev,
        ul.npfloat(min_gap_tau), ul.npfloat(s_eps),
        idx["x"], idx["y"],
        idx["vx"], idx["vy"], idx["vz"],
        idx["Bx"], idx["By"], idx["Bz"],
    )

    bounce_state["last_s"] = last_s
    bounce_state["last_t"] = last_t
    bounce_state["last_y"] = last_y
    bounce_state["last_cross_t"] = last_cross_t
    if n_cross > 0:
        bounce_state["crossing_times"].extend(cross_times[:n_cross].tolist())

    if has_phi_prev:
        drift_state["last_phi"] = last_phi
    if n_samples > 0:
        drift_state["t_samples"].extend(sample_ts[:n_samples].tolist())
        drift_state["phi_samples"].extend(sample_phis[:n_samples].tolist())


# ===================================================================
# === Finalisation ==================================================
# ===================================================================
def bounce_summary(crossing_times_tau, time_scale_sec=None):
    """Final stats are returned as plain float64 (not npfloat) for CSV / JSON
    serialization — the float128 setup convention only applies upstream of
    here, in the integrator and per-step kernel."""
    c = np.asarray(crossing_times_tau, dtype=float)
    full_tau = (c[2:] - c[:-2]) if c.size >= 3 else np.array([], float)

    out = {
        "n_crossings": int(c.size),
        "full_mean_tau": float(np.mean(full_tau)) if full_tau.size else None,
    }

    if time_scale_sec is not None:
        full_s = full_tau * time_scale_sec
        out.update({
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
