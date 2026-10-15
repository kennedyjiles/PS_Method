#!/usr/bin/env python3
"""
boris_section.py -- standalone Boris surface of section (Dragt, z = 0).

COMPLETELY SEPARATE from the PS pipeline:
  * imports nothing from ps_method/ or configs/ (constants are copied below);
  * writes only into its own output folder (default boris_sos/output/<label>/);
  * opens PS files read-only, and only when --ps-h5 is given (overlay plots).

What it does
  1. Reads the physics fields of an existing run config (e.g. a Tu.yml):
     particle, energy_eV, pitch_deg, phi_deg, x/y/z_initial, gyroperiods,
     round_decimals, r_atmosphere. Initial conditions are built exactly as in
     dipoleb.py, so Boris starts from the same state as the PS run.
  2. Integrates with the same synchronized drift-rotate-drift Boris step as
     ps_method.dipoleb_physics.boris_fixed_step (same arithmetic, same order),
     streaming: only equatorial crossings are kept, never the full trajectory.
     dt = round(T_gyro / N, round_decimals), T_gyro = 2*pi*x0^3, as in the
     config loader. Stops when r < r_atmosphere (use --no-stop to continue).
  3. Builds the Dragt section exactly as analysis_chunked does: linear
     interpolation to z = 0, (rho/L, rho_dot*L^2), L from P_phi at launch.
  4. Figures: <label>_boris_section.png, and with --ps-h5 also
     <label>_ps_section.png, <label>_overlay_section.png and a three-panel
     <label>_section_triptych.png (PS | Boris | overlay, shared axes).

Usage
  python boris_section.py RUN_YML [--steps-per-gyro 65] [--gyroperiods 1e6]
                          [--ps-h5 PATH] [--out DIR] [--label NAME]
                          [--max-plot 200000] [--no-stop]
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import yaml
from numba import njit

# ---- constants: copied from ps_method/constants.py (float64) ----
Q_E = 1.602176634e-19
M_E = 9.1093837139e-31
M_P = 1.67262192595e-27
C_LIGHT = 299792458.0
RE = 6378137.0
B_0 = 3.12e-5
EV_TO_J = 1.602176634e-19

COLOR_PS = "#009E73"     # same as dipoleb_plots.COLORS["ps"]
COLOR_BORIS = "#0072B2"  # same as dipoleb_plots.COLORS["boris"]
DOT_MS = 1.0             # crossing marker size; set with --ms


# =====================================================================
# Integrator
# =====================================================================
@njit(cache=True)
def boris_chunk(state, dt, nsteps, cs, r_atm2, stop_on_hit, step0, cbuf):
    """Advance `state` (len-6, modified in place) by up to nsteps Boris steps.

    Arithmetic is identical to boris_fixed_step. Equatorial crossings are
    written to cbuf rows as (t, x, y, vx, vy, vz), linearly interpolated in z
    exactly as analysis_chunked does.

    Returns (n_cross, steps_done, r2_min, pphi_min, pphi_max, hit_step).
    hit_step = -1 if r never fell below r_atmosphere.
    """
    half = 0.5; one = 1.0; two = 2.0; three = 3.0; twopointfive = 2.5
    x = state[0]; y = state[1]; z = state[2]
    vx = state[3]; vy = state[4]; vz = state[5]
    nc = 0
    max_c = cbuf.shape[0]
    r2_min = 1e300
    p_min = 1e300
    p_max = -1e300
    hit = -1
    done = 0
    for i in range(nsteps):
        xo = x; yo = y; zo = z; vxo = vx; vyo = vy; vzo = vz

        # --- half drift ---
        xh = x + half * dt * vx
        yh = y + half * dt * vy
        zh = z + half * dt * vz
        # --- field at the midpoint ---
        r2 = xh*xh + yh*yh + zh*zh
        r5inv = r2**(-twopointfive) if r2 != 0 else 0.0
        Bx = -three * xh * zh * r5inv
        By = -three * yh * zh * r5inv
        Bz = -(three * zh*zh - r2) * r5inv
        # --- rotation ---
        tx = cs * half * dt * Bx
        ty = cs * half * dt * By
        tz = cs * half * dt * Bz
        t2 = tx*tx + ty*ty + tz*tz
        f = two / (one + t2)
        sx = f * tx; sy = f * ty; sz = f * tz
        vpx = vx + (vy * tz - vz * ty)
        vpy = vy + (vz * tx - vx * tz)
        vpz = vz + (vx * ty - vy * tx)
        vx = vx + (vpy * sz - vpz * sy)
        vy = vy + (vpz * sx - vpx * sz)
        vz = vz + (vpx * sy - vpy * sx)
        # --- second half drift ---
        x = xh + half * dt * vx
        y = yh + half * dt * vy
        z = zh + half * dt * vz
        done += 1

        # --- diagnostics ---
        rr2 = x*x + y*y + z*z
        if rr2 < r2_min:
            r2_min = rr2
        rho2 = x*x + y*y
        pphi = (x*vy - y*vx) - cs * rho2 / (rr2 * np.sqrt(rr2))
        if pphi < p_min:
            p_min = pphi
        if pphi > p_max:
            p_max = pphi

        # --- equatorial crossing (same test and interpolation as PS) ---
        if zo * z < 0.0:
            if nc >= max_c:
                # caller sizes cbuf so this cannot happen; flag it if it does
                return -1, done, r2_min, p_min, p_max, hit
            tf = abs(zo) / (abs(zo) + abs(z))
            cbuf[nc, 0] = (step0 + i + tf) * dt
            cbuf[nc, 1] = xo + tf * (x - xo)
            cbuf[nc, 2] = yo + tf * (y - yo)
            cbuf[nc, 3] = vxo + tf * (vx - vxo)
            cbuf[nc, 4] = vyo + tf * (vy - vyo)
            cbuf[nc, 5] = vzo + tf * (vz - vzo)
            nc += 1

        if rr2 < r_atm2 and hit < 0:
            hit = step0 + i + 1
            if stop_on_hit:
                break

    state[0] = x; state[1] = y; state[2] = z
    state[3] = vx; state[4] = vy; state[5] = vz
    return nc, done, r2_min, p_min, p_max, hit


# =====================================================================
# Setup (mirrors dipoleb.py / config_loader.py)
# =====================================================================
def setup_from_yml(path, spg, gyroperiods_override):
    with open(path) as fh:
        cfg = yaml.safe_load(fh)
    if cfg.get("use_gyroradius_L_correction", False):
        sys.exit("use_gyroradius_L_correction=true is not supported here "
                 "(T_gyro would differ from 2*pi*x0^3). Aborting.")
    particle = str(cfg["particle"]).lower()
    if particle.startswith("p"):
        mass, cs, pname = M_P, 1.0, "Proton"
    elif particle.startswith("e"):
        mass, cs, pname = M_E, -1.0, "Electron"
    else:
        sys.exit(f"unknown particle {cfg['particle']!r}")

    ke = float(cfg["energy_eV"]) * EV_TO_J
    gamma = 1.0 + ke / (mass * C_LIGHT**2)
    v_si = C_LIGHT * np.sqrt(1.0 - 1.0 / gamma**2)
    tau0 = gamma * mass / (Q_E * B_0)
    v_tau = v_si * tau0 / RE

    pitch = np.radians(float(cfg["pitch_deg"]))
    phi = np.radians(float(cfg["phi_deg"]))
    v_par = v_tau * np.cos(pitch)
    v_perp = v_tau * np.sin(pitch)
    vx0 = v_perp * np.cos(phi)
    vy0 = v_perp * np.sin(phi)
    vz0 = v_par
    eps = np.finfo(np.float64).eps
    vx0 = 0.0 if abs(vx0) < eps else vx0
    vy0 = 0.0 if abs(vy0) < eps else vy0
    vz0 = 0.0 if abs(vz0) < eps else vz0
    x0 = float(cfg["x_initial"]); y0 = float(cfg.get("y_initial", 0.0))
    z0 = float(cfg.get("z_initial", 0.0))
    s0 = np.array([x0, y0, z0, vx0, vy0, vz0], dtype=np.float64)

    T_gyro = 2.0 * np.pi * x0**3
    rd = int(cfg.get("round_decimals", 1))
    dt = round(T_gyro / spg, rd)
    if dt <= 0:
        sys.exit(f"dt rounded to {dt}; raise round_decimals or lower steps/gyro")
    gyro = float(gyroperiods_override if gyroperiods_override is not None
                 else cfg["gyroperiods"])
    steps = int(gyro * T_gyro / dt)
    return dict(cfg=cfg, particle=pname, cs=cs, gamma=gamma, tau0=tau0,
                state0=s0, T_gyro=T_gyro, dt=dt, steps=steps,
                gyroperiods=gyro, spg_requested=spg,
                spg_actual=T_gyro / dt,
                r_atm=float(cfg.get("r_atmosphere", 1.0)))


def dragt_frame(s0, cs):
    """L-shell, W0^2, P_phi (Dragt units) from the launch state, as run_section."""
    x, y, z, vx, vy, vz = s0
    rho = np.hypot(x, y)
    r = np.sqrt(x*x + y*y + z*z)
    v_phi = (x*vy - y*vx) / rho
    p_code = rho * v_phi - cs * rho**2 / r**3
    L = -cs / p_code if cs * p_code < 0 else r**3 / rho**2
    v = np.sqrt(vx*vx + vy*vy + vz*vz)
    W0_sq = (v * L**2)**2
    rd, r_d = rho / L, r / L
    P_phi = rd * (v_phi * L**2) - cs * rd**2 / r_d**3
    return L, W0_sq, P_phi


def boundary(W0_sq, P_phi, cs):
    """Copy of dipoleb_dragt_analysis.compute_boundary."""
    trapped = (cs * P_phi < 0)
    rho_upper = (2.0 / abs(P_phi)) * 1.2 if (trapped and abs(P_phi) > 0) else 3.0
    rb = np.linspace(0.1, rho_upper, 5000)
    V = 0.5 * (P_phi / rb + cs / rb**2)**2
    ok = (W0_sq - 2.0 * V) >= 0
    if not np.any(ok):
        return None, None
    return rb[ok], np.sqrt(W0_sq - 2.0 * V[ok])


def to_section(c, L):
    """crossing rows (t,x,y,vx,vy,vz) -> (rho/L, rho_dot*L^2)."""
    rs = np.hypot(c[:, 1], c[:, 2])
    rd = (c[:, 1] * c[:, 3] + c[:, 2] * c[:, 4]) / rs
    return rs / L, rd * L**2


# =====================================================================
# PS crossings (read-only), same algorithm as analysis_chunked
# =====================================================================
def ps_crossings(h5_path, t_max=None, chunk=1_000_000):
    import h5py
    rows = []
    with h5py.File(h5_path, "r") as h:
        g = h["ps"]
        ds = g["y"]
        dt = float(g.attrs["dt"]) if "dt" in g.attrs else None
        N = ds.shape[1]
        if t_max is not None and dt is not None:
            N = min(N, int(np.ceil(t_max / dt)) + 2)
        prev = None
        for i0 in range(0, N, chunk):
            i1 = min(i0 + chunk, N)
            a = ds[:6, i0:i1].astype(np.float64)
            if prev is not None:
                a = np.concatenate([prev[:, None], a], axis=1)
                base = i0 - 1
            else:
                base = i0
            z = a[2]
            idx = np.where(z[:-1] * z[1:] < 0)[0]
            if len(idx):
                tf = np.abs(z[idx]) / (np.abs(z[idx]) + np.abs(z[idx + 1]))
                out = np.empty((len(idx), 6))
                out[:, 0] = ((base + idx + tf) * dt) if dt is not None else np.nan
                for k, row in zip(range(1, 6), (0, 1, 3, 4, 5)):
                    out[:, k] = a[row, idx] + tf * (a[row, idx + 1] - a[row, idx])
                rows.append(out)
            prev = a[:, -1].copy()
            print(f"  PS h5: {i1:,}/{N:,} steps read", end="\r")
    print()
    return (np.concatenate(rows) if rows else np.empty((0, 6))), dt


# =====================================================================
# Plotting
# =====================================================================
def thin(a, n):
    if len(a) <= n:
        return a
    return a[np.linspace(0, len(a) - 1, n).astype(np.int64)]


def draw(ax, bnd, launch, sets, title):
    rb, rdb = bnd
    if rb is not None:
        ax.plot(rb, rdb, "k-", lw=1.5, label="Boundary")
        ax.plot(rb, -rdb, "k-", lw=1.5)
    for (rho, rdot, color, label, style) in sets:
        alpha = 1.0 if style == "solo" else 0.5
        ax.plot(rho, rdot, ".", color=color, ms=DOT_MS, mew=0, alpha=alpha,
                label=label, rasterized=True)
    ax.plot(*launch, "D", mfc="blue", mec="black", ms=6, label="Launch (t=0)")
    ax.set_xlabel(r"$\rho$ (Dimensionless)")
    ax.set_ylabel(r"$\dot{\rho}$ (Dimensionless)")
    ax.set_title(title)
    ax.grid(True)
    leg = ax.legend(loc="upper right", fontsize=8)
    # enlarge only the crossing markers in the legend (not boundary/launch)
    handles = getattr(leg, "legend_handles", None) or getattr(leg, "legendHandles", [])
    names = {lab for (_, _, _, lab, _) in sets}
    for h, t in zip(handles, leg.get_texts()):
        if t.get_text() in names and hasattr(h, "set_markersize"):
            h.set_markersize(6)


# =====================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_yml")
    ap.add_argument("--steps-per-gyro", type=float, default=65)
    ap.add_argument("--gyroperiods", type=float, default=None,
                    help="override the yml's gyroperiods")
    ap.add_argument("--ps-h5", default=None, help="PS trajectory h5 (read-only) for overlay")
    ap.add_argument("--out", default=None)
    ap.add_argument("--label", default=None)
    ap.add_argument("--max-plot", type=int, default=200_000,
                    help="crossings drawn per set (uniform subsample; all are saved)")
    ap.add_argument("--no-stop", action="store_true",
                    help="keep integrating after r < r_atmosphere")
    ap.add_argument("--chunk", type=int, default=2_000_000)
    ap.add_argument("--ms", type=float, default=1.0,
                    help="marker size for crossings (default 1.0)")
    a = ap.parse_args()
    global DOT_MS
    DOT_MS = a.ms

    P = setup_from_yml(a.run_yml, a.steps_per_gyro, a.gyroperiods)
    run_dir = os.path.basename(os.path.dirname(os.path.abspath(a.run_yml)))
    spg_tag = f"{a.steps_per_gyro:g}"
    label = a.label or f"{run_dir}_boris{spg_tag}"
    here = os.path.dirname(os.path.abspath(__file__))
    out = a.out or os.path.join(here, "output", label)
    os.makedirs(out, exist_ok=True)

    cs, dt, steps = P["cs"], P["dt"], P["steps"]
    print(f"{P['particle']} {float(P['cfg']['energy_eV']):.3g} eV, pitch {P['cfg']['pitch_deg']} deg, "
          f"phi {P['cfg']['phi_deg']} deg, x0 {P['state0'][0]:.6f}")
    print(f"T_gyro={P['T_gyro']:.6g}  dt={dt}  (requested {a.steps_per_gyro:g}/gyro, "
          f"actual {P['spg_actual']:.3f}/gyro)  steps={steps:,}  "
          f"gyroperiods={P['gyroperiods']:.3g}")

    # --- integrate ---
    state = P["state0"].copy()
    cbuf = np.empty((a.chunk, 6))
    parts = []
    r2min, pmin, pmax, hit = 1e300, 1e300, -1e300, -1
    done_total = 0
    t0 = time.time()
    while done_total < steps:
        n = min(a.chunk, steps - done_total)
        nc, done, r2m, pmi, pma, h = boris_chunk(
            state, dt, n, cs, P["r_atm"]**2, not a.no_stop, done_total, cbuf)
        if nc < 0:
            sys.exit("crossing buffer overflow (should not happen)")
        if nc:
            parts.append(cbuf[:nc].copy())
        r2min, pmin, pmax = min(r2min, r2m), min(pmin, pmi), max(pmax, pma)
        if h >= 0 and hit < 0:
            hit = h
        done_total += done
        el = time.time() - t0
        print(f"  {done_total:,}/{steps:,} steps  ({done_total*dt/P['T_gyro']:.3g} T)  "
              f"{el:.0f}s", end="\r")
        if h >= 0 and not a.no_stop:
            break
    print()
    cross = np.concatenate(parts) if parts else np.empty((0, 6))
    elapsed = time.time() - t0

    # --- invariants / section quantities ---
    s0 = P["state0"]
    E0 = 0.5 * float(s0[3:] @ s0[3:])
    E1 = 0.5 * float(state[3:] @ state[3:])
    x, y, z = s0[:3]
    rho0 = np.hypot(x, y); r0 = np.sqrt(x*x + y*y + z*z)
    p0 = (x*s0[4] - y*s0[3]) - cs * rho0**2 / r0**3
    L, W0_sq, P_phi = dragt_frame(s0, cs)
    bnd = boundary(W0_sq, P_phi, cs)
    launch = (rho0 / L, ((x*s0[3] + y*s0[4]) / rho0) * L**2)
    mu_launch = (s0[3]**2 + s0[4]**2) * r0**3 / 2.0  # valid: launch is at z=0
    if len(cross):
        rs = np.hypot(cross[:, 1], cross[:, 2])
        mu_c = (cross[:, 3]**2 + cross[:, 4]**2) * rs**3 / 2.0
        mu_ratio = mu_c / mu_launch
    else:
        mu_ratio = np.empty(0)

    np.savez(os.path.join(out, f"{label}_crossings.npz"),
             crossings=cross, columns=np.array(["t", "x", "y", "vx", "vy", "vz"]),
             L_shell_dragt=L, W0_sq=W0_sq, P_phi_dragt=P_phi, dt=dt,
             T_gyro=P["T_gyro"], state0=s0)

    summary = {
        "run_yml": os.path.abspath(a.run_yml),
        "particle": P["particle"], "energy_eV": float(P["cfg"]["energy_eV"]),
        "pitch_deg": float(P["cfg"]["pitch_deg"]), "phi_deg": float(P["cfg"]["phi_deg"]),
        "x_initial": float(s0[0]),
        "steps_per_gyro_requested": a.steps_per_gyro,
        "steps_per_gyro_actual": P["spg_actual"], "dt": dt,
        "T_gyro": P["T_gyro"], "gyroperiods_requested": P["gyroperiods"],
        "steps_done": done_total,
        "gyroperiods_done": done_total * dt / P["T_gyro"],
        "hit_atmosphere": hit >= 0, "r_atmosphere": P["r_atm"],
        "hit_gyroperiods": (hit * dt / P["T_gyro"]) if hit >= 0 else None,
        "r_min": float(np.sqrt(r2min)),
        "KE_rel_err_end": abs(E1 - E0) / E0,
        "Pphi_rel_range": (pmax - pmin) / abs(p0),
        "n_eq_crossings": int(len(cross)),
        "mu_cross_over_mu_launch_min": float(mu_ratio.min()) if len(mu_ratio) else None,
        "mu_cross_over_mu_launch_max": float(mu_ratio.max()) if len(mu_ratio) else None,
        "note_mu": ("mu at equatorial crossings divided by mu at launch; "
                    "NOT the same quantity as the PS log's mu_max_run"),
        "L_shell_dragt": L, "W0_sq": W0_sq, "P_phi_dragt": P_phi,
        "wall_seconds": elapsed,
    }
    with open(os.path.join(out, f"{label}_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2, default=float)
    for k, v in summary.items():
        print(f"  {k:32s}: {v}")

    # --- figures ---
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    b_rho, b_rd = to_section(thin(cross, a.max_plot), L) if len(cross) else ([], [])
    T_end = done_total * dt / P["T_gyro"]
    tag = (f"Boris, {P['spg_actual']:.1f} steps/gyro, {T_end:.3g} T"
           + (f", hit r<{P['r_atm']} at {summary['hit_gyroperiods']:.3g} T" if hit >= 0 else ""))

    fig, ax = plt.subplots(figsize=(10, 7))
    draw(ax, bnd, launch, [(b_rho, b_rd, COLOR_BORIS, "Boris crossings", "solo")],
         f"Dragt surface of section at z=0 -- {tag}")
    fig.savefig(os.path.join(out, f"{label}_boris_section.png"), dpi=300)
    plt.close(fig)

    if a.ps_h5:
        print("Reading PS crossings (read-only) ...")
        t_lim = cross[-1, 0] if len(cross) else None
        ps_c, ps_dt = ps_crossings(a.ps_h5, t_max=t_lim)
        if ps_dt is not None and t_lim is not None:
            ps_c = ps_c[ps_c[:, 0] <= t_lim]
        p_rho, p_rd = to_section(thin(ps_c, a.max_plot), L) if len(ps_c) else ([], [])
        np.savez(os.path.join(out, f"{label}_ps_crossings_used.npz"),
                 crossings=ps_c, source=os.path.abspath(a.ps_h5))
        ps_tag = f"PS, {len(ps_c):,} crossings to the same time"

        fig, ax = plt.subplots(figsize=(10, 7))
        draw(ax, bnd, launch, [(p_rho, p_rd, COLOR_PS, "PS crossings", "solo")],
             f"Dragt surface of section at z=0 -- {ps_tag}")
        fig.savefig(os.path.join(out, f"{label}_ps_section.png"), dpi=300)
        plt.close(fig)

        ov = [(p_rho, p_rd, COLOR_PS, "PS", "dot"),
              (b_rho, b_rd, COLOR_BORIS, "Boris", "dot")]
        fig, ax = plt.subplots(figsize=(10, 7))
        draw(ax, bnd, launch, ov, "Overlay: PS (green) and Boris (blue)")
        fig.savefig(os.path.join(out, f"{label}_overlay_section.png"), dpi=300)
        plt.close(fig)

        fig, axs = plt.subplots(1, 3, figsize=(24, 7), sharex=True, sharey=True)
        draw(axs[0], bnd, launch, [(p_rho, p_rd, COLOR_PS, "PS", "solo")], ps_tag)
        draw(axs[1], bnd, launch, [(b_rho, b_rd, COLOR_BORIS, "Boris", "solo")], tag)
        draw(axs[2], bnd, launch, ov, "Overlay")
        fig.tight_layout()
        fig.savefig(os.path.join(out, f"{label}_section_triptych.png"), dpi=200)
        plt.close(fig)

    print(f"Output: {out}")


if __name__ == "__main__":
    main()
