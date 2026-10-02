"""
Surface-of-section explainer video (companion to animate_mu_kicks.py).

Left: the 3-D orbit. Right: at every northbound equator crossing (z from - to
+, linearly interpolated) the point (rho, rho_dot/v) is added to the section:
a regular orbit traces one closed curve, a chaotic one scatters. Per run: a
title card, the first SLOW_BOUNCES bounces in detail, a sped-up fill and, for
a run that reaches the atmosphere, a pause at first contact followed by the
dimmed continuation; then a side-by-side finale of all runs.

Usage:
    python scripts/plots/animate_section.py                       # DEFAULT_RUNS in DEFAULT_DATA
    python scripts/plots/animate_section.py --runs 900047:regular 2a2b24:chaotic --data DIR
    python scripts/plots/animate_section.py --preview [--preview-every N]   # PNG stills only
    (--out FILE, default section_explainer.mp4; --dpi N, default 120)
Input:  <data>/<stem>_full.h5 with ps/y rows [x y z vx vy vz ...] and the
        summary_json attribute; loaded whole; must be stored with decimate = 1
        (checked) so every crossing is resolved.
Output: mp4 via ffmpeg (checked for on PATH), or <out stem>_preview/frame_*.png.
Conventions: eps = 3 v_perp L^2 (equatorial launch); rho_dot is scaled by |v|
(the Dragt figures use rho_dot L^2); R_ATM must equal the run's r_atmosphere.
"""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter

# --- Style (matches animate_merged.py) ---
plt.style.use('dark_background')
CUSTOM_BG = '#212330'
plt.rcParams['savefig.facecolor'] = CUSTOM_BG
LINE_C = '#1C9175'      # trajectory / section points
POINT_C = '#C75F09'     # current particle / newest crossing
DIM_C = '#5d6275'       # post-loss points
LOSS_C = '#e0484f'      # atmosphere hit
BOUND_C = '#9aa0b4'     # allowed-region boundary
EARTH_C = '#3a5a8c'
TEXT_C = '#e6e6e6'

DEFAULT_DATA = Path.home() / 'Developer/ps_method_data/dipoleb/section_demo/_rawdata'
DEFAULT_RUNS = ['900047:regular', '2a2b24:chaotic']
R_ATM = 1.0157          # same as r_atmosphere in the run configs

# --- Pacing (frames at FPS) ---
FPS = 50
SLOW_BOUNCES = 6        # bounces drawn in real detail
FRAMES_PER_BOUNCE = 80  # during the slow scene
FILL_FRAMES = 400       # speed-up scene
LOSS_HOLD = 100         # pause on the atmosphere hit
POST_FRAMES = 300       # dimmed continuation after the hit
INTRO_HOLD = 90         # title card per run
FINAL_HOLD = 250        # comparison


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_run(path):
    """Load one run: crossings, atmosphere hit, eps and the z = 0 allowed-region boundary."""
    with h5py.File(path, 'r') as f:
        dec = int(f['ps'].attrs.get('decimate', 1))
        if dec != 1:
            raise SystemExit(f'{path}: stored with decimate={dec}; crossings are found on '
                             f'consecutive samples, so this script needs decimate = 1')
        y = f['ps/y'][:6]
        meta = json.loads(f.attrs['summary_json'])['meta']
    x, yy, z, vx, vy, vz = y
    rho = np.hypot(x, yy)
    rhod = (x * vx + yy * vy) / rho
    vphi = (x * vy - yy * vx) / rho
    speed = np.sqrt(vx[0] ** 2 + vy[0] ** 2 + vz[0] ** 2)

    # northbound equator crossings, linear interpolation to z = 0
    i = np.where((z[:-1] < 0) & (z[1:] >= 0))[0]
    w = -z[i] / (z[i + 1] - z[i])
    lerp = lambda a: a[i] + w * (a[i + 1] - a[i])
    sec = dict(idx=i, rho=lerp(rho), rhod=lerp(rhod) / speed,
               x=lerp(x), y=lerp(yy))

    # first atmosphere contact
    r3 = np.sqrt(x * x + yy * yy + z * z)
    hit = np.where(r3 < R_ATM)[0]
    hit_step = int(hit[0]) if len(hit) else None
    n_before = int((i < hit_step).sum()) if hit_step is not None else len(i)

    # epsilon = 3 v sin(alpha) L^2 (normalized units, equatorial launch)
    L = rho[0]
    vperp0 = np.sqrt(speed ** 2 - vz[0] ** 2) if abs(z[0]) < 1e-12 else speed
    eps = 3 * speed * (vperp0 / speed) * L ** 2

    # allowed region at z = 0 from the conserved canonical momentum
    # P_phi = rho*v_phi + k*rho^2/r^3 (dipole).  k is fitted from the
    # trajectory itself, so units and sign convention don't matter.
    samp = np.linspace(0, len(x) - 1, 20000).astype(int)
    r3s = np.sqrt(x[samp] ** 2 + yy[samp] ** 2 + z[samp] ** 2) ** 3
    A = np.c_[np.ones_like(samp, float), rho[samp] ** 2 / r3s]
    (a, b), *_ = np.linalg.lstsq(A, rho[samp] * vphi[samp], rcond=None)
    rr = np.linspace(rho.min() * 0.9, rho.max() * 1.1, 4000)
    vphi_r = (a + b / rr) / rr          # at z = 0: rho*v_phi = a + b/rho
    avail = speed ** 2 - vphi_r ** 2
    ok = np.where(avail > 0)[0]
    # close the curve: interpolate the two turning points where avail = 0
    i0, i1 = ok[0], ok[-1]
    rl = np.interp(0, [avail[i0 - 1], avail[i0]], [rr[i0 - 1], rr[i0]])
    rh = np.interp(0, [avail[i1 + 1], avail[i1]], [rr[i1 + 1], rr[i1]])
    bound = (np.r_[rl, rr[ok], rh], np.r_[0, np.sqrt(avail[ok]) / speed, 0])

    return dict(stem=meta['stem'], x=x, y=yy, z=z, rho=rho, sec=sec,
                hit_step=hit_step, n_before=n_before, L=L, eps=eps,
                E=meta['energy_eV'], pitch=meta['pitch_deg'],
                bound=bound)


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------
def energy_label(E):
    for v, u in [(1e9, 'GeV'), (1e6, 'MeV'), (1e3, 'keV')]:
        if E >= v:
            return f'{E / v:g} {u}'
    return f'{E:g} eV'


def style_2d(ax):
    ax.set_facecolor(CUSTOM_BG)
    for s in ax.spines.values():
        s.set_color('#6b7085')
    ax.tick_params(colors='#b8bccb', labelsize=11)
    ax.set_xlabel(r'$\rho$  ($R_E$)', color=TEXT_C, fontsize=14)
    ax.set_ylabel(r'$\dot\rho\,/\,v$', color=TEXT_C, fontsize=14)


def section_limits(run):
    br, bv = run['bound']
    pad_r = 0.06 * (br.max() - br.min())
    return (br.min() - pad_r, br.max() + pad_r), (-1.12 * bv.max(), 1.12 * bv.max())


def draw_boundary(ax, run):
    br, bv = run['bound']
    ax.plot(np.r_[br, br[::-1]], np.r_[bv, -bv[::-1]], color=BOUND_C,
            lw=1.2, alpha=0.7)


# Matplotlib's 3-D axes have no depth buffer: each artist is drawn whole,
# in one piece, so a line can't pass partly in front of and partly behind
# the Earth.  We fix the draw order by hand (computed_zorder = False) and
# split every line, point set and the equatorial plane into a "behind"
# part (drawn under the Earth) and a "front" part (drawn over it), using
# the sign of the depth along the viewing direction.  For anything outside
# the Earth (r > 1) this is exact.
ELEV, AZIM = 22, -60
Z_PLANE_B, Z_BEHIND, Z_EARTH, Z_PLANE_F, Z_FRONT = 1, 2, 3, 4, 5


def view_dir(elev=ELEV, azim=AZIM):
    e, a = np.radians(elev), np.radians(azim)
    return np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])


def is_front(x, y, z):
    v = view_dir()
    return x * v[0] + y * v[1] + z * v[2] > 0


def split_line(x, y, z):
    """Two copies of a polyline, NaN-masked into front / behind parts.
    Samples next to a switch are kept in both so the line stays joined."""
    f = is_front(x, y, z)
    fk = f.copy(); fk[1:] |= f[:-1]; fk[:-1] |= f[1:]
    b = ~f
    bk = b.copy(); bk[1:] |= b[:-1]; bk[:-1] |= b[1:]
    nanm = lambda a, m: np.where(m, a, np.nan)
    return ((nanm(x, fk), nanm(y, fk), nanm(z, fk)),
            (nanm(x, bk), nanm(y, bk), nanm(z, bk)))


def draw_earth(ax3d):
    u, v = np.mgrid[0:2 * np.pi:40j, 0:np.pi:20j]
    ax3d.plot_surface(np.cos(u) * np.sin(v), np.sin(u) * np.sin(v), np.cos(v),
                      color=EARTH_C, alpha=1.0, linewidth=0, shade=True,
                      zorder=Z_EARTH)


def draw_plane(ax3d, run):
    """Translucent z = 0 disc, split front/back of the Earth for draw order."""
    r0, r1 = run['rho'].min() * 0.85, run['rho'].max() * 1.15
    v = view_dir()
    a_front = np.arctan2(v[1], v[0])            # azimuth facing the viewer
    for off, zo in ((0, Z_PLANE_F), (np.pi, Z_PLANE_B)):
        t = np.linspace(a_front + off - np.pi / 2, a_front + off + np.pi / 2, 60)
        R, T = np.meshgrid([r0, r1], t)
        ax3d.plot_surface(R * np.cos(T), R * np.sin(T), 0 * R, color='#8fa3c7',
                          alpha=0.12, linewidth=0, shade=False, zorder=zo)


class Scene:
    """Two-panel layout: 3-D orbit (left), section (right)."""

    def __init__(self, fig, run):
        self.run = run
        fig.clf()
        fig.patch.set_facecolor(CUSTOM_BG)
        self.ax3 = fig.add_axes([0.0, 0.04, 0.5, 0.84], projection='3d')
        self.ax3.set_facecolor(CUSTOM_BG)
        self.ax3.set_axis_off()
        self.ax2 = fig.add_axes([0.56, 0.17, 0.40, 0.66])
        style_2d(self.ax2)

        R = run['rho'].max() * 1.05
        zmax = np.abs(run['z']).max() * 1.05
        self.ax3.set_xlim(-R, R); self.ax3.set_ylim(-R, R); self.ax3.set_zlim(-zmax, zmax)
        self.ax3.set_box_aspect((1, 1, zmax / R))
        self.ax3.view_init(elev=ELEV, azim=AZIM)
        self.ax3.computed_zorder = False
        draw_plane(self.ax3, run)
        draw_earth(self.ax3)

        kw = dict(color=LINE_C, lw=0.9, alpha=0.85)
        self.traj_f, = self.ax3.plot([], [], [], zorder=Z_FRONT, **kw)
        self.traj_b, = self.ax3.plot([], [], [], zorder=Z_BEHIND, **kw)
        kw = dict(color=POINT_C, ms=3, alpha=0.9, ls='none', marker='.')
        self.plane_f, = self.ax3.plot([], [], [], zorder=Z_FRONT, **kw)
        self.plane_b, = self.ax3.plot([], [], [], zorder=Z_BEHIND, **kw)
        self.pt3, = self.ax3.plot([], [], [], 'o', color=POINT_C, ms=5, zorder=Z_FRONT + 1)

        draw_boundary(self.ax2, run)
        xl, yl = section_limits(run)
        self.ax2.set_xlim(*xl); self.ax2.set_ylim(*yl)
        self.sec_dim, = self.ax2.plot([], [], '.', color=DIM_C, ms=1.6, alpha=0.6)
        self.sec_pts, = self.ax2.plot([], [], '.', color=LINE_C, ms=2.2, alpha=0.9)
        self.sec_new, = self.ax2.plot([], [], 'o', color=POINT_C, ms=7)
        self.sec_hit, = self.ax2.plot([], [], 'X', color=LOSS_C, ms=12)

        hdr = (f"Proton, {energy_label(run['E'])}, {run['pitch']:g}° pitch    "
               f"L = {run['L']:.1f},  $\\varepsilon$ = {run['eps']:.3f}")
        fig.text(0.03, 0.95, hdr, color=TEXT_C, fontsize=15, ha='left', va='top')
        self.caption = fig.text(0.5, 0.025, '', color=TEXT_C, fontsize=16,
                                ha='center', va='bottom')
        self.counter = fig.text(0.96, 0.86, '', color='#b8bccb', fontsize=12,
                                ha='right', va='bottom')
        self.ax2.set_title('surface of section  (z = 0, heading north)',
                           color='#b8bccb', fontsize=12, pad=8)

    def show(self, step, n_sec, trail=None, n_split=None, highlight=True, hit=False):
        """step: trajectory index shown; n_sec: crossings shown on the section;
        trail: first index of the 3-D trail; n_split: crossings drawn bright
        (the rest dimmed)."""
        r, s = self.run, self.run['sec']
        t0 = 0 if trail is None else trail
        seg = slice(t0, step + 1)
        stride = max(1, (step + 1 - t0) // 4000)
        front, back = split_line(r['x'][seg][::stride], r['y'][seg][::stride],
                                 r['z'][seg][::stride])
        for ln, (a, b, c) in ((self.traj_f, front), (self.traj_b, back)):
            ln.set_data(a, b)
            ln.set_3d_properties(c)
        px, py, pz = r['x'][step], r['y'][step], r['z'][step]
        self.pt3.set_data([px], [py])
        self.pt3.set_3d_properties([pz])
        self.pt3.set_zorder(Z_FRONT + 1 if is_front(px, py, pz) else Z_BEHIND)

        nb = n_sec if n_split is None else min(n_split, n_sec)
        # plane dots: keep 3-D light, only the most recent few hundred
        k0 = max(0, nb - 400)
        qx, qy = s['x'][k0:nb], s['y'][k0:nb]
        f = is_front(qx, qy, 0.0)
        for ln, m in ((self.plane_f, f), (self.plane_b, ~f)):
            ln.set_data(qx[m], qy[m])
            ln.set_3d_properties(np.zeros(m.sum()))

        self.sec_pts.set_data(s['rho'][:nb], s['rhod'][:nb])
        self.sec_pts.set_markersize(5 if nb < 150 else 2.2)
        if n_split is not None and n_sec > n_split:
            self.sec_dim.set_data(s['rho'][n_split:n_sec], s['rhod'][n_split:n_sec])
        else:
            self.sec_dim.set_data([], [])
        if highlight and n_sec > 0:
            self.sec_new.set_data([s['rho'][n_sec - 1]], [s['rhod'][n_sec - 1]])
        else:
            self.sec_new.set_data([], [])
        if hit and r['n_before'] > 0:
            j = r['n_before'] - 1
            self.sec_hit.set_data([s['rho'][j]], [s['rhod'][j]])
        else:
            self.sec_hit.set_data([], [])
        self.counter.set_text(f'{n_sec:,} crossings')


# ---------------------------------------------------------------------------
# Scene scripts
# ---------------------------------------------------------------------------
def title_card(fig, writer, text, sub, n):
    fig.clf()
    fig.patch.set_facecolor(CUSTOM_BG)
    fig.text(0.5, 0.56, text, color=TEXT_C, fontsize=30, ha='center', va='center')
    fig.text(0.5, 0.44, sub, color='#b8bccb', fontsize=17, ha='center', va='center')
    for _ in range(n):
        writer.grab_frame()


def play_run(fig, writer, run, kind):
    """Title card, slow bounces, fast fill and (if lost) the loss scene for one run."""
    s = run['sec']
    idx = s['idx']
    ncross = len(idx)
    lost = run['hit_step'] is not None
    n_ok = run['n_before']

    word = 'Regular' if kind == 'regular' else 'Chaotic'
    title_card(fig, writer, f'{word} orbit',
               f"{energy_label(run['E'])} proton,  {run['pitch']:g}° pitch,  "
               f"L = {run['L']:.1f},  $\\varepsilon$ = {run['eps']:.3f}", INTRO_HOLD)

    sc = Scene(fig, run)

    # 1. slow: first SLOW_BOUNCES bounces in detail
    nslow = min(SLOW_BOUNCES, ncross, n_ok if lost else ncross)
    end = idx[nslow - 1] + 1 if nslow else len(run['x']) // 100
    steps = np.linspace(0, end, nslow * FRAMES_PER_BOUNCE).astype(int)
    sc.caption.set_text('Each time the proton crosses the equator heading north, '
                        r'record its radius $\rho$ and radial speed $\dot\rho$')
    for st in steps:
        n = int(np.searchsorted(idx, st, side='right'))
        sc.show(st, n)
        writer.grab_frame()
    for _ in range(40):
        writer.grab_frame()

    # 2. fill: exponential ramp through the crossings
    stop = n_ok if lost else ncross
    ramp = np.unique(np.geomspace(max(nslow, 1), stop, FILL_FRAMES).astype(int))
    if kind == 'regular':
        sc.caption.set_text('Regular: the points land on one closed curve, '
                            'the orbit never wanders off it')
    else:
        sc.caption.set_text('Chaotic: the points scatter instead of '
                            'lining up on a curve')
    trail_bounces = 3
    for n in ramp:
        st = idx[n - 1]
        tr = idx[max(0, n - 1 - trail_bounces)]
        sc.show(st, n, trail=tr)
        writer.grab_frame()
    # stretch short ramps (e.g. a quick loss) so the scene isn't a blink
    for _ in range(max(0, FILL_FRAMES // 3 - len(ramp))):
        writer.grab_frame()

    if not lost:
        for _ in range(100):
            writer.grab_frame()
        return

    # 3. loss
    sc.show(run['hit_step'], n_ok, trail=idx[max(0, n_ok - trail_bounces)],
            highlight=False, hit=True)
    sc.pt3.set_color(LOSS_C)
    sc.caption.set_text(f'Reaches the atmosphere after {n_ok} bounces: lost')
    for _ in range(LOSS_HOLD):
        writer.grab_frame()
    sc.pt3.set_color(POINT_C)

    sc.caption.set_text("Continuing as if the atmosphere weren't there: "
                        'the orbit wanders over the whole allowed region')
    ramp = np.unique(np.geomspace(n_ok + 1, ncross, POST_FRAMES).astype(int))
    for n in ramp:
        st = idx[n - 1]
        tr = idx[max(0, n - 1 - trail_bounces)]
        sc.show(st, n, trail=tr, n_split=n_ok, hit=True)
        writer.grab_frame()
    for _ in range(100):
        writer.grab_frame()


def finale(fig, writer, runs):
    """All runs' sections side by side."""
    fig.clf()
    fig.patch.set_facecolor(CUSTOM_BG)
    n = len(runs)
    w = 0.84 / n
    for j, (run, kind) in enumerate(runs):
        ax = fig.add_axes([0.08 + j * (w + 0.04 / max(1, n - 1)), 0.14, w - 0.04, 0.66])
        style_2d(ax)
        draw_boundary(ax, run)
        s = run['sec']
        xl, yl = section_limits(run)
        ax.set_xlim(*xl); ax.set_ylim(*yl)
        nb = run['n_before']
        if nb < len(s['rho']):
            ax.plot(s['rho'][nb:], s['rhod'][nb:], '.', color=DIM_C, ms=1.6, alpha=0.6)
        ax.plot(s['rho'][:nb], s['rhod'][:nb], '.', color=LINE_C,
                ms=5 if nb < 150 else 2.2)
        if nb < len(s['rho']):
            ax.plot([s['rho'][nb - 1]], [s['rhod'][nb - 1]], 'X', color=LOSS_C, ms=12)
        word = 'Regular' if kind == 'regular' else 'Chaotic'
        fate = f"lost after {nb} bounces" if run['hit_step'] is not None else 'stays trapped'
        ax.set_title(f"{word}:  L = {run['L']:.1f},  $\\varepsilon$ = {run['eps']:.3f}\n{fate}",
                     color=TEXT_C, fontsize=15, pad=10)
    r0 = runs[0][0]
    fig.text(0.5, 0.95, f"Same energy ({energy_label(r0['E'])}), same pitch "
             f"({r0['pitch']:g}°): only L, and so $\\varepsilon$, changes",
             color=TEXT_C, fontsize=18, ha='center', va='top')
    fig.text(0.5, 0.03, 'closed curve = regular, trapped;   scattered points = chaotic',
             color='#b8bccb', fontsize=15, ha='center', va='bottom')
    for _ in range(FINAL_HOLD):
        writer.grab_frame()


class StillWriter:
    """Stand-in for FFMpegWriter that saves every Nth frame as a PNG."""

    def __init__(self, fig, outdir, every):
        self.fig, self.outdir, self.every, self.k = fig, Path(outdir), every, 0
        self.outdir.mkdir(parents=True, exist_ok=True)

    def grab_frame(self):
        if self.k % self.every == 0:
            self.fig.savefig(self.outdir / f'frame_{self.k:05d}.png',
                             facecolor=CUSTOM_BG, dpi=60)
        self.k += 1


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data', type=Path, default=DEFAULT_DATA,
                    help='folder with <stem>_full.h5 files')
    ap.add_argument('--runs', nargs='+', default=DEFAULT_RUNS,
                    help='stem:regular or stem:chaotic, in play order')
    ap.add_argument('--out', default='section_explainer.mp4')
    ap.add_argument('--dpi', type=int, default=120)
    ap.add_argument('--preview', action='store_true',
                    help='write sparse PNG stills to <out stem>_preview/ instead of a video')
    ap.add_argument('--preview-every', type=int, default=150)
    args = ap.parse_args()

    runs = []
    for spec in args.runs:
        stem, kind = spec.split(':')
        run = load_run(args.data / f'{stem}_full.h5')
        runs.append((run, kind))
        print(f"{stem}: L={run['L']:.2f} eps={run['eps']:.4f} "
              f"crossings={len(run['sec']['idx'])} "
              f"hit={'step ' + str(run['hit_step']) if run['hit_step'] is not None else 'no'} "
              f"(crossings before hit: {run['n_before']})")

    fig = plt.figure(figsize=(16, 8))
    if args.preview:
        w = StillWriter(fig, Path(args.out).with_suffix('').as_posix() + '_preview',
                        args.preview_every)
        for run, kind in runs:
            play_run(fig, w, run, kind)
        finale(fig, w, runs)
        print(f'{w.k} frames ({w.k / FPS:.0f} s); stills in {w.outdir}')
        return

    if not FFMpegWriter.isAvailable():
        raise SystemExit('ffmpeg not found on PATH (use --preview for PNG stills)')
    writer = FFMpegWriter(fps=FPS, bitrate=6000)
    with writer.saving(fig, args.out, dpi=args.dpi):
        for run, kind in runs:
            play_run(fig, writer, run, kind)
        finale(fig, writer, runs)
    print(f'saved {args.out}')


if __name__ == '__main__':
    main()
