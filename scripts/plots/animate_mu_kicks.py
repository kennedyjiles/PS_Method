"""
Magnetic-moment "kick" explainer video (companion to animate_section.py).

mu is nearly constant between equator crossings and changes by a kick at each
one; the kick is close to A sin(theta - theta0) in the gyrophase theta, so the
motion is the map (theta_n, mu_n) -> (theta_n+1, mu_n+1). Per run: gyrophase
dial + kick bar, kick-vs-theta with the fitted sine, running mu, and the
(theta, mu) map. With more than one run an accumulation scene compares
max|mu - mu0| against sqrt(n); the finale is Chirikov's standard map at
K = 0.6 and 1.4 (skip with --no-standard-map).

Usage:
    python scripts/plots/animate_mu_kicks.py                       # DEFAULT_RUNS in DEFAULT_DATA
    python scripts/plots/animate_mu_kicks.py --runs 900047:regular 2a2b24:chaotic --data DIR
    python scripts/plots/animate_mu_kicks.py --preview [--preview-every N]   # PNG stills only
    (--out FILE, default mu_kicks_explainer.mp4; --dpi N, default 120)
Input:  <data>/<stem>_full.h5 with ps/y rows [x y z vx vy vz Bx By Bz] and
        summary_json; loaded whole; decimate must be 1 (checked) and GYRO_AVG
        must equal the run's steps per gyroperiod.
Definitions: mu = v_perp^2/B box-averaged over GYRO_AVG samples, read at the
mirror points before/after each crossing, in units of its first value;
theta = atan2(v_phi, v_rho) at the sample before the crossing (both crossing
directions counted; runs are cut at the first r < R_ATM).
"""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter

# --- Style (matches animate_merged.py / animate_section.py) ---
plt.style.use('dark_background')
CUSTOM_BG = '#212330'
plt.rcParams['savefig.facecolor'] = CUSTOM_BG
LINE_C = '#1C9175'
POINT_C = '#C75F09'
LOSS_C = '#e0484f'
GUIDE_C = '#9aa0b4'
TEXT_C = '#e6e6e6'
SUB_C = '#b8bccb'
UP_C = '#3fb58f'
DOWN_C = '#d9734a'

DEFAULT_DATA = Path.home() / 'Developer/ps_method_data/dipoleb/section_demo/_rawdata'
DEFAULT_RUNS = ['900047:regular', '2a2b24:chaotic']
R_ATM = 1.0157
GYRO_AVG = 20           # steps per gyro-average (= steps_per_gyro in the runs)

FPS = 50
INTRO_HOLD = 90
SLOW_KICKS = 6
FRAMES_PER_SLOW = 50
MED_KICKS = 60          # crossings shown at a moderate pace after the slow ones
FRAMES_PER_MED = 4
FAST_FRAMES = 300
END_HOLD = 120
LOSS_HOLD = 110
MAP_FRAMES = 300
ACC_FRAMES = 400
ACC_PAUSE = 110
ACC_HOLD = 250
MAP_HOLD = 200


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_run(path):
    """Per-crossing (theta, mu_before, mu_after, kick) plus the sine-fit amplitude and phase."""
    with h5py.File(path, 'r') as f:
        dec = int(f['ps'].attrs.get('decimate', 1))
        if dec != 1:
            raise SystemExit(f'{path}: stored with decimate={dec}; this script needs decimate = 1')
        y = f['ps/y'][:]
        meta = json.loads(f.attrs['summary_json'])['meta']
    x, yy, z, vx, vy, vz, bx, by, bz = y
    B = np.sqrt(bx * bx + by * by + bz * bz)
    vpar = (vx * bx + vy * by + vz * bz) / B
    speed2 = vx * vx + vy * vy + vz * vz
    mu = (speed2 - vpar ** 2) / B
    mub = np.convolve(mu, np.ones(GYRO_AVG) / GYRO_AVG, 'same')

    mirror = np.where(np.sign(vpar[:-1]) != np.sign(vpar[1:]))[0]
    eq = np.where(np.sign(z[:-1]) != np.sign(z[1:]))[0]

    r3 = np.sqrt(x * x + yy * yy + z * z)
    hit = np.where(r3 < R_ATM)[0]
    hit_step = int(hit[0]) if len(hit) else None

    j = np.searchsorted(mirror, eq)
    ok = (j > 0) & (j < len(mirror))
    if hit_step is not None:
        ok &= eq < hit_step
    eq, j = eq[ok], j[ok]

    rho = np.hypot(x, yy)
    vr = (vx * x + vy * yy) / rho
    vphi = (-vx * yy + vy * x) / rho
    theta = np.arctan2(vphi[eq], vr[eq])

    mu0 = mub[mirror[0]]
    mu_before = mub[mirror[j - 1]] / mu0
    mu_after = mub[mirror[j]] / mu0
    kick = mu_after - mu_before

    # sine fit  kick = a sin(theta) + b cos(theta)
    Amat = np.c_[np.sin(theta), np.cos(theta)]
    (a, b), *_ = np.linalg.lstsq(Amat, kick, rcond=None)
    amp, ph = np.hypot(a, b), np.arctan2(-b, a)

    L = rho[0]
    eps = 3 * np.sqrt(speed2[0] - vz[0] ** 2) * L ** 2

    return dict(stem=meta['stem'], E=meta['energy_eV'], pitch=meta['pitch_deg'],
                L=L, eps=eps, theta=theta, before=mu_before, after=mu_after,
                kick=kick, amp=amp, ph=ph, lost=hit_step is not None)


def energy_label(E):
    for v, u in [(1e9, 'GeV'), (1e6, 'MeV'), (1e3, 'keV')]:
        if E >= v:
            return f'{E / v:g} {u}'
    return f'{E:g} eV'


def pct(v):
    """Readable percent: 0.21% / 23%."""
    a = abs(v) * 100
    return f'{v * 100:+.2f}%' if a < 1 else f'{v * 100:+.0f}%'


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
def style(ax, xlabel, ylabel, title):
    """Dark-theme axes with labels and title."""
    ax.set_facecolor(CUSTOM_BG)
    for s in ax.spines.values():
        s.set_color('#6b7085')
    ax.tick_params(colors=SUB_C, labelsize=10)
    ax.set_xlabel(xlabel, color=TEXT_C, fontsize=12)
    ax.set_ylabel(ylabel, color=TEXT_C, fontsize=12)
    ax.set_title(title, color=SUB_C, fontsize=12, pad=6)


THETA_TICKS = ([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi],
               [r'$-\pi$', r'$-\pi/2$', '0', r'$\pi/2$', r'$\pi$'])


class Scene:
    def __init__(self, fig, run, kind):
        self.run = run
        fig.clf()
        fig.patch.set_facecolor(CUSTOM_BG)
        word = 'Regular' if kind == 'regular' else 'Chaotic'
        fig.text(0.03, 0.965, f"{word}:  proton, {energy_label(run['E'])}, "
                 f"{run['pitch']:g}° pitch,  L = {run['L']:.1f},  "
                 f"$\\varepsilon$ = {run['eps']:.3f}",
                 color=TEXT_C, fontsize=15, ha='left', va='top')
        self.caption = fig.text(0.5, 0.018, '', color=TEXT_C, fontsize=15,
                                ha='center', va='bottom')

        A = run['amp']
        k = run['kick']
        kl = 1.25 * max(A, np.abs(k).max())

        # --- dial ---
        self.axd = fig.add_axes([0.035, 0.53, 0.20, 0.36])
        self.axd.set_aspect('equal'); self.axd.set_axis_off()
        self.axd.set_xlim(-1.35, 1.35); self.axd.set_ylim(-1.35, 1.35)
        t = np.linspace(0, 2 * np.pi, 200)
        self.axd.plot(np.cos(t), np.sin(t), color=GUIDE_C, lw=1.2)
        for ang, lab in [(0, r'$0$'), (np.pi / 2, r'$\pi/2$'), (np.pi, r'$\pi$'),
                         (-np.pi / 2, r'$-\pi/2$')]:
            self.axd.text(1.2 * np.cos(ang), 1.2 * np.sin(ang), lab, color=SUB_C,
                          fontsize=10, ha='center', va='center')
        self.needle, = self.axd.plot([0, 1], [0, 0], color=POINT_C, lw=3)
        self.tip, = self.axd.plot([1], [0], 'o', color=POINT_C, ms=8)
        self.axd.set_title('gyrophase $\\theta$ at this equator crossing',
                           color=SUB_C, fontsize=12, pad=4)

        # --- kick bar ---
        self.axb = fig.add_axes([0.255, 0.56, 0.035, 0.30])
        self.axb.set_facecolor(CUSTOM_BG)
        self.axb.set_xlim(-1, 1); self.axb.set_ylim(-kl, kl)
        self.axb.axhline(0, color=GUIDE_C, lw=1)
        self.axb.set_xticks([]); self.axb.tick_params(colors=SUB_C, labelsize=9)
        self.axb.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{100 * v:.2g}%'))
        for s in self.axb.spines.values():
            s.set_color('#6b7085')
        self.bar = self.axb.bar([0], [0], width=1.4, color=UP_C)[0]
        self.axb.set_title(r'kick $\Delta\mu$', color=SUB_C, fontsize=11, pad=4)
        self.kick_txt = fig.text(0.272, 0.52, '', color=TEXT_C, fontsize=13, ha='center')

        # --- kick rule ---
        self.axk = fig.add_axes([0.40, 0.56, 0.23, 0.32])
        style(self.axk, r'gyrophase $\theta$', r'kick $\Delta\mu\ /\ \mu_0$',
              'the kick rule')
        self.axk.set_xlim(-np.pi, np.pi); self.axk.set_ylim(-kl, kl)
        self.axk.set_xticks(*THETA_TICKS)
        self.axk.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{100 * v:.2g}%'))
        self.axk.axhline(0, color='#4a4f63', lw=0.8)
        tt = np.linspace(-np.pi, np.pi, 300)
        self.sine, = self.axk.plot(tt, A * np.sin(tt - run['ph']), color=GUIDE_C,
                                   lw=1.3, ls='--', alpha=0)
        self.kpts, = self.axk.plot([], [], '.', color=LINE_C, ms=4)
        self.knew, = self.axk.plot([], [], 'o', color=POINT_C, ms=8)

        # --- running mu ---
        self.axm = fig.add_axes([0.05, 0.155, 0.58, 0.27])
        style(self.axm, 'equator crossing  n', r'$\mu_n\ /\ \mu_0$', r'$\mu$ after each kick')
        self.mline, = self.axm.plot([], [], '-', color=LINE_C, lw=1.0)
        self.mnew, = self.axm.plot([], [], 'o', color=POINT_C, ms=6)
        self.axm.axhline(1, color='#4a4f63', lw=0.8)
        if run['lost']:
            self.axm.axhline(0, color=LOSS_C, lw=1, ls=':', alpha=0.7)

        # --- (theta, mu) map ---
        self.axp = fig.add_axes([0.72, 0.155, 0.25, 0.73])
        style(self.axp, r'gyrophase $\theta$', r'$\mu\ /\ \mu_0$  (before the kick)',
              r'the map  $(\theta_n,\ \mu_n)$')
        self.axp.set_xlim(-np.pi, np.pi); self.axp.set_xticks(*THETA_TICKS)
        lo, hi = run['before'].min(), run['before'].max()
        pad = 0.08 * (hi - lo)
        self.axp.set_ylim(lo - pad, hi + pad)
        self.ppts, = self.axp.plot([], [], '.', color=LINE_C, ms=3 if len(run['kick']) > 500 else 5)
        self.pnew, = self.axp.plot([], [], 'o', color=POINT_C, ms=7)

        self.n_all = len(k)

    def show(self, n, needle=None, bar_frac=1.0, highlight=True):
        """Show crossings 0..n-1 (n-1 is the current one)."""
        r = self.run
        i = n - 1
        th = r['theta'][i] if needle is None else needle
        self.needle.set_data([0, np.cos(th)], [0, np.sin(th)])
        self.tip.set_data([np.cos(th)], [np.sin(th)])
        kv = r['kick'][i] * bar_frac
        self.bar.set_height(kv)
        self.bar.set_color(UP_C if kv >= 0 else DOWN_C)
        self.kick_txt.set_text(pct(kv) if bar_frac > 0 else '')

        m = n if bar_frac >= 1 else n - 1
        self.kpts.set_data(r['theta'][:m], r['kick'][:m])
        self.ppts.set_data(r['theta'][:m], r['before'][:m])
        self.mline.set_data(np.arange(1, m + 1), r['after'][:m])
        if highlight and m > 0:
            self.knew.set_data([r['theta'][m - 1]], [r['kick'][m - 1]])
            self.pnew.set_data([r['theta'][m - 1]], [r['before'][m - 1]])
            self.mnew.set_data([m], [r['after'][m - 1]])
        else:
            for a in (self.knew, self.pnew, self.mnew):
                a.set_data([], [])
        # x range of the running-mu panel grows with n
        xmax = max(10, m * 1.08)
        self.axm.set_xlim(0, xmax)
        seg = r['after'][:max(m, 1)]
        lo, hi = min(seg.min(), 1), max(seg.max(), 1)
        if r['lost']:
            lo = 0
        span = max(hi - lo, 4 * r['amp'])
        self.axm.set_ylim(lo - 0.1 * span, hi + 0.1 * span)


def title_card(fig, writer, text, sub, n):
    fig.clf()
    fig.patch.set_facecolor(CUSTOM_BG)
    fig.text(0.5, 0.57, text, color=TEXT_C, fontsize=30, ha='center', va='center')
    fig.text(0.5, 0.43, sub, color=SUB_C, fontsize=17, ha='center', va='center')
    for _ in range(n):
        writer.grab_frame()


def play_run(fig, writer, run, kind):
    """Slow kicks, medium pace with the sine fading in, fast ramp, loss hold."""
    word = 'Regular' if kind == 'regular' else 'Chaotic'
    title_card(fig, writer, f'{word} orbit: the kicks to $\\mu$',
               f"{energy_label(run['E'])} proton,  {run['pitch']:g}° pitch,  "
               f"L = {run['L']:.1f},  $\\varepsilon$ = {run['eps']:.3f}", INTRO_HOLD)
    sc = Scene(fig, run, kind)
    N = sc.n_all

    # 1. slow: needle swings in, bar grows, points drop
    sc.caption.set_text(r'Between equator crossings $\mu$ is (almost) constant.  '
                        r'At each crossing it gets a kick that depends on the gyrophase $\theta$')
    prev = run['theta'][0] - 1.0
    for n in range(1, min(SLOW_KICKS, N) + 1):
        th = run['theta'][n - 1]
        dth = (th - prev + np.pi) % (2 * np.pi) - np.pi
        for f in range(FRAMES_PER_SLOW):
            u = f / FRAMES_PER_SLOW
            if u < 0.35:        # needle swings to the new phase
                s = u / 0.35
                sc.show(n, needle=prev + dth * (3 * s * s - 2 * s ** 3), bar_frac=0)
            elif u < 0.6:       # kick bar grows
                s = (u - 0.35) / 0.25
                sc.show(n, bar_frac=s, highlight=False)
            else:
                sc.show(n)
            writer.grab_frame()
        prev = th

    # 2. medium pace, the sine appears
    sc.caption.set_text(r'The kick rule is a sine:  $\Delta\mu \approx A\,\sin(\theta-\theta_0)$,  '
                        f"A = {100 * run['amp']:.2g}%.  So the motion is a map "
                        r'$(\theta_n,\mu_n)\to(\theta_{n+1},\mu_{n+1})$')
    stop_med = min(MED_KICKS, N)
    for n in range(SLOW_KICKS + 1, stop_med + 1):
        for f in range(FRAMES_PER_MED):
            sc.sine.set_alpha(min(0.8, (n - SLOW_KICKS) / 20))
            sc.show(n)
            writer.grab_frame()
    sc.sine.set_alpha(0.8)

    # 3. fast ramp to the end
    if kind == 'regular':
        sc.caption.set_text('Regular: successive phases stay correlated, so the kicks cancel.  '
                            rf"$\mu$ stays within $\pm${100 * np.abs(run['after'] - 1).max():.2g}%"
                            f' over all {N:,} kicks: an invariant curve')
    else:
        sc.caption.set_text('Chaotic: one kick shifts the next phase by a radian or more, '
                            r'so the phases scramble and $\mu$ random-walks')
    if N > stop_med:
        ramp = np.unique(np.geomspace(stop_med + 1, N, FAST_FRAMES).astype(int))
        for n in ramp:
            sc.show(n)
            writer.grab_frame()
    for _ in range(END_HOLD):
        writer.grab_frame()

    if run['lost']:
        sc.mnew.set_color(LOSS_C)
        sc.caption.set_text(f'The pitch angle random-walks into the loss cone: '
                            f'the proton reaches the atmosphere after {N} kicks ({N // 2} bounces)')
        for _ in range(LOSS_HOLD):
            writer.grab_frame()


def loss_cone_dev(run):
    """|mu - mu0|/mu0 at which the particle reaches the loss cone (dipole field
    line through L, atmosphere at R_ATM, energy conserved)."""
    L, a0 = run['L'], np.radians(run['pitch'])
    cos2 = R_ATM / L                      # r = L cos^2(lat) at the atmosphere
    sin2 = 1 - cos2
    b_ratio = np.sqrt(1 + 3 * sin2) / cos2 ** 3
    sin2_lc = 1 / b_ratio                 # equatorial loss-cone sin^2(alpha)
    return 1 - sin2_lc / np.sin(a0) ** 2


def accumulation(fig, writer, runs):
    """Both runs on one log-log plot: the largest change in mu so far, in units
    of a single kick.  Regular stays below one kick; chaotic grows like a random
    walk until mu reaches the loss cone."""
    fig.clf()
    fig.patch.set_facecolor(CUSTOM_BG)
    fig.text(0.5, 0.955, 'Do the kicks add up?', color=TEXT_C, fontsize=24,
             ha='center', va='top')
    cap = fig.text(0.5, 0.025, '', color=TEXT_C, fontsize=15, ha='center', va='bottom')
    ax = fig.add_axes([0.08, 0.14, 0.58, 0.74])
    style(ax, 'number of kicks  n',
          r'largest $|\mu_n-\mu_0|$ so far,  in units of one kick $A$', '')
    ax.set_xscale('log'); ax.set_yscale('log')

    curves = []
    nmax = 1
    ytop = 30
    for run, kind in runs:
        dev = np.abs(run['after'] - 1)
        env = np.maximum.accumulate(dev) / run['amp']
        col = LINE_C if kind == 'regular' else POINT_C
        word = 'regular' if kind == 'regular' else 'chaotic'
        ln, = ax.plot([], [], color=col, lw=3,
                      label=f"{word}  ($\\varepsilon$ = {run['eps']:.3f},  A = {100 * run['amp']:.2g}%)")
        tip, = ax.plot([], [], 'o', color=col, ms=8)
        curves.append(dict(run=run, kind=kind, env=env, ln=ln, tip=tip, col=col))
        nmax = max(nmax, len(env))
    n = np.arange(1, nmax + 1)
    ax.plot(n, np.sqrt(n), '--', color=GUIDE_C, lw=1.5, label=r'random walk:  $\sqrt{n}$')
    ax.plot(n, n, ':', color=GUIDE_C, lw=1.5, label=r'every kick adds:  $n$')
    ax.axhline(1, color='#4a4f63', lw=1)
    ax.text(nmax, 1.08, 'one kick', color=SUB_C, fontsize=11, ha='right', va='bottom')
    for c in curves:
        if c['run']['lost']:
            lc = loss_cone_dev(c['run']) / c['run']['amp']
            ax.axhline(lc, color=LOSS_C, lw=1.2, ls='-.', alpha=0.8)
            ax.text(nmax, lc * 1.1, 'loss cone: lost if μ changes this much',
                    color=LOSS_C, fontsize=12, ha='right', va='bottom')
            ytop = lc * 5
    lo = min(c['env'].min() for c in curves)
    ax.set_xlim(0.8, nmax * 1.3); ax.set_ylim(lo * 0.6, ytop)
    ax.legend(loc='upper left', frameon=False, fontsize=12, labelcolor=TEXT_C)

    side = fig.text(0.70, 0.80, '', color=TEXT_C, fontsize=15, ha='left', va='top',
                    wrap=True)

    ramp = np.unique(np.geomspace(1, nmax, ACC_FRAMES).astype(int))
    lost_shown = False
    for m in ramp:
        for c in curves:
            k = min(m, len(c['env']))
            c['ln'].set_data(np.arange(1, k + 1), c['env'][:k])
            c['tip'].set_data([k], [c['env'][k - 1]])
            if c['run']['lost'] and m >= len(c['env']):
                c['tip'].set_marker('X'); c['tip'].set_color(LOSS_C); c['tip'].set_markersize(14)
        chaotic_done = any(c['run']['lost'] and m >= len(c['env']) for c in curves)
        if chaotic_done and not lost_shown:
            cap.set_text('Chaotic: the change grows with every kick until μ reaches the loss cone, '
                         'and the particle is lost')
            for _ in range(ACC_PAUSE):
                writer.grab_frame()
            lost_shown = True
        elif not chaotic_done:
            cap.set_text('Both orbits get kicked at every equator crossing.  Watch how the largest change grows')
        else:
            cap.set_text('Regular: thousands of kicks later, μ has still never moved more than one kick')
        writer.grab_frame()

    reg = [c for c in curves if c['kind'] == 'regular']
    cha = [c for c in curves if c['kind'] != 'regular']
    lines = []
    if reg:
        r = reg[0]
        lines.append(f"Regular: {len(r['env']):,} kicks,\nnever more than {r['env'].max():.1f} of one kick\n"
                     f"({100 * r['env'].max() * r['run']['amp']:.2g}% of μ₀).")
    if cha:
        c = cha[0]
        lines.append(f"Chaotic: grew to {c['env'][-1]:.1f} kicks' worth\n"
                     f"({100 * c['env'][-1] * c['run']['amp']:.0f}% of μ₀) in {len(c['env'])} kicks,\nthen lost.")
    lines.append('Same kick rule.\nCorrelated phases cancel;\nscrambled phases accumulate.')
    side.set_text('\n\n'.join(lines))
    cap.set_text('')
    for _ in range(ACC_HOLD):
        writer.grab_frame()


def standard_map(fig, writer, runs):
    """Finale: Chirikov standard map below and above K ~ 1."""
    fig.clf()
    fig.patch.set_facecolor(CUSTOM_BG)
    fig.text(0.5, 0.955, "Chirikov's standard map:   "
             r'$p_{n+1} = p_n + K\sin\theta_n,\qquad \theta_{n+1} = \theta_n + p_{n+1}$',
             color=TEXT_C, fontsize=18, ha='center', va='top')
    fig.text(0.5, 0.02, r'$p \leftrightarrow \mu$,   $\theta \leftrightarrow$ gyrophase,   '
             r"$K \approx A\cdot d\Phi/d\mu$  (kick size $\times$ how much a kick shifts the next phase)",
             color=SUB_C, fontsize=14, ha='center', va='bottom')
    Ks = [(0.6, 'K = 0.6: invariant curves survive\n(like the regular orbit)'),
          (1.4, 'K = 1.4: curves broken, chaotic sea\n(like the chaotic orbit)')]
    rng = np.random.default_rng(1)
    n_ic, n_it = 60, 600
    th0 = rng.uniform(0, 2 * np.pi, n_ic)
    p0 = np.linspace(0, 2 * np.pi, n_ic, endpoint=False)
    data = []
    for j, (K, lab) in enumerate(Ks):
        ax = fig.add_axes([0.07 + j * 0.47, 0.13, 0.40, 0.70])
        style(ax, r'$\theta$', r'$p$', lab)
        ax.title.set_color(TEXT_C); ax.title.set_fontsize(14)
        ax.set_xlim(0, 2 * np.pi); ax.set_ylim(0, 2 * np.pi)
        th, p = th0.copy(), p0.copy()
        T = np.empty((n_it, n_ic)); P = np.empty((n_it, n_ic))
        for i in range(n_it):
            p = (p + K * np.sin(th)) % (2 * np.pi)
            th = (th + p) % (2 * np.pi)
            T[i], P[i] = th, p
        cols = plt.cm.viridis(np.linspace(0.15, 0.95, n_ic))
        sc = ax.scatter([], [], s=0.6, c=[], cmap=None)
        data.append((ax, T, P, cols, sc))
    steps = np.unique(np.geomspace(1, n_it, MAP_FRAMES).astype(int))
    for m in steps:
        for ax, T, P, cols, sc in data:
            sc.set_offsets(np.c_[T[:m].ravel(), P[:m].ravel()])
            sc.set_color(np.tile(cols, (m, 1)))
        writer.grab_frame()
    for _ in range(MAP_HOLD):
        writer.grab_frame()


class StillWriter:
    def __init__(self, fig, outdir, every):
        self.fig, self.outdir, self.every, self.k = fig, Path(outdir), every, 0
        self.outdir.mkdir(parents=True, exist_ok=True)

    def grab_frame(self):
        if self.k % self.every == 0:
            self.fig.savefig(self.outdir / f'frame_{self.k:05d}.png', dpi=60)
        self.k += 1


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data', type=Path, default=DEFAULT_DATA)
    ap.add_argument('--runs', nargs='+', default=DEFAULT_RUNS)
    ap.add_argument('--out', default='mu_kicks_explainer.mp4')
    ap.add_argument('--dpi', type=int, default=120)
    ap.add_argument('--no-standard-map', action='store_true')
    ap.add_argument('--preview', action='store_true')
    ap.add_argument('--preview-every', type=int, default=100)
    args = ap.parse_args()

    runs = []
    for spec in args.runs:
        stem, kind = spec.split(':')
        r = load_run(args.data / f'{stem}_full.h5')
        runs.append((r, kind))
        print(f"{stem}: eps={r['eps']:.3f} kicks={len(r['kick'])} "
              f"A={100 * r['amp']:.3g}% lost={r['lost']}")

    fig = plt.figure(figsize=(16, 8))
    w = (StillWriter(fig, Path(args.out).with_suffix('').as_posix() + '_preview',
                     args.preview_every) if args.preview else None)

    def render(writer):
        for r, kind in runs:
            play_run(fig, writer, r, kind)
        if len(runs) > 1:
            accumulation(fig, writer, runs)
        if not args.no_standard_map:
            standard_map(fig, writer, runs)

    if w:
        render(w)
        print(f'{w.k} frames ({w.k / FPS:.0f} s); stills in {w.outdir}')
    else:
        if not FFMpegWriter.isAvailable():
            raise SystemExit('ffmpeg not found on PATH (use --preview for PNG stills)')
        writer = FFMpegWriter(fps=FPS, bitrate=6000)
        with writer.saving(fig, args.out, dpi=args.dpi):
            render(writer)
        print(f'saved {args.out}')


if __name__ == '__main__':
    main()
