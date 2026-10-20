"""
epsilon_map.py plus manually measured launch-fate regime lines.

Identical to epsilon_map.py (same inputs, verdicts, calibration and Stormer
curve) except: the red first-loss curve is labelled eps_{l,0} (first loss
among the phi0 = 0 survey launches) and drawn dash-dot; REGIME_LINES adds the
per-pitch regime edges measured from the gyrophase sweeps (eps in the runs'
eps_initial units; pitch angles without an entry draw nothing), toggled with
--regimes / --no-regimes; the pitch-angle corner label is always drawn; and
the default output is epsilon_map_b_a<pitch>.png so the two never overwrite
each other. The two files should eventually be merged into one with a
--regimes switch.

Usage:
    python3 scripts/plots/epsilon_map_b.py data/dipoleb/<group>/master_simulation_log.csv
        [--regimes | --no-regimes] [--out FILE] [--height 4.0] [--no-legend]
        [--title TEXT] [--eps-l VALUE] [--lost-cls CHAOTIC_LOST,SEA] [--launch-phi DEG]
"""
import csv
import re
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------- constants ---
Q = 1.602176634e-19          # C
B0 = 3.07e-5                 # T   (dipole moment field at equator, L=1)
RE = 6.371e6                 # m
C = 2.99792458e8             # m/s
MP_MEV = 938.27208816        # proton rest mass, MeV
# The red curve's symbol, as it reads in the paper. Deliberately NOT eps_c /
# "chaos threshold": below it the surface of section still holds invariant
# curves -- round tori and odd librating loops interspersed -- and every one of
# them keeps mu bounded, whatever the Dragt & Finn criterion says about them.
# Above it the section fills the accessible region and mu transports to the
# loss cone. Named for that loss ("l" = lost to atmosphere, matching the
# legend); it was eps_b ("bounded") before.
# Defined as the *first* loss over the whole sweep, so it is a lower envelope,
# not a separatrix: interspersed verdicts above it are expected, not a
# contradiction. Change these two together.
# Version b: the red curve is eps_{l,0}, the first loss among the phi0 = 0 survey
# launches. eps_l (no subscript 0) is now the phase-independent edge, the blue
# regime line: the first loss at ANY launch gyrophase.
EPS_SYMBOL = r"\varepsilon_{\ell,0}"   # figure label (mathtext)
EPS_NAME   = "eps_l0"                 # console / plain-text name

# ------------------------------------------------ launch-fate regime lines ---
# Edges of the regimes measured in the gyrophase sweeps (Tu/launch_radius_plot.py),
# in the same eps as the runs' eps_initial (the phi0 = 0 / guiding-centre eps).
# MANUAL and pitch-angle dependent: add an entry per alpha_eq once it is measured.
#   key: alpha_eq in degrees  ->  list of
#        (eps, symbol (mathtext), legend text, colour, linestyle, right-margin label?)
# 30 deg (sweeps 1e6-1e9 eV, 2026-09-24):
#   eps_l: derived 0.0705 (= launch eps 0.074 at the arc top); PLOTTED as 0.071.
#          Data gap it must sit in: all-trapped cells reach 0.0699, first loss 0.0715.
#   eps_f: 0.095 = earliest fragmentation onset seen (1e7); above it trapped/lost interleave
# Line style: the thresholds eps_l, eps_f and Stormer are SOLID; eps_{l,0} (phi0 = 0 only)
# is red DASH-DOT; the grey reference guides are plain dashes. REGIME_LW / EPS_L0_LW set the weights.
SHOW_REGIMES = True          # default for --regimes / --no-regimes
# Grey iso-eps guides (0.05, 0.1, 0.2, 0.3) are DROPPED (curve and label) when a
# special line -- eps_{l,0}, Stormer, or a drawn regime line -- lies within this
# fractional distance of them, so two near-identical curves never both appear.
GUIDE_DROP_REL = 0.15
REGIME_LW = 1.8              # eps_l, eps_f
EPS_L0_LW = 1.4              # eps_{l,0}: the phi0 = 0 special case of eps_l, drawn lighter
EPS_L0_LS = (0, (6, 1.6, 1.2, 1.6))   # ...and dash-dot (phi0 = 0 only); grey guides are plain dashes (4, 3)
REGIME_LINES = {
    30: [(0.071, r"\varepsilon_\ell", "all phases trapped below", "#2f6fb0", "-", True),
         (0.095, r"\varepsilon_f",     "fragmented above",         "#16876f", "-", True)],   # teal
}

EPS_L_85 = 0.457             # measured at alpha_eq = 85 deg (last KAM curve destroyed)
EPS_L = EPS_L_85             # overwritten below (CLI value, or set by the runs)
EPS_DEF_SCALE = 1.0          # overwritten below by calibration against the CSV

# qB0RE in momentum units (MeV/c): the natural rigidity scale of the dipole
P_SCALE_MEV = Q * B0 * RE * C / (1.602176634e-13)   # ~ 5.9e4 MeV

def pc_mev(ke_mev):
    """Relativistic momentum*c in MeV for proton kinetic energy in MeV."""
    return np.sqrt((ke_mev + MP_MEV) ** 2 - MP_MEV ** 2)

SIN_ALPHA = 1.0              # sin(alpha_eq) of the CSV's pitch angle, set after loading

def epsilon(ke_mev, L):
    """launch adiabaticity eps = 3 p_perp L^2 / (q B0 RE), p_perp = p sin(alpha_eq)"""
    return EPS_DEF_SCALE * 3.0 * pc_mev(ke_mev) * SIN_ALPHA * L ** 2 / P_SCALE_MEV

def L_at_eps(ke_mev, eps):
    return np.sqrt(eps * P_SCALE_MEV / (3.0 * EPS_DEF_SCALE * SIN_ALPHA * pc_mev(ke_mev)))

# ------------------------------------------------------------ load the runs ---
import argparse
_ap = argparse.ArgumentParser(description="epsilon map of mu-conservation verdicts")
_ap.add_argument("csv", help="master_simulation_log CSV (or an FMA per-run summary CSV)")
_ap.add_argument("--eps-l", "--eps-b", "--eps-x", "--eps-c", dest="eps_l", type=float,
                 default=None,
                 help="override eps_l instead of letting the lost runs set it")
_ap.add_argument("--lost-cls", default="CHAOTIC_LOST",
                 help="comma-separated classes that count as hitting the "
                      "atmosphere; the lowest-eps run among them sets the red "
                      "curve (default CHAOTIC_LOST; e.g. CHAOTIC_LOST,SEA)")
_ap.add_argument("--launch-phi", type=float, default=None,
                 help="launch gyrophase (deg) for the Stormer curve; default: "
                      "the CSV's phi_deg, else 0")
_ap.add_argument("--out", default=None, help="output png (default epsilon_map_a<pitch>.png)")
_ap.add_argument("--title", default=None, help="optional figure title (default: none)")
_ap.add_argument("--height", type=float, default=5.4,
                 help="figure height in inches at 7.2 in wide (default 5.4; "
                      "use 4.0 to stack three panels)")
_ap.add_argument("--label-L", type=float, default=6.1,
                 help="(unused since labels moved to the right margin; kept so old commands still run)")
_ap.add_argument("--regimes", action=argparse.BooleanOptionalAction, default=SHOW_REGIMES,
                 help="draw the manual launch-fate regime lines (REGIME_LINES) for this "
                      "pitch angle; --no-regimes hides them")
_ap.add_argument("--legend", action=argparse.BooleanOptionalAction, default=True,
                 help="show the legend (default); --no-legend hides it. The pitch-angle "
                      "label in the corner stays either way")
_args = _ap.parse_args()
csv_path = Path(_args.csv)

runs = []          # (L, E_MeV, alpha_deg, eps_measured, cls)
phis = set()       # launch gyrophases, where the CSV records them
with open(csv_path) as f:
    reader = csv.DictReader(f)
    cols = set(reader.fieldnames)
    for row in reader:
        if "eps_init_est" in cols:
            # FMA per-run summary (e.g. fma_1e8_line_runs.csv)
            m = re.search(r"E=([\d.eE+-]+)\s+L=[\d.]+\s+a=([\d.]+)", row["run"])
            runs.append((float(row["L"]), float(m.group(1)) / 1e6,
                         float(m.group(2)), float(row["eps_init_est"]),
                         row["cls"].strip().upper()))
        elif "eps_initial" in cols:
            # master simulation log (e.g. master_simulation_log_protons.csv).
            # Only PS rows carry the Dragt/eps columns; RK rows of a mixed log
            # would otherwise be mis-classified and drawn twice.
            if row.get("method", "PS").strip().upper() != "PS":
                continue
            # Older rows carry 1-2 extra fields, shifting later columns right;
            # the real hit_atmosphere boolean is the True/False landmark, and
            # eps_initial is trusted only for unshifted rows.
            hit, shifted = None, True
            for c in ("hit_atmosphere", "hit_atm_r", "n_mirror_crossings",
                      "bounce_period_s"):
                if row[c].strip() in ("True", "False"):
                    hit = row[c].strip() == "True"
                    shifted = c != "hit_atmosphere"
                    break
            cls = row["orbit_character"].strip().upper()
            if cls == "CHAOTIC" and hit:
                cls = "CHAOTIC_LOST"
            if row["boundary"].strip().upper() == "OPEN":
                cls = "UNTRAPPED"
            eps_m = None if shifted else float(row["eps_initial"])
            phis.add(float(row["phi_deg"]))
            runs.append((float(row["L_eff"]), float(row["energy_eV"]) / 1e6,
                         float(row["pitch_deg"]), eps_m, cls))
        else:
            raise SystemExit(f"unrecognized CSV format: columns {sorted(cols)}")

# calibrate the analytic epsilon to the runs' own definition
# pitch angle first, so the analytic eps carries sin(alpha_eq) and the
# calibration factor below stays ~1 at every pitch (it was 0.98*sin(alpha),
# which at 30 deg fell under the 0.5 sanity floor and emptied `sane`)
_alphas0 = sorted({a for _L, _e, a, _m, _c in runs})
SIN_ALPHA = float(np.sin(np.radians(_alphas0[0])))
ratios = [eps_m / epsilon(e, L) for L, e, _a, eps_m, _c in runs
          if eps_m is not None]
sane = [r for r in ratios if 0.5 < r < 2.0]
if not sane:
    raise SystemExit(f"calibration failed: no usable eps_initial values (ratios: "
                     f"{sorted(ratios)[:3]} ... ) — check the CSV's eps_initial column")
EPS_DEF_SCALE = float(np.median(sane))
print(f"EPS_DEF_SCALE calibrated to {EPS_DEF_SCALE:.4f} "
      f"(spread {min(sane):.4f}-{max(sane):.4f} over {len(sane)} runs)")
n_off = sum(1 for r in ratios if not 0.95 < r / EPS_DEF_SCALE < 1.05)
if n_off:
    print(f"WARNING: {n_off} rows have eps_initial >5% off the p*L^2 law "
          f"(check those rows' eps/L_eff — possible stale or corrupt entries)")

# chaos threshold for this pitch angle
_alphas = sorted({a for _L, _e, a, _m, _c in runs})
if len(_alphas) > 1:
    print(f"WARNING: CSV mixes pitch angles {_alphas}; EPS_DEF_SCALE and {EPS_NAME} "
          "assume a single alpha_eq — filter the CSV to one pitch first")
ALPHA = _alphas[0]

# The red curve is the first transition into atmosphere-hitting runs: the runs
# set the line. Every lost run has eps = 3 p L^2 sin(alpha) / (q B0 RE), so the
# smallest eps among them is the earliest crossing, whichever energy it sits at
# (a low-E run at large L and a high-E run at small L compete on equal footing).
# The iso-eps curve through that run is drawn in red.
LOST_CLS = {c.strip().upper() for c in _args.lost_cls.split(",") if c.strip()}
lost = [(epsilon(e, L), e, L) for L, e, _a, _m, c in runs if c in LOST_CLS]
if _args.eps_l is not None:
    EPS_L, EPS_L_NOTE = _args.eps_l, "set on the command line"
elif lost:
    EPS_L, _e_set, _L_set = min(lost)
    EPS_L_NOTE = (f"first {'/'.join(sorted(LOST_CLS))} run, "
                  f"E={_e_set:g} MeV at L={_L_set:g}")
else:
    EPS_L = EPS_L_85 * (np.sin(np.radians(ALPHA)) / np.sin(np.radians(85.0))) ** 2
    EPS_L_NOTE = ("measured" if abs(ALPHA - 85.0) < 0.5
                  else rf"Dragt-scaled from 0.457 at $85^\circ$ by $\sin^2\alpha$")
    print(f"NOTE: no {'/'.join(sorted(LOST_CLS))} runs in this CSV — the red "
          f"curve falls back to the analytic {EPS_NAME} instead of being set by the "
          "runs (pass --lost-cls to widen which classes count as lost)")
print(f"alpha_eq = {ALPHA:g} deg -> {EPS_NAME} = {EPS_L:.3f} ({EPS_L_NOTE})")

# Stormer trapping boundary. With P_phi conserved, the meridional motion sees
# V_eff = (P_phi/rho + rho/r^3)^2 / 2 (Dragt units), whose saddle on the
# equator at rho = 2/|P_phi| is the neck between the trapping region and
# infinity. The region is closed iff W0 < P_phi^2/4 (compute_params in
# ps_method/dipoleb_dragt_analysis.py). For an equatorial launch at r = L,
# W0 = p L^2/(q B0 RE) and P_phi = W0 s - 1 with s = v_phi/v =
# sin(alpha) sin(phi), so closure is W0 < w_st(s) = ((sqrt(1+s) - 1)/s)^2
# (1/4 at s = 0). The dipole has one length scale, so this is again a
# p L^2 = const curve: an iso-eps line at eps_st = 3 sin(alpha) w_st, in the
# same units as the runs' eps_initial (r_g |grad B|/B at the equator).
if _args.launch_phi is not None:
    PHI = _args.launch_phi
elif phis:
    if len(phis) > 1:
        print(f"WARNING: CSV mixes launch gyrophases {sorted(phis)}; the Stormer "
              "curve depends on phi — using the first (pass --launch-phi)")
    PHI = sorted(phis)[0]
else:
    PHI = 0.0
_s = np.sin(np.radians(ALPHA)) * np.sin(np.radians(PHI))
W_ST = 0.25 if abs(_s) < 1e-12 else ((np.sqrt(1.0 + _s) - 1.0) / _s) ** 2
EPS_ST = 3.0 * np.sin(np.radians(ALPHA)) * W_ST
print(f"Stormer boundary: launch phi = {PHI:g} deg -> W0_st = {W_ST:.4f}, "
      f"eps_st = {EPS_ST:.3f}  (runs above it are untrapped)")
if lost:
    _by_E = {}
    for _eps, _e, _L in lost:
        if _e not in _by_E or _eps < _by_E[_e][0]:
            _by_E[_e] = (_eps, _L)
    print("first lost run per energy (E MeV: eps @ L):",
          {e: (round(v[0], 3), v[1]) for e, v in sorted(_by_E.items())})

# ------------------------------------------------------------------- figure ---
# palette: keep the run-classification colors used in the paper's other figures
C_TORUS = "#2b5f8a"   # blue   - mu bounded for all time
C_WOBBL = "#E08A00"   # gold-orange - mixed zone (lighter than crimson, CVD-separable)
# CHAOTIC runs are currently drawn in the trapped color, so they read as one
# population with the blue points; they are still classified separately
# everywhere else. Set this back to C_WOBBL to bring the gold markers back.
C_CHAOTIC = C_TORUS
C_SEA   = "#8f1f3f"   # dark red - mu walks
INK     = "#333333"
MUTED   = "#8a8a8a"

fig, ax = plt.subplots(figsize=(7.2, _args.height), dpi=200)

U = 1e6   # display unit: internal MeV -> eV on the axes

E_lo = 10 ** np.floor(np.log10(min(e for _l, e, _a, _m, _c in runs)))
E_hi = 10 ** np.ceil(np.log10(max(e for _l, e, _a, _m, _c in runs)))
E_lo = min(E_lo, 0.1)      # always span the standard survey range (1e5-1e9 eV),
E_hi = max(E_hi, 1000.0)   # so a partial sweep is drawn on the same axes
E = np.logspace(np.log10(E_lo * 0.6), np.log10(E_hi * 1.8), 400)

# threshold curve
Lc = L_at_eps(E, EPS_L)
ax.plot(Lc, E * U, color=C_SEA, lw=EPS_L0_LW, ls=EPS_L0_LS, zorder=4)

# Stormer trapping boundary: above it the particle is not confined at all
L_st = L_at_eps(E, EPS_ST)
ax.plot(L_st, E * U, color=INK, lw=1.6, zorder=4)

# manual launch-fate regime lines (toggle: --regimes / --no-regimes)
regime_handles = []
regime_margin = []          # (eps, text, colour) -> right-margin labels, added below
regime_eps_drawn = []       # every regime eps actually drawn (margin label or not)
if _args.regimes:
    _key = next((a for a in REGIME_LINES if abs(a - ALPHA) < 0.5), None)
    if _key is None:
        print(f"regime lines: none defined for alpha_eq = {ALPHA:g} deg "
              f"(REGIME_LINES has {sorted(REGIME_LINES)}); nothing drawn")
    else:
        for _eps, _sym, _lab, _col, _ls, _marg in REGIME_LINES[_key]:
            ax.plot(L_at_eps(E, _eps), E * U, color=_col, lw=REGIME_LW, ls=_ls, zorder=4)
            regime_eps_drawn.append(_eps)
            regime_handles.append((_eps, plt.Line2D([], [], color=_col, lw=REGIME_LW, ls=_ls,
                                             label=rf"${_sym}$: {_lab}")))   # no value: one legend can serve several pitch-angle panels
            if _marg:
                regime_margin.append((_eps, rf"${_sym} = {_eps:g}$", _col))
        print(f"regime lines drawn for {_key} deg:",
              [e for e, *_ in REGIME_LINES[_key]])

# iso-epsilon guides, each labeled inline where it crosses the axes
def ke_at(eps, L):
    """kinetic energy (MeV) on the iso-epsilon curve at shell L"""
    pc = eps * P_SCALE_MEV / (3.0 * EPS_DEF_SCALE * SIN_ALPHA * L ** 2)
    return np.sqrt(pc ** 2 + MP_MEV ** 2) - MP_MEV

y_lo, y_hi = E_lo * 0.6, E_hi * 1.8
# Curve labels sit in the RIGHT MARGIN (outside the axes), each at the height
# where its curve leaves the plot at the right edge, L = X_MAX. Placed after the
# final layout (below) so they can be spread apart when two curves exit close.
X_MAX = 9.0
_margin_labels = []          # (eps, text, colour)
_specials = [EPS_L, EPS_ST] + [e for e, _t, _c in regime_margin] + \
            [h_eps for h_eps in regime_eps_drawn]
for eps, lab in [(0.05, r"$\varepsilon=0.05$"), (0.1, r"$\varepsilon=0.1$"),
                 (0.2, r"$\varepsilon=0.2$"), (0.3, r"$\varepsilon=0.3$")]:
    _near = [sp for sp in _specials if abs(sp - eps) / eps < GUIDE_DROP_REL]
    if _near:
        print(f"grey guide eps={eps:g} dropped: special line at eps={_near[0]:.3g} is within "
              f"{GUIDE_DROP_REL:.0%}")
        continue
    Li = L_at_eps(E, eps)
    ax.plot(Li, E * U, color=MUTED, lw=0.9, ls=(0, (4, 3)), zorder=3)
    _margin_labels.append((eps, lab, MUTED))
# threshold curves get margin labels too. Stormer is labeled by name, not by eps:
# its condition is gamma = -1 (for a phi = 0 launch, r = C_st/2 at any pitch
# angle); the eps value is only its image on this map (printed as eps_st).
_margin_labels += [(EPS_L, rf"${EPS_SYMBOL} = {EPS_L:.3f}$", C_SEA),
                   (EPS_ST, "Störmer limit", INK)] + regime_margin

# overlay every run from the CSV at its (L, E); color/shape = verdict
STYLE = {"TORUS":        (C_TORUS, "o", "torus"),
         "LOOP_WOBBLY":  (C_WOBBL, "D", "librating"),
         "SEA":          (C_SEA,   "s", "sea"),
         "REGULAR":      (C_TORUS, "o", "stably trapped"),
         "CHAOTIC":      (C_CHAOTIC, "o", "chaotic (Dragt & Finn 1976 criterion)"),
         "CHAOTIC_LOST": (C_SEA,   "o", "lost to atmosphere"),
         "UNTRAPPED":    (INK,     "x", "untrapped")}

E_sweep = sorted({e for _L, e, _a, _m, _c in runs})
alpha_set = sorted({a for _L, _e, a, _m, _c in runs})
# (sweep description footer removed — stated in the figure caption instead)

def mark_kw(col, mk, ms=5.5):
    if mk == "x":
        return dict(marker=mk, ms=ms + 1, mfc="none", mec=col, mew=2.0)
    return dict(marker=mk, ms=ms, mfc=col, mec="none")

for L_r, e_r, _a, _m, cls in runs:
    col, mk, _ = STYLE[cls]
    ax.plot(L_r, e_r * U, ls="none", zorder=6, alpha=0.95, **mark_kw(col, mk))
# Drawn and classified as usual, just not given its own legend row: while
# CHAOTIC shares the trapped color it would be an indistinguishable swatch.
# Empty this set to list it separately again.
LEGEND_HIDE = {"CHAOTIC"}
present = sorted({cls for _L, _e, _a, _m, cls in runs} - LEGEND_HIDE,
                 key=list(STYLE).index)
# pitch angle: its own label in the top-right corner, independent of the legend,
# so it stays when the legend is hidden (--no-legend) or shared across panels
ax.text(0.99, 0.99, rf"$\alpha_0 = {ALPHA:g}$°", transform=ax.transAxes,
        ha="right", va="top", fontsize=10.5, color=INK, zorder=7)
_leg_handles = ([plt.Line2D([], [], ls="none", label=STYLE[c][2],
                        **mark_kw(STYLE[c][0], STYLE[c][1], ms=6.5))
             for c in present] + [h for _e, h in sorted(regime_handles + [(EPS_L,
                 plt.Line2D([], [], color=C_SEA, lw=EPS_L0_LW, ls=EPS_L0_LS,
                            label=rf"${EPS_SYMBOL}$: first loss, $\varphi_0=0$ launches"))],
                 key=lambda t: t[0])])
if _args.legend:
    # no title and no values, so one legend can serve several pitch-angle panels;
    # sits just under the pitch-angle label in the top-right corner
    leg = ax.legend(handles=_leg_handles,
                    loc="upper right", bbox_to_anchor=(0.99, 0.965), bbox_transform=ax.transAxes,
                    frameon=False, fontsize=9, ncol=1)

# consistency check: does the red curve separate lost from not-lost? It is
# drawn through the first lost run, so every lost run should sit at or above it
# (>=, not >: the run that sets the curve lies exactly on it).
mis = [(L, e, cls) for L, e, _a, _m, cls in runs
       if (cls in LOST_CLS) != (epsilon(e, L) >= EPS_L)]
print(f"runs on the 'wrong' side of the {'/'.join(sorted(LOST_CLS))} "
      f"threshold: {len(mis)}", mis or "")

mis_st = [(L, e, cls) for L, e, _a, _m, cls in runs
          if (cls == "UNTRAPPED") != (epsilon(e, L) >= EPS_ST)]
print(f"runs on the 'wrong' side of the Stormer boundary: {len(mis_st)}",
      mis_st or "")

# formula annotation

ax.set_yscale("log")
ax.set_xlim(1.5, X_MAX)
ax.set_ylim(E_lo * 0.6 * U, E_hi * 1.8 * U)
ax.set_xlabel("$L$")
ax.set_ylabel("$E$ (eV)")
if _args.title:
    ax.set_title(_args.title, fontsize=11, loc="left", pad=10)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.tick_params(colors=INK)
ax.grid(True, which="major", color="#dddddd", lw=0.5, zorder=0)

fig.tight_layout()
# right-margin labels: y where each curve crosses L = X_MAX, spread apart so
# neighbours never overlap (min gap in display pixels), skipped if off-scale
_ymin, _ymax = ax.get_ylim()
_pts = []
for eps, lab, col in _margin_labels:
    yv = ke_at(eps, X_MAX) * U
    if _ymin < yv < _ymax:
        _pts.append([ax.transData.transform((X_MAX, yv))[1], lab, col])
_pts.sort(key=lambda t: t[0])
_gap = 11.0 * fig.dpi / 72.0          # ~11 pt between label centres
for k in range(1, len(_pts)):
    _pts[k][0] = max(_pts[k][0], _pts[k - 1][0] + _gap)
_tr = ax.transAxes.inverted()
for ypix, lab, col in _pts:
    yax = _tr.transform((0, ypix))[1]
    ax.text(1.015, yax, lab, color=col, fontsize=8.5, ha="left", va="center",
            transform=ax.transAxes, clip_on=False)
_out = _args.out or f"epsilon_map_b_a{ALPHA:.0f}.png"
fig.savefig(_out, bbox_inches="tight")
print("wrote", _out)
print("epsilon at (100 MeV, L=3.0, 4.5, 4.6):",
      [round(float(epsilon(100, l)), 3) for l in (3.0, 4.5, 4.6)])
print("L_c at 10, 30, 100, 300 MeV:",
      [round(float(L_at_eps(e, EPS_L)), 2) for e in (10, 30, 100, 300)])
print("L_st at 10, 30, 100, 300, 1000 MeV:",
      [round(float(L_at_eps(e, EPS_ST)), 2) for e in (10, 30, 100, 300, 1000)])
