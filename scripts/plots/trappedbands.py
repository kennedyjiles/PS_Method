"""
=====================================
eps vs L with three HORIZONTAL zone bands:
  1. Trapped 20+ years     (eps below ~0.8)  — green
  2. Trapped, hits atm     (eps ~0.8 to ~1)  — blue
  3. Untrapped / open      (eps above ~1)    — red

L=1 removed (misleading — surface proximity, not GCA issue).
Dragt REGULAR/CHAOTIC colors the points.

Usage:
    python trappedbands.py path/to/master_simulation_log.csv
"""
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

csv_path = sys.argv[1] if len(sys.argv) > 1 else "master_simulation_log.csv"
# on_bad_lines='warn' skips rows whose column count doesn't match the header,
# printing a warning so we know which rows were dropped.  Defensive: a single
# corrupt row shouldn't prevent the whole plot from rendering.
df = pd.read_csv(csv_path, on_bad_lines='warn')
df["energy_MeV"] = df["energy_eV"] / 1e6
df["L"] = df["L_eff"].round(2)
df["eps_plot"] = df["eps_max"].clip(upper=5.0)
df["mu_max_err_plot"] = df["mu_max_err"].clip(lower=1e-3, upper=10.0)

# ── Remove L=1 ──
df = df[df["L"] > 1.0].copy()

# ── Determine horizontal band boundaries from data ──
# The blue "hits atmosphere" zone spans from the LOWEST atm-hit eps to the
# HIGHEST closed+atm eps.  The green zone is everything below that, red above.
# Note: KAM islands can cause no-atm cases to exist WITHIN the blue range —
# these are stable islands inside the chaotic sea and plot as green dots
# inside the blue band, which is physically correct.
atm_hits = df[(df["hit_atmosphere"] == True)]
closed_atm = df[(df["hit_atmosphere"] == True) & (df["boundary"] == "CLOSED")]
open_cases = df[df["boundary"] == "OPEN"]

eps_bot_atm = None
eps_top_atm = None

if len(atm_hits) > 0:
    eps_bot_atm = atm_hits["eps_plot"].min()
    # Lower boundary: use the highest no-atm case that is BELOW the atm range,
    # then go just above it.  This ensures all no-atm cases near the boundary
    # stay in the green zone.  KAM islands above this are expected (stable
    # orbits inside the chaotic sea) and will appear as green dots in blue.
    trapped_no_atm = df[(df["boundary"] == "CLOSED") & (df["hit_atmosphere"] == False)]
    no_atm_below = trapped_no_atm[trapped_no_atm["eps_plot"] < eps_bot_atm]
    if len(no_atm_below) > 0:
        eps_boundary_1 = no_atm_below["eps_plot"].max() * 1.02  # just above highest no-atm below atm range
    else:
        eps_boundary_1 = eps_bot_atm * 0.95
else:
    # No atmosphere hits — push boundaries above all data so only the green
    # "trapped" band is visible.
    eps_boundary_1 = df["eps_plot"].max() * 1.5 if len(df) > 0 else 1.0

if len(closed_atm) > 0 and len(open_cases) > 0:
    eps_top_atm = closed_atm["eps_plot"].max()
    eps_bot_open = open_cases["eps_plot"].min()
    eps_boundary_2 = np.sqrt(eps_top_atm * eps_bot_open)
elif len(closed_atm) > 0:
    eps_top_atm = closed_atm["eps_plot"].max()
    eps_boundary_2 = eps_top_atm * 1.2
elif len(open_cases) > 0:
    eps_boundary_2 = open_cases["eps_plot"].min() * 0.95
else:
    # No atmosphere, no open — push above all data
    eps_boundary_2 = eps_boundary_1 * 1.5

print(f"Zone boundaries: trapped < {eps_boundary_1:.2f} < atmosphere < {eps_boundary_2:.2f} < open")
print(f"  (atm min: {eps_bot_atm if eps_bot_atm is not None else 'N/A'}, "
      f"atm max (closed): {eps_top_atm if eps_top_atm is not None else 'N/A'})")

# ── Point colors (Dragt classification) ──
zone_colors = {"REGULAR": "#2ca02c", "CHAOTIC": "#ff7f0e"}
zone_labels_pts = {
    "REGULAR": r"Regular ($W_0^2 < 0.012\mu^2$)",
    "CHAOTIC": r"Chaotic ($W_0^2 > 0.012\mu^2$)",
}

# ══════════════════════════════════════════════════════════════════
# MAIN PLOT
# ══════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(11, 7))

y_lo = 0.003
y_hi = 5.0

# ── Three horizontal band fills ──
ax.axhspan(y_lo, eps_boundary_1, color="#d4edda", alpha=0.45, zorder=0)
ax.axhspan(eps_boundary_1, eps_boundary_2, color="#cce5ff", alpha=0.45, zorder=0)
ax.axhspan(eps_boundary_2, y_hi, color="#f8d7da", alpha=0.45, zorder=0)

# (zone boundaries are shown by color change only — no dashed lines)

# ── Adiabaticity reference lines from literature ──
ax.axhline(0.05, color="0.3", ls="--", lw=1.2, alpha=0.5)
ax.text(9.7, 0.05, r"$\epsilon = 0.05$" "\n" r"Borovsky (2014)",
        fontsize=8, color="0.3", va="center", ha="left", fontstyle="italic", clip_on=False)
ax.axhline(1.0, color="0.3", ls="--", lw=1.2, alpha=0.5)
ax.text(9.7, 1.0, r"$\epsilon = 1.0$" "\n" r"GCA meaningless",
        fontsize=8, color="0.3", va="center", ha="left", fontstyle="italic", clip_on=False)

# ── Zone labels inside plot area (right side) ──
ax.text(8.8, np.sqrt(y_lo * eps_boundary_1) * 0.5, "Trapped (20+ years)",
        fontsize=10, color="#1b5e20", va="center", ha="right", fontstyle="italic",
        fontweight="bold", alpha=0.5)
ax.text(8.8, np.sqrt(eps_boundary_1 * eps_boundary_2), "Trapped, hits atm",
        fontsize=10, color="#0d47a1", va="center", ha="right", fontstyle="italic",
        fontweight="bold", alpha=0.5)
ax.text(8.8, np.sqrt(eps_boundary_2 * y_hi), "Untrapped",
        fontsize=10, color="#b71c1c", va="center", ha="right", fontstyle="italic",
        fontweight="bold", alpha=0.5)

# ── Plot data points ──
for zone in ["REGULAR", "CHAOTIC"]:
    sub = df[df["orbit_character"] == zone]
    if sub.empty:
        continue
    # Separate closed vs open boundary
    closed = sub[sub["boundary"] != "OPEN"]
    opened = sub[sub["boundary"] == "OPEN"]

    if not closed.empty:
        ax.scatter(
            closed["L"], closed["eps_plot"],
            c=zone_colors[zone], s=60,
            label=zone_labels_pts[zone],
            edgecolors="none", linewidths=0, alpha=0.85, zorder=5,
        )

    if not opened.empty:
        ax.scatter(
            opened["L"], opened["eps_plot"],
            c="#d62728", s=60,
            label=r"Open boundary ($W_0^2 > P_\phi^4/16$)",
            edgecolors="none", linewidths=0, alpha=0.85, zorder=5,
            marker="D",
        )

# ── Atmosphere X markers ──
atm = df[df["hit_atmosphere"] == True]
if len(atm) > 0:
    ax.scatter(atm["L"], atm["eps_plot"], marker="x", s=40, c="navy",
               linewidths=1.2, zorder=10, label="Hits atmosphere")

# ── Energy series: faint connecting line per energy, labeled once ──
for e_kev, grp in df.groupby("energy_eV"):
    series = (grp.groupby("L")["eps_plot"].mean()
                  .reset_index().sort_values("L"))
    if len(series) < 1:
        continue
    ax.plot(series["L"], series["eps_plot"],
            color="0.4", lw=0.8, alpha=0.35, zorder=2)
    # Label once at the leftmost (lowest-L) point of the series
    x_lbl = series["L"].iloc[0]
    y_lbl = series["eps_plot"].iloc[0]
    label_txt = f"{e_kev:.0e}".replace("e+0", "e").replace("e+", "e").replace("e-0", "e-")
    ax.annotate(label_txt, (x_lbl, y_lbl), fontsize=8,
                ha="right", va="center",
                xytext=(-5, 0), textcoords="offset points",
                color="0.35", fontstyle="italic")

# ── Legend ──
point_handles = []
for zone in ["REGULAR", "CHAOTIC"]:
    point_handles.append(
        Line2D([0], [0], marker="o", color="w", markerfacecolor=zone_colors[zone],
               markeredgecolor="none", markersize=9, label=zone_labels_pts[zone])
    )
if not df[df["boundary"] == "OPEN"].empty:
    point_handles.append(
        Line2D([0], [0], marker="D", color="w", markerfacecolor="#d62728",
               markeredgecolor="none", markersize=9,
               label=r"Open ($W_0^2 > P_\phi^4/16$)")
    )
if len(atm) > 0:
    point_handles.append(
        Line2D([0], [0], marker="x", color="navy", ms=7, lw=1.5, linestyle="None",
               label="Hits atmosphere")
    )

ax.legend(handles=point_handles, loc="lower right", fontsize=8,
         framealpha=0.95, title="Dragt Classification", title_fontsize=9)

ax.set_xlabel("L-shell", fontsize=13)
ax.set_ylabel(r"$\epsilon_{\max}$ (adiabaticity parameter)", fontsize=13)
ax.set_title(
    r"Adiabaticity vs L-shell — Dragt classification with trapping zones"
    "\n" f"({', '.join(f'{p:.0f}°' for p in sorted(df['pitch_deg'].unique()))} equatorial pitch angle, proton)",
    fontsize=12,
)
ax.set_yscale("log")
ax.set_ylim(y_lo, y_hi)
ax.set_xlim(0.8, 9.5)
ax.grid(True, alpha=0.15, which="both")

fig.subplots_adjust(right=0.82)
fig.tight_layout(rect=[0, 0, 0.85, 1])
fig.savefig("walt_eps_vs_L_zones_v3.png", dpi=300, bbox_inches="tight")
print("Saved: walt_eps_vs_L_zones_v3.png")

plt.show()