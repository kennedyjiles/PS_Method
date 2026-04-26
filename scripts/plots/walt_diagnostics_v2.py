"""
Walt Transition Diagnostic Plots v2
=====================================
Goes back to the original eps vs L and multi-panel formats, but adds
three background zones to the eps vs L plot:
  1. Trapped 20+ years (CLOSED + no atmosphere hit)  — green shading
  2. Trapped but hits atmosphere (CLOSED + atm hit)   — blue shading
  3. Untrapped / open boundary                        — red shading

The Dragt REGULAR/CHAOTIC classification colors the points themselves.

Usage:
    python walt_diagnostics_v2.py path/to/master_simulation_log.csv
"""
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib.patheffects as pe

csv_path = sys.argv[1] if len(sys.argv) > 1 else "master_simulation_log.csv"
df = pd.read_csv(csv_path)
df["energy_MeV"] = df["energy_keV"] / 1e6
df["L"] = df["L_eff"].round(2)
df["eps_max_plot"] = df["eps_max"].clip(upper=5.0)
df["mu_max_err_plot"] = df["mu_max_err"].clip(lower=1e-3, upper=10.0)

# Dragt classification for point colors
zone_colors = {"REGULAR": "#2ca02c", "CHAOTIC": "#ff7f0e"}
zone_labels_pts = {
    "REGULAR": r"Regular ($W_0^2 < 0.012\mu^2$)",
    "CHAOTIC": r"Chaotic ($W_0^2 > 0.012\mu^2$)",
}

phi_markers = {-90.0: "v", -45.0: "<", 0.0: "o", 45.0: ">", 90.0: "^"}
phi_labels  = {-90.0: r"$\phi=-90°$ (inward)", -45.0: r"$\phi=-45°$",
               0.0: r"$\phi=0°$", 45.0: r"$\phi=+45°$",
               90.0: r"$\phi=+90°$ (outward)"}

# ── Aggregate per (energy, L) for zone boundaries ──
agg_rows = []
for (e, l), grp in df.groupby(["energy_MeV", "L"]):
    any_atm = grp["hit_atmosphere"].any()
    bnd = grp["boundary"].iloc[0]
    char = grp["orbit_character"].iloc[0]
    if bnd == "OPEN":
        trap_zone = "OPEN"
    elif any_atm:
        trap_zone = "ATM"
    else:
        trap_zone = "TRAPPED"
    agg_rows.append({
        "energy_MeV": e, "L": l,
        "eps_max": grp["eps_max_plot"].max(),
        "trap_zone": trap_zone,
        "orbit_char": char,
    })
agg = pd.DataFrame(agg_rows)


# (PLOT 1 removed — replaced by walt_diagnostics_v3.py)

# ══════════════════════════════════════════════════════════════════
# Multi-panel: per-energy breakdown of L vs diagnostics with Dragt zones
# ══════════════════════════════════════════════════════════════════
energies = sorted(df["energy_MeV"].unique())
n_e = len(energies)

fig2, axes = plt.subplots(2, n_e, figsize=(6 * n_e, 10), sharex=True)
if n_e == 1:
    axes = axes.reshape(2, 1)

for col, energy in enumerate(energies):
    sub = df[df["energy_MeV"] == energy].copy()
    ax_top = axes[0, col]
    ax_bot = axes[1, col]

    # Transition L
    regular_Ls = sub[sub["orbit_character"] == "REGULAR"]["L"].unique()
    chaotic_Ls = sub[sub["orbit_character"] == "CHAOTIC"]["L"].unique()
    transition_L = None
    if len(regular_Ls) > 0 and len(chaotic_Ls) > 0:
        transition_L = (regular_Ls.max() + chaotic_Ls.min()) / 2.0

    # ── Top: epsilon vs L by phi ──
    for phi, marker in phi_markers.items():
        mask = sub["phi_deg"] == phi
        if not mask.any():
            continue
        d = sub[mask].sort_values("L")
        for _, row in d.iterrows():
            color = zone_colors.get(row["orbit_character"], "#d62728")
            if row["boundary"] == "OPEN":
                color = "#d62728"
            ax_top.plot(row["L"], row["eps_max_plot"], marker=marker, ms=8,
                        color=color, alpha=0.85,
                        markeredgecolor="k", markeredgewidth=0.3)
        ax_top.plot(d["L"], d["eps_max_plot"], lw=0.8, color="0.6", alpha=0.4, zorder=1)

    if transition_L is not None:
        ax_top.axvspan(transition_L - 0.3, transition_L + 0.3, color="orange", alpha=0.08)
        ax_top.axvline(transition_L, color="orange", ls="--", lw=1.2, alpha=0.6)
        ax_top.text(transition_L, 3.5, f"Dragt\ntransition\nL≈{transition_L:.0f}",
                    fontsize=8, ha="center", color="darkorange", fontstyle="italic")

    ax_top.set_yscale("log")
    ax_top.set_ylim(0.003, 5.0)
    ax_top.set_title(f"{energy:.0f} MeV proton", fontsize=13, fontweight="bold")
    ax_top.set_ylabel(r"$\epsilon_{\max}$", fontsize=11)
    ax_top.grid(True, alpha=0.3, which="both")
    if col == 0:
        phi_handles = [Line2D([0], [0], marker=m, color="0.5", ms=7, lw=0,
                              markeredgecolor="k", markeredgewidth=0.3, label=phi_labels[p])
                       for p, m in phi_markers.items()]
        ax_top.legend(handles=phi_handles, fontsize=7, loc="upper left", framealpha=0.9)

    # ── Bottom: mu drift vs L by phi ──
    for phi, marker in phi_markers.items():
        mask = sub["phi_deg"] == phi
        if not mask.any():
            continue
        d = sub[mask].sort_values("L")
        for _, row in d.iterrows():
            color = zone_colors.get(row["orbit_character"], "#d62728")
            if row["boundary"] == "OPEN":
                color = "#d62728"
            ax_bot.plot(row["L"], row["mu_max_err_plot"], marker=marker, ms=8,
                        color=color, alpha=0.85,
                        markeredgecolor="k", markeredgewidth=0.3)
        ax_bot.plot(d["L"], d["mu_max_err_plot"], lw=0.8, color="0.6", alpha=0.4, zorder=1)

    if transition_L is not None:
        ax_bot.axvspan(transition_L - 0.3, transition_L + 0.3, color="orange", alpha=0.08)
        ax_bot.axvline(transition_L, color="orange", ls="--", lw=1.2, alpha=0.6)

    # Atmosphere markers
    atm_sub = sub[sub["hit_atmosphere"] == True]
    for _, row in atm_sub.iterrows():
        ax_bot.axvline(row["L"], color="blue", ls=":", lw=0.8, alpha=0.3)
    if len(atm_sub) > 0:
        unique_atm_L = atm_sub["L"].unique()
        for al in unique_atm_L:
            ax_bot.text(al, 8, "atm", fontsize=7, color="blue",
                        ha="center", fontstyle="italic")

    # Phi spread annotations
    for l_val in sub["L"].unique():
        grp = sub[sub["L"] == l_val]
        mu_min = grp["mu_max_err"].min()
        mu_max_v = grp["mu_max_err"].max()
        if mu_min > 0 and mu_max_v / mu_min > 2.0 and l_val > 2:
            ax_bot.annotate(f"{mu_max_v/mu_min:.1f}x", xy=(l_val, min(mu_max_v, 10.0)),
                            fontsize=7, ha="center", va="bottom",
                            xytext=(0, 5), textcoords="offset points",
                            color="0.4", fontstyle="italic")

    ax_bot.set_yscale("log")
    ax_bot.set_ylim(0.003, 10.0)
    ax_bot.set_xlabel("L-shell", fontsize=11)
    ax_bot.set_ylabel(r"$\Delta\mu / \mu_0$ (max)", fontsize=11)
    ax_bot.grid(True, alpha=0.3, which="both")
    if col == 0:
        zone_handles = [Line2D([0], [0], marker="s", color=c, ms=10, lw=0,
                               markeredgecolor="k", markeredgewidth=0.3, label=l)
                        for z, (c, l) in {"REGULAR": (zone_colors["REGULAR"], zone_labels_pts["REGULAR"]),
                                          "CHAOTIC": (zone_colors["CHAOTIC"], zone_labels_pts["CHAOTIC"])}.items()]
        ax_bot.legend(handles=zone_handles, fontsize=7, loc="upper left", framealpha=0.9)

fig2.suptitle(
    r"GCA Breakdown by Dragt Classification: $\epsilon$ and $\mu$ vs L-shell",
    fontsize=14, fontweight="bold", y=1.01,
)
fig2.tight_layout()
fig2.savefig("walt_multipanel_dragt_v2.png", dpi=300, bbox_inches="tight")
print("Saved: walt_multipanel_dragt_v2.png")

plt.show()
