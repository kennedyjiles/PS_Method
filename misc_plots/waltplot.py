import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator
from matplotlib.lines import Line2D
from matplotlib.ticker import LogLocator, LogFormatterSciNotation, NullFormatter, FuncFormatter
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from functions.functions_library_universal import plt_config

plt_config(scale=1)

# --- Input/output setup ---
csv_path = "L_energy_table_p.csv"   # <== your input CSV
csv_basename = os.path.splitext(os.path.basename(csv_path))[0]
out_file = f"{csv_basename}.png"    # Output will auto-match CSV name
print(f"Generating plot from {csv_path} → {out_file}")


def parse_energy_label(label: str) -> float:
    """
    Convert labels like "10^6 eV" -> 1e6 (float).
    Falls back to NaN if unparsable.
    """
    try:
        base = label.split()[0]  # e.g. "10^6"
        # "10^k" -> "1e{k}"
        if "^" in base and base.startswith("10"):
            k = base.split("^", 1)[1]
            return float(f"1e{k}")
        # If already numeric-like, try direct float
        return float(base)
    except Exception:
        return np.nan


def main(csv_path: str, out_file: str):
    df = pd.read_csv(csv_path)

    # Columns that represent energies on x-axis
    energy_cols = ["10^1 eV","10^2 eV","10^3 eV","10^4 eV",
                   "10^5 eV","10^6 eV","10^7 eV","10^8 eV"]

    # --- Melt to long form ---
    long = df.melt(
        id_vars=["Method", "L"],
        value_vars=energy_cols,
        var_name="energy_label",
        value_name="time"
    )

    # Parse energy to numeric (eV)
    long["energy_eV"] = long["energy_label"].map(parse_energy_label)

    # --- Tag helpers ---
    long["is_analytical"] = long["Method"].astype(str).str.startswith("Analytical")
    long["is_ps"] = long["Method"].astype(str).str.startswith("PS Method")
    long["is_bounce"] = long["Method"].astype(str).str.contains("Bounce", case=False, na=False)

    # --- Clean for log plotting: must be finite and > 0 ---
    long = long.replace([np.inf, -np.inf], np.nan)
    long = long.dropna(subset=["time", "energy_eV", "L", "Method"])
    long = long[(long["time"] > 0) & (long["energy_eV"] > 0)]

    # If nothing left after cleaning, bail early
    if long.empty:
        raise ValueError("No positive, finite data left to plot on log axes.")

    # --- Color by L value ---
    L_values = sorted(long["L"].unique())
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    if not color_cycle:
        color_cycle = ["C0","C1","C2","C3","C4","C5","C6","C7","C8","C9"]
    color_map = {L: color_cycle[i % len(color_cycle)] for i, L in enumerate(L_values)}

    fig, ax = plt.subplots(figsize=(9, 7))

    # --- Plot PS curves first (solid), label each L once (prefer the 'bounce' subset if present) ---
    handles, labels = [], []

    # Group PS by L, then within each L prefer the is_bounce subgroup for labeling
    for L, grpL in long[long["is_ps"]].groupby("L"):
        grpL = grpL.sort_values(["is_bounce", "energy_eV"])  # ensures stable plotting order

        # If there is a bounce subset for this L, plot+label it; otherwise label the first PS subgroup encountered
        has_bounce = grpL["is_bounce"].any()

        for is_bounce, grp in grpL.groupby("is_bounce"):
            grp = grp.sort_values("energy_eV")
            line = ax.loglog(grp["energy_eV"], grp["time"],
                             "-", color=color_map[L], alpha=0.9)[0]

            if (has_bounce and is_bounce) or (not has_bounce and len([lab for lab in labels if lab == f"L={L}"]) == 0):
                handles.append(line)
                labels.append(f"L={L}")

    # --- Plot Analytical on top (dashed black) ---
    for (_, L), grp in long[long["is_analytical"]].groupby(["Method", "L"]):
        grp = grp.sort_values("energy_eV")
        ax.loglog(grp["energy_eV"], grp["time"], "--", color="black", alpha=0.75)

    # Single legend entry for Analytical
    analytical_handle = Line2D([0], [0], color="black", linestyle="--", alpha=0.75)
    handles.append(analytical_handle)
    labels.append("Analytical")

    # Legend
    ax.legend(handles, labels, fontsize=9, ncol=2)

    # --- Axes styling & grid ---
    ax.set_axisbelow(True)

    # Decade majors and 2–9 minors; lots of numticks prevents pruning on large ranges
    maj = LogLocator(base=10.0, subs=(1.0,), numticks=1000)
    minr = LogLocator(base=10.0, subs=tuple(range(2, 10)), numticks=1000)
    ax.xaxis.set_major_locator(maj)
    ax.xaxis.set_minor_locator(minr)
    ax.yaxis.set_major_locator(maj)
    ax.yaxis.set_minor_locator(minr)

    ax.xaxis.grid(True, which="major", linestyle="-", linewidth=0.8, alpha=0.6, color="0.45")
    ax.xaxis.grid(True, which="minor", linestyle="-", linewidth=0.5, alpha=0.35, color="0.75")
    ax.yaxis.grid(True, which="major", linestyle="-", linewidth=0.8, alpha=0.6, color="0.45")
    ax.yaxis.grid(True, which="minor", linestyle="-", linewidth=0.5, alpha=0.35, color="0.75")

    # Labels & ticks
    ax.set_xlabel("Energy (eV)")
    ax.set_ylabel("Period (s)")
    ax.tick_params(axis="both", which="both", length=0)

    # --- Let Matplotlib set limits, then add breathing room so the lowest curve isn't 'cut off' ---
    ax.relim()
    ax.autoscale(enable=True, axis="both", tight=False)
    ax.margins(x=0.05, y=0.10)  # ~5% x, ~10% y padding

    ax.text(2e6, 4e-2, "Bounce Period", ha="left", va="center", fontsize=14)
    ax.text(2e6, 1e4, "Drift Period", ha="left", va="center", fontsize=14)

    fig.tight_layout()
    fig.savefig(out_file, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main(csv_path, out_file)
