import pandas as pd
import matplotlib.pyplot as plt
import re
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import os
from matplotlib.ticker import LogLocator, LogFormatterSciNotation, NullFormatter, FuncFormatter
import sys
sys.path.append('/Users/heatherjiles/Documents/GitHub/GradSchool')
from PSMethod.code.definitions.functions_library_universal import plt_config, sparse_labels

# === Load CSV ===
csv_path = "electron_summary_results.csv"
df = pd.read_csv(csv_path)
plt_config(scale=1)

# === Toggle electron/proton  and mu/E data ===
USE_ELECTRON = True  # True for electrons, False for protons
USE_MU = False   # True for Δμ, False for ΔE


# === Toggle which error to plot ===
if USE_MU:
    ERR_COLUMN = "errMu_mean"
    YLABEL = r"$|\Delta \mu|/\mu_0$"
    suffix = "_mu"
else:
    ERR_COLUMN = "errE_mean"
    YLABEL = r"$|\Delta E|/E_0$"
    suffix = "_energy"

base, _ = os.path.splitext(os.path.basename(csv_path))
outputfile = f"{base}{suffix}.png"

# === Keep only needed data ===
df_energy = df[["energy_label", "pitch_deg", "method", ERR_COLUMN, "runtime_s"]].copy()

# Coerce numerics (handles strings like "N/A")
df_energy[ERR_COLUMN] = pd.to_numeric(df_energy[ERR_COLUMN], errors="coerce")
df_energy["runtime_s"] = pd.to_numeric(df_energy["runtime_s"], errors="coerce")
df_energy = df_energy.dropna(subset=[ERR_COLUMN, "runtime_s"])

# Convert energy labels like "10 keV" or "1 MeV" to numeric values
def energy_to_numeric(label: str) -> float:
    m = re.match(r"\s*([0-9.]+)\s*(keV|MeV)\s*$", str(label), re.IGNORECASE)
    if not m:
        return float("nan")
    val = float(m.group(1))
    unit = m.group(2).lower()
    return val * (1e3 if unit == "kev" else 1e6)

df_energy["energy_numeric_eV"] = df_energy["energy_label"].apply(energy_to_numeric)

# === Fixed bubble sizes for four energies ===
if USE_ELECTRON:
    energy_size_map = {
        50e3: 30,    # 50 keV
        1e6: 70,    # 1 MeV
        100e6: 150,   # 100 MeV
        150e6: 270
        }
else:
    energy_size_map = {
        1e4: 30,    # 10 keV
        1e5: 70,    # 100 keV
        1e6: 150,   # 1 MeV
        1e7: 270 
        }


df_energy["size"] = df_energy["energy_numeric_eV"].map(energy_size_map)

# === Scatter plot (runtime on x, error on y) ===
fig, ax = plt.subplots(figsize=(7, 6))

# Explicit hex color map (you can edit these easily)
color_map = {
    "RK4":  "#CC79A7",  # purple
    "RK45": "#E69F00",  # orange
    "RKG":  "#CC0000",  # deep red
    "PS":   "#009E73",  # bluish green
}

for method, df_m in df_energy.groupby("method"):
    color = color_map.get(method, "gray")

    # Pitch = 90 (plain fill)
    df_90 = df_m[df_m["pitch_deg"] == 90]
    if not df_90.empty:
        plt.scatter(
            df_90["runtime_s"], df_90[ERR_COLUMN],
            s=df_90["size"], alpha=0.8, color=color, label=method
        )
    if USE_ELECTRON:
        # Pitch = 60 (black ring)
        df_60 = df_m[df_m["pitch_deg"] == 60]
        if not df_60.empty:
            plt.scatter(
                df_60["runtime_s"], df_60[ERR_COLUMN],
                s=df_60["size"], alpha=0.8,
                facecolors=color, edgecolors="black", linewidths=0.8)    
    else:
        # Pitch = 30 (black ring)
        df_30 = df_m[df_m["pitch_deg"] == 30]
        if not df_30.empty:
            plt.scatter(
                df_30["runtime_s"], df_30[ERR_COLUMN],
                s=df_30["size"], alpha=0.8,
                facecolors=color, edgecolors="black", linewidths=0.8)
# === Log-Log Scale ===


ax.set_xscale("log")
ax.set_yscale('log') 

ax.set_xlabel("Runtime (s)")
ax.set_ylabel(YLABEL)

ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=100))
ax.yaxis.set_major_formatter(LogFormatterSciNotation(base=10.0))  # or LogFormatterMathtext()
ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=[]))
ax.yaxis.set_minor_formatter(NullFormatter())
ax.yaxis.set_major_formatter(FuncFormatter(sparse_labels))
ax.grid(True, which="both", linestyle="--", linewidth=0.7, alpha=0.7)


# === Legend (custom order, circle markers, no lines) ===
method_order = ["RK45", "RK4", "RKG", "PS"]
method_handles = [
    Line2D([0], [0], marker="o", linestyle="None",
           markerfacecolor=color_map[m], markeredgecolor=color_map[m],
           markersize=10, label=m)
    for m in method_order
]
if USE_ELECTRON:
    angle_handles = [
        Line2D([0], [0], marker="o", linestyle="None",
            markerfacecolor="white", markeredgecolor="black",
            markersize=10, label="60° = black ring"),
        Line2D([0], [0], marker="o", linestyle="None",
            markerfacecolor="lightgray", markeredgecolor="lightgray",
            markersize=10, label="90° = plain fill"),
    ]
else:
        angle_handles = [
        Line2D([0], [0], marker="o", linestyle="None",
            markerfacecolor="white", markeredgecolor="black",
            markersize=10, label="30° = black ring"),
        Line2D([0], [0], marker="o", linestyle="None",
            markerfacecolor="lightgray", markeredgecolor="lightgray",
            markersize=10, label="90° = plain fill"),
    ]
ax.legend(handles=method_handles + angle_handles, loc="upper right")

fig.tight_layout()
fig.savefig(outputfile, dpi=600, bbox_inches="tight")
plt.show()
