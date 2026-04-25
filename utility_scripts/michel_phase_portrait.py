#!/usr/bin/env python3
"""
Michel Phase Portrait
=====================
Produces α (pitch angle) vs φ (gyrophase) plots at equatorial crossings,
following Michel (1971) "Permanent Trapping" Fig. 1 and Fig. 2.

At each z=0 crossing, the script records:
  - α = equatorial pitch angle = arctan(v_perp / v_parallel)
  - φ = gyrophase = angle of perpendicular velocity in the (v_rho, v_phi) plane
        Zero phase = particle closest to dipole source (v_rho direction)

Single h5 file:  traces one orbit's path through (φ, α) space (Michel Fig. 1)
Multiple h5 files:  overlays many orbits to build the full phase portrait (Michel Fig. 2)

Usage:
    # Single orbit trace
    python michel_phase_portrait.py outputs/outputs_rawdata/run_abc123.h5

    # Multiple orbits (glob pattern) — builds the phase portrait
    python michel_phase_portrait.py outputs/outputs_rawdata/run_*.h5

    # Specific directory with options
    python michel_phase_portrait.py outputs/outputs_rawdata/*.h5 --chunk-size 500000

    # Filter by energy and L-shell from multiple files
    python michel_phase_portrait.py outputs/outputs_rawdata/*.h5 --energy 10e6 --L 3.0
"""

# python michel_phase_portrait.py outputs/outputs_rawdata/run_abc123.h5 --lines  # optional

import sys
import os
import json
import glob
import argparse
import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# ═══════════════════════════════════════════════════════════════════
# Core physics: extract (α, φ) at equatorial crossings
# ═══════════════════════════════════════════════════════════════════

def find_equatorial_crossings_chunk(z_chunk, x_chunk, y_chunk, vx_chunk, vy_chunk, vz_chunk,
                                    z_prev=None, x_prev=None, y_prev=None,
                                    vx_prev=None, vy_prev=None, vz_prev=None):
    """
    Find z=0 crossings within a chunk, including the boundary with the previous chunk.

    Returns arrays: (x_cross, y_cross, vx_cross, vy_cross, vz_cross)
    """
    results = {k: [] for k in ["x", "y", "vx", "vy", "vz"]}

    # Handle boundary between previous chunk and this chunk
    if z_prev is not None and z_prev * z_chunk[0] < 0:
        t_frac = (0.0 - z_prev) / (z_chunk[0] - z_prev)
        results["x"].append(x_prev + t_frac * (x_chunk[0] - x_prev))
        results["y"].append(y_prev + t_frac * (y_chunk[0] - y_prev))
        results["vx"].append(vx_prev + t_frac * (vx_chunk[0] - vx_prev))
        results["vy"].append(vy_prev + t_frac * (vy_chunk[0] - vy_prev))
        results["vz"].append(vz_prev + t_frac * (vz_chunk[0] - vz_prev))

    # Find sign changes within the chunk
    mask = z_chunk[1:] * z_chunk[:-1] < 0
    idx = np.where(mask)[0]

    if len(idx) > 0:
        t_frac = (0.0 - z_chunk[idx]) / (z_chunk[idx + 1] - z_chunk[idx])
        results["x"].append(x_chunk[idx] + t_frac * (x_chunk[idx + 1] - x_chunk[idx]))
        results["y"].append(y_chunk[idx] + t_frac * (y_chunk[idx + 1] - y_chunk[idx]))
        results["vx"].append(vx_chunk[idx] + t_frac * (vx_chunk[idx + 1] - vx_chunk[idx]))
        results["vy"].append(vy_chunk[idx] + t_frac * (vy_chunk[idx + 1] - vy_chunk[idx]))
        results["vz"].append(vz_chunk[idx] + t_frac * (vz_chunk[idx + 1] - vz_chunk[idx]))

    # Concatenate
    out = {}
    for k in results:
        pieces = results[k]
        if len(pieces) == 0:
            out[k] = np.array([])
        else:
            out[k] = np.concatenate([np.atleast_1d(p) for p in pieces])

    return out["x"], out["y"], out["vx"], out["vy"], out["vz"]


def crossings_to_alpha_phi(x_cross, y_cross, vx_cross, vy_cross, vz_cross):
    """
    Convert equatorial crossing data to Michel coordinates (α, φ).

    α = equatorial pitch angle (degrees, 0-180)
        = arctan(v_perp / |v_parallel|) where v_parallel is along B (z-direction at equator)

    φ = gyrophase (degrees, 0-360)
        = angle of v_perp in the (radially inward, azimuthal) plane
        Convention: φ=0 when particle is closest to dipole (v_rho pointing inward)
        Following Michel: "Zero phase is taken to be the time when the particle
        is closest to the dipole source."
    """
    if len(x_cross) == 0:
        return np.array([]), np.array([])

    rho = np.sqrt(x_cross**2 + y_cross**2)

    # Decompose velocity into cylindrical components
    v_rho = (x_cross * vx_cross + y_cross * vy_cross) / rho
    v_phi = (x_cross * vy_cross - y_cross * vx_cross) / rho
    v_z   = vz_cross

    # v_perp = sqrt(v_rho² + v_phi²)  (perpendicular to B, which is along z at equator)
    v_perp = np.sqrt(v_rho**2 + v_phi**2)

    # Pitch angle: angle between velocity and B-field (z-axis at equator)
    # α = arctan(v_perp / |v_z|)
    alpha = np.degrees(np.arctan2(v_perp, np.abs(v_z)))

    # Gyrophase: angle of v_perp in the (v_rho, v_phi) plane
    # Michel convention: φ=0 when closest to dipole → v_rho pointing radially inward (negative)
    # arctan2(v_phi, -v_rho) gives 0 when v_rho is most negative (closest approach)
    gyrophase = np.degrees(np.arctan2(v_phi, -v_rho)) % 360.0

    return alpha, gyrophase


def process_h5_file(h5_path, chunk_size=1_000_000):
    """
    Read an h5 trajectory file and extract (α, φ) at all equatorial crossings.

    Returns:
        alpha, phi (arrays in degrees)
        meta (dict with energy_eV, L_shell, pitch_deg, etc.)
    """
    meta = {}

    with h5py.File(h5_path, "r") as f:
        # Read metadata
        if "summary_json" in f.attrs:
            summary = json.loads(f.attrs["summary_json"])
            m = summary.get("meta", {})
            meta["energy_eV"] = m.get("energy_eV", None)
            meta["pitch_deg"] = m.get("pitch_deg", None)
            meta["phi_deg"]   = m.get("phi_deg", None)
            meta["L_eff"]     = m.get("x0", None)  # equatorial launch = L
            meta["particle"]  = m.get("particle", "unknown")
            meta["stem"]      = m.get("stem", os.path.basename(h5_path))
        else:
            meta["stem"] = os.path.basename(h5_path)

        # Find the PS trajectory dataset
        if "ps" not in f or "y" not in f["ps"]:
            print(f"  Warning: {h5_path} has no ps/y dataset, skipping.")
            return None, None, meta

        y_ds = f["ps"]["y"]
        n_steps = y_ds.shape[1]

        # Process in chunks to handle large files
        all_alpha = []
        all_phi = []

        z_prev = x_prev = y_prev = vx_prev = vy_prev = vz_prev = None

        for start in range(0, n_steps, chunk_size):
            end = min(start + chunk_size, n_steps)
            chunk = y_ds[:, start:end]  # shape (9, chunk_len): [x,y,z,vx,vy,vz,Bx,By,Bz]

            x_c  = chunk[0]
            y_c  = chunk[1]
            z_c  = chunk[2]
            vx_c = chunk[3]
            vy_c = chunk[4]
            vz_c = chunk[5]

            x_cross, y_cross, vx_cross, vy_cross, vz_cross = \
                find_equatorial_crossings_chunk(
                    z_c, x_c, y_c, vx_c, vy_c, vz_c,
                    z_prev, x_prev, y_prev, vx_prev, vy_prev, vz_prev
                )

            if len(x_cross) > 0:
                alpha, phi = crossings_to_alpha_phi(x_cross, y_cross, vx_cross, vy_cross, vz_cross)
                all_alpha.append(alpha)
                all_phi.append(phi)

            # Save last values for chunk boundary
            z_prev  = z_c[-1]
            x_prev  = x_c[-1]
            y_prev  = y_c[-1]
            vx_prev = vx_c[-1]
            vy_prev = vy_c[-1]
            vz_prev = vz_c[-1]

    if len(all_alpha) == 0:
        return np.array([]), np.array([]), meta

    return np.concatenate(all_alpha), np.concatenate(all_phi), meta


# ═══════════════════════════════════════════════════════════════════
# Plotting
# ═══════════════════════════════════════════════════════════════════

def plot_single_orbit(alpha, phi, meta, save_path=None, show_lines=False):
    """
    Michel Fig. 1 style: single orbit trace through (φ, α) space.
    Points colored by bounce number.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot the trace
    if show_lines:
        ax.plot(phi, alpha, "k--", lw=0.8, alpha=0.4, zorder=1)
    ax.scatter(phi, alpha, c=np.arange(len(phi)), cmap="viridis",
               s=15, zorder=5, edgecolors="none")

    # Number the first few points like Michel
    n_label = min(10, len(phi))
    for i in range(n_label):
        ax.annotate(str(i + 1), (phi[i], alpha[i]),
                     fontsize=8, fontweight="bold",
                     xytext=(4, 4), textcoords="offset points",
                     color="red", clip_on=True)

    ax.set_xlabel(r"Gyrophase $\phi$ (degrees)", fontsize=13)
    ax.set_ylabel(r"Pitch angle $\alpha$ (degrees)", fontsize=13)
    ax.set_xlim(0, 360)
    ax.set_xticks([0, 90, 180, 270, 360])
    ax.grid(True, alpha=0.2)

    E_str = f"{meta.get('energy_eV', 0) / 1e6:.0f} MeV" if meta.get("energy_eV") else ""
    L_str = f"L={meta.get('L_eff', '?')}" if meta.get("L_eff") else ""
    p_str = f"α₀={meta.get('pitch_deg', '?')}°" if meta.get("pitch_deg") else ""
    ax.set_title(f"Michel Phase Trace — {E_str}  {L_str}  {p_str}\n"
                 f"({len(phi)} equatorial crossings)",
                 fontsize=12)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Saved: {save_path}")
    else:
        plt.show()
    plt.close(fig)


def plot_phase_portrait(orbit_data, save_path=None, title_extra=""):
    """
    Michel Fig. 2 style: many orbits overlaid in (φ, α) space.
    Each orbit is colored by its initial pitch angle.

    orbit_data: list of (alpha, phi, meta) tuples
    """
    fig, ax = plt.subplots(figsize=(11, 8))

    # Color by initial pitch angle
    pitches = [d[2].get("pitch_deg", 0) for d in orbit_data]
    pitch_min, pitch_max = min(pitches), max(pitches)

    cmap = plt.cm.viridis
    norm = plt.Normalize(vmin=pitch_min, vmax=pitch_max)

    for alpha, phi, meta in orbit_data:
        if len(alpha) == 0:
            continue
        pitch = meta.get("pitch_deg", 0)
        color = cmap(norm(pitch))

        # Plot points (small, semi-transparent for density)
        ax.scatter(phi, alpha, c=[color], s=3, alpha=0.5,
                   edgecolors="none", zorder=2)

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, label=r"Initial pitch angle $\alpha_0$ (degrees)",
                        shrink=0.8)

    ax.set_xlabel(r"Gyrophase $\phi$ (degrees)", fontsize=13)
    ax.set_ylabel(r"Pitch angle $\alpha$ (degrees)", fontsize=13)
    ax.set_xlim(0, 360)
    ax.set_xticks([0, 90, 180, 270, 360])
    ax.grid(True, alpha=0.15)

    # Get common energy and L from the data
    energies = set(d[2].get("energy_eV", 0) for d in orbit_data)
    Ls = set(d[2].get("L_eff", 0) for d in orbit_data)
    E_str = ", ".join(f"{e/1e6:.0f}" for e in sorted(energies) if e) + " MeV"
    L_str = ", ".join(f"{L:.1f}" for L in sorted(Ls) if L)

    ax.set_title(f"Michel Phase Portrait — {E_str}  L={L_str}\n"
                 f"{len(orbit_data)} orbits, {sum(len(d[0]) for d in orbit_data)} crossings"
                 + (f"\n{title_extra}" if title_extra else ""),
                 fontsize=12)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Saved: {save_path}")
    else:
        plt.show()
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Michel (1971) phase portrait: α vs φ at equatorial crossings",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single orbit trace (Michel Fig. 1)
  python %(prog)s outputs/outputs_rawdata/run_abc123.h5

  # Phase portrait from multiple files (Michel Fig. 2)
  python %(prog)s outputs/outputs_rawdata/run_*.h5

  # Filter to specific energy and L
  python %(prog)s outputs/outputs_rawdata/*.h5 --energy 10e6 --L 3.0
        """)
    parser.add_argument("h5_files", nargs="+",
                        help="H5 trajectory file(s). Supports glob patterns.")
    parser.add_argument("--energy", type=float, default=None,
                        help="Filter to this energy in eV (e.g. 10e6 for 10 MeV)")
    parser.add_argument("--L", type=float, default=None,
                        help="Filter to this L-shell")
    parser.add_argument("--chunk-size", type=int, default=1_000_000,
                        help="H5 read chunk size (default: 1M steps)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output PNG path (default: auto-named)")
    parser.add_argument("--max-crossings", type=int, default=50_000,
                        help="Max crossings per orbit to plot (default: 50000)")
    parser.add_argument("--lines", action="store_true",
                        help="Draw connecting lines between bounces (off by default)")
    args = parser.parse_args()

    # Expand glob patterns
    h5_paths = []
    for pattern in args.h5_files:
        expanded = glob.glob(pattern)
        if expanded:
            h5_paths.extend(expanded)
        elif os.path.exists(pattern):
            h5_paths.append(pattern)
        else:
            print(f"Warning: no match for '{pattern}'")

    h5_paths = sorted(set(h5_paths))

    if not h5_paths:
        print("No h5 files found.")
        sys.exit(1)

    print(f"Processing {len(h5_paths)} h5 file(s)...\n")

    # Process each file
    orbit_data = []
    for path in h5_paths:
        print(f"  {os.path.basename(path)}...", end=" ", flush=True)
        alpha, phi, meta = process_h5_file(path, chunk_size=args.chunk_size)

        if alpha is None or len(alpha) == 0:
            print("no crossings")
            continue

        # Apply filters
        if args.energy is not None and meta.get("energy_eV") is not None:
            if abs(meta["energy_eV"] - args.energy) / args.energy > 0.01:
                print(f"skipped (E={meta['energy_eV']/1e6:.0f} MeV)")
                continue

        if args.L is not None and meta.get("L_eff") is not None:
            if abs(meta["L_eff"] - args.L) > 0.05:
                print(f"skipped (L={meta['L_eff']:.2f})")
                continue

        # Limit crossings for plotting
        if len(alpha) > args.max_crossings:
            stride = len(alpha) // args.max_crossings
            alpha = alpha[::stride]
            phi = phi[::stride]

        E_str = f"{meta.get('energy_eV', 0)/1e6:.0f}MeV" if meta.get("energy_eV") else "?"
        print(f"{len(alpha)} crossings  (E={E_str}, L={meta.get('L_eff', '?')}, "
              f"α₀={meta.get('pitch_deg', '?')}°)")

        orbit_data.append((alpha, phi, meta))

    if not orbit_data:
        print("\nNo valid orbits to plot.")
        sys.exit(1)

    # Decide plot type
    if len(orbit_data) == 1:
        # Single orbit trace (Michel Fig. 1)
        alpha, phi, meta = orbit_data[0]
        save_path = args.output or "michel_phase_trace.png"
        plot_single_orbit(alpha, phi, meta, save_path=save_path,
                          show_lines=args.lines)
    else:
        # Phase portrait (Michel Fig. 2)
        save_path = args.output or "michel_phase_portrait.png"
        plot_phase_portrait(orbit_data, save_path=save_path)


if __name__ == "__main__":
    main()
