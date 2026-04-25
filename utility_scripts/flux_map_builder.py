#!/usr/bin/env python3
"""
Flux Map Builder
================
Post-processes batch run h5 trajectories into an AP-8-style meridian-plane
flux map: omnidirectional proton flux vs (ρ, z) in Earth radii.

The idea:
  1. For each trapped orbit, compute how much TIME it spends in each (ρ, z) bin
     → this gives the local particle density (dwell time ∝ density)
  2. Weight each orbit by a power-law energy spectrum j(E) ∝ E^(-γ)
     and by sin(α₀) for pitch-angle isotropy
  3. Sum across all orbits → omnidirectional flux map

Usage:
    # Build from all batch h5 files
    python flux_map_builder.py outputs/outputs_rawdata/*.h5

    # With energy range filter (AP-8 style: > 10 MeV)
    python flux_map_builder.py outputs/outputs_rawdata/*.h5 --E-min 10e6

    # Custom grid resolution
    python flux_map_builder.py outputs/outputs_rawdata/*.h5 --bins 100

    # Only process files from the flux_map output folder
    python flux_map_builder.py outputs/outputs_rawdata/*.h5 --output-dir outputs/flux_map
"""

import sys
import os
import json
import glob
import argparse
import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Circle


# ═══════════════════════════════════════════════════════════════════
# Dwell-time accumulation
# ═══════════════════════════════════════════════════════════════════

def accumulate_dwell_time(h5_path, rho_edges, z_edges, chunk_size=1_000_000,
                          exclude_lost=False):
    """
    Read an h5 trajectory and accumulate dwell time in (ρ, z) bins.

    ρ = sqrt(x² + y²)  (cylindrical radius in R_E)
    z = z               (height above equator in R_E)

    Returns:
        hist_2d : 2D array of dwell time per bin (arbitrary time units)
        meta    : dict with run metadata
        success : bool
    """
    meta = {}

    with h5py.File(h5_path, "r") as f:
        # Read metadata
        if "summary_json" in f.attrs:
            summary = json.loads(f.attrs["summary_json"])
            m = summary.get("meta", {})
            meta["energy_eV"]  = m.get("energy_eV", None)
            meta["pitch_deg"]  = m.get("pitch_deg", None)
            meta["phi_deg"]    = m.get("phi_deg", None)
            meta["L_eff"]      = m.get("x0", None)
            meta["particle"]   = m.get("particle", "unknown")
            meta["stem"]       = m.get("stem", os.path.basename(h5_path))
            meta["norm_time"]  = m.get("norm_time", None)

            # Check if PS data exists
            ps_cfg = summary.get("ps", {})
            if not ps_cfg.get("enabled", False):
                return None, meta, False
        else:
            meta["stem"] = os.path.basename(h5_path)

        if "ps" not in f or "y" not in f["ps"]:
            return None, meta, False

        # Check atmosphere flag
        hit_atm = False
        if "ps" in f and "hit_atmosphere" in f["ps"].attrs:
            hit_atm = bool(f["ps"].attrs["hit_atmosphere"])
        meta["hit_atmosphere"] = hit_atm

        if exclude_lost and hit_atm:
            return None, meta, False

        y_ds = f["ps"]["y"]
        n_steps = y_ds.shape[1]

        # Check for time dataset
        has_time = "t" in f["ps"]

        hist_2d = np.zeros((len(rho_edges) - 1, len(z_edges) - 1), dtype=np.float64)

        for start in range(0, n_steps, chunk_size):
            end = min(start + chunk_size, n_steps)
            chunk = y_ds[:, start:end]

            x = chunk[0]
            y = chunk[1]
            z = chunk[2]

            rho = np.sqrt(x**2 + y**2)

            # Read time steps for proper weighting
            if has_time:
                t_chunk = f["ps"]["t"][start:end]
                if len(t_chunk) > 1:
                    # dt per step (use forward difference, last step gets same dt)
                    dt = np.empty_like(t_chunk)
                    dt[:-1] = t_chunk[1:] - t_chunk[:-1]
                    dt[-1] = dt[-2] if len(dt) > 1 else 1.0
                else:
                    dt = np.ones_like(t_chunk)
            else:
                # No time info → uniform weight (fixed step assumed)
                dt = np.ones(end - start, dtype=np.float64)

            # Accumulate weighted histogram
            h, _, _ = np.histogram2d(rho, z, bins=[rho_edges, z_edges], weights=dt)
            hist_2d += h

    return hist_2d, meta, True


def spectrum_weight(energy_eV, spectral_index=2.0, E_ref=10e6):
    """
    Power-law energy spectrum weight: j(E) ∝ (E/E_ref)^(-γ)
    Default γ=2.0 is typical for inner belt protons.
    """
    return (energy_eV / E_ref) ** (-spectral_index)


def pitch_angle_weight(pitch_deg, delta_alpha_deg=None):
    """
    Weight for omnidirectional flux integration: dΩ = 2π sin(α) dα.

    If delta_alpha_deg is provided, the weight is sin(α) × Δα, which
    properly accounts for unequal pitch-angle spacing in the grid.
    Without it, falls back to sin(α) alone (equal-spacing assumption).
    """
    sin_a = np.sin(np.radians(pitch_deg))
    if delta_alpha_deg is not None:
        return sin_a * np.radians(delta_alpha_deg)
    return sin_a


def compute_pitch_bin_widths(pitch_list_deg):
    """
    Given a sorted list of sampled pitch angles, compute the Δα each
    one represents using midpoint bin edges (clamped to [0°, 90°]).

    E.g. for pitches [10, 30, 50, 70, 89]:
      bin edges = [0, 20, 40, 60, 79.5, 90]  (midpoints, clamped)
      Δα        = [20, 20, 20, 19.5, 10.5]
    """
    pitches = np.array(sorted(set(pitch_list_deg)), dtype=np.float64)
    if len(pitches) == 1:
        return {pitches[0]: 180.0}  # full hemisphere

    # Midpoints between consecutive angles
    mids = 0.5 * (pitches[:-1] + pitches[1:])
    # Clamp lower edge to 0°, upper edge to 90°
    edges = np.concatenate([[max(0.0, pitches[0] - (mids[0] - pitches[0]))],
                             mids,
                             [min(90.0, pitches[-1] + (pitches[-1] - mids[-1]))]])
    widths = np.diff(edges)
    return dict(zip(pitches, widths))


# ═══════════════════════════════════════════════════════════════════
# Plotting
# ═══════════════════════════════════════════════════════════════════

def _ap8_colormap():
    """
    Build a custom colormap matching the AP-8 MAX style:
    dark purple → blue → cyan → green → yellow → orange → red → white/pink
    """
    from matplotlib.colors import LinearSegmentedColormap
    colors = [
        (0.15, 0.0,  0.35),   # dark purple
        (0.30, 0.0,  0.60),   # purple
        (0.0,  0.0,  0.80),   # blue
        (0.0,  0.40, 1.0),    # medium blue
        (0.0,  0.75, 1.0),    # cyan-blue
        (0.0,  1.0,  0.85),   # cyan
        (0.0,  0.90, 0.40),   # green
        (0.30, 1.0,  0.0),    # yellow-green
        (0.85, 1.0,  0.0),    # yellow
        (1.0,  0.75, 0.0),    # orange
        (1.0,  0.35, 0.0),    # red-orange
        (1.0,  0.0,  0.0),    # red
        (1.0,  0.60, 0.60),   # pink
        (1.0,  0.90, 0.90),   # white-pink
    ]
    return LinearSegmentedColormap.from_list("ap8_style", colors, N=512)


def plot_flux_map(flux_2d, rho_edges, z_edges, save_path=None, title="",
                  vmin=None, vmax=None, rescale_peak=None):
    """
    Plot the meridian-plane flux map in AP-8 MAX style.
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    # White background
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # Bin centers for contours
    rho_centers = 0.5 * (rho_edges[:-1] + rho_edges[1:])
    z_centers   = 0.5 * (z_edges[:-1] + z_edges[1:])

    # Rescale so the peak value matches a target (e.g. AP-8 peak ~ 1e5 cm⁻² s⁻¹)
    if rescale_peak is not None and flux_2d.max() > 0:
        scale_factor = rescale_peak / flux_2d.max()
        flux_2d = flux_2d * scale_factor
        units_label = r"Omnidirectional Flux (cm$^{-2}$ s$^{-1}$, rescaled to AP-8 peak)"
    else:
        units_label = "Omnidirectional Flux (arb. units)"

    # Mask zero bins for log scale
    flux_plot = np.ma.masked_where(flux_2d <= 0, flux_2d)

    if vmin is None:
        vmin = flux_plot[flux_plot > 0].min() if flux_plot.count() > 0 else 1e-1
    if vmax is None:
        vmax = flux_plot.max() if flux_plot.count() > 0 else 1.0

    # Snap vmin/vmax to nearest power of 10 for cleaner colorbar
    vmin_log = 10 ** np.floor(np.log10(vmin))
    vmax_log = 10 ** np.ceil(np.log10(vmax))

    ap8_cmap = _ap8_colormap()
    norm = LogNorm(vmin=vmin_log, vmax=vmax_log)

    # pcolormesh: x-axis = rho, y-axis = z
    pcm = ax.pcolormesh(rho_edges, z_edges, flux_plot.T,
                         norm=norm, cmap=ap8_cmap, shading="flat")

    # Colorbar
    cbar = fig.colorbar(pcm, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label(units_label, fontsize=11, rotation=270, labelpad=18)

    # Draw Earth (white fill with black outline)
    earth = Circle((0, 0), 1.0, color="white", ec="black", lw=1.5, zorder=10)
    ax.add_patch(earth)

    # Dipole field lines for reference
    for L in [2, 3, 4, 5, 6]:
        theta = np.linspace(-np.pi / 2, np.pi / 2, 200)
        r_fl = L * np.cos(theta) ** 2
        rho_fl = r_fl * np.cos(theta)
        z_fl = r_fl * np.sin(theta)
        ax.plot(rho_fl, z_fl, "k-", lw=0.3, alpha=0.25)

    ax.set_xlabel(r"$\rho$ (R$_E$)", fontsize=13)
    ax.set_ylabel(r"$z$ (R$_E$)", fontsize=13)
    ax.set_aspect("equal")
    ax.set_xlim(0, rho_edges[-1])
    ax.set_ylim(z_edges[0], z_edges[-1])
    ax.tick_params(direction="in", which="both")

    if title:
        ax.set_title(title, fontsize=12)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {save_path}")
    else:
        plt.show()
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Build AP-8-style meridian flux map from batch h5 trajectories",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python %(prog)s outputs/outputs_rawdata/*.h5
  python %(prog)s outputs/outputs_rawdata/*.h5 --E-min 10e6 --bins 80
  python %(prog)s outputs/outputs_rawdata/*.h5 --spectral-index 2.5
        """)
    parser.add_argument("h5_files", nargs="+",
                        help="H5 trajectory files (supports glob)")
    parser.add_argument("--E-min", type=float, default=None,
                        help="Minimum energy in eV (e.g. 10e6 for >10 MeV)")
    parser.add_argument("--E-max", type=float, default=None,
                        help="Maximum energy in eV")
    parser.add_argument("--bins", type=int, default=80,
                        help="Number of bins per axis (default: 80)")
    parser.add_argument("--rho-max", type=float, default=7.0,
                        help="Maximum ρ in R_E (default: 7.0)")
    parser.add_argument("--z-max", type=float, default=3.0,
                        help="Maximum |z| in R_E (default: 3.0)")
    parser.add_argument("--spectral-index", type=float, default=2.0,
                        help="Power-law spectral index γ for j(E)∝E^(-γ) (default: 2.0)")
    parser.add_argument("--chunk-size", type=int, default=1_000_000,
                        help="H5 read chunk size (default: 1M)")
    parser.add_argument("--output", type=str, default="flux_map_meridian.png",
                        help="Output file path")
    parser.add_argument("--proton-only", action="store_true", default=True,
                        help="Only include proton runs (default: True)")
    parser.add_argument("--exclude-lost", action="store_true", default=False,
                        help="Exclude orbits that hit the atmosphere (only count trapped particles)")
    parser.add_argument("--rescale-peak", type=float, default=None,
                        help="Rescale flux so peak matches this value (e.g. 1e5 for AP-8 scale)")
    parser.add_argument("--vmin", type=float, default=None,
                        help="Force colorbar minimum (e.g. 1e0)")
    parser.add_argument("--vmax", type=float, default=None,
                        help="Force colorbar maximum (e.g. 1e5)")
    args = parser.parse_args()

    # Expand glob patterns
    h5_paths = []
    for pattern in args.h5_files:
        expanded = glob.glob(pattern)
        if expanded:
            h5_paths.extend(expanded)
        elif os.path.exists(pattern):
            h5_paths.append(pattern)
    h5_paths = sorted(set(h5_paths))

    if not h5_paths:
        print("No h5 files found.")
        sys.exit(1)

    print(f"Found {len(h5_paths)} h5 files.\n")

    # ── First pass: discover all pitch angles for bin-width calculation ──
    print("Scanning pitch angles for Δα bin widths...")
    all_pitches = set()
    for path in h5_paths:
        with h5py.File(path, "r") as f:
            if "summary_json" in f.attrs:
                summary = json.loads(f.attrs["summary_json"])
                p = summary.get("meta", {}).get("pitch_deg", None)
                if p is not None:
                    all_pitches.add(float(p))
    if all_pitches:
        pitch_bin_widths = compute_pitch_bin_widths(all_pitches)
        print(f"  Pitch angles found: {sorted(all_pitches)}")
        print(f"  Bin widths (deg): {[f'{pitch_bin_widths[p]:.1f}' for p in sorted(pitch_bin_widths)]}\n")
    else:
        pitch_bin_widths = {}
        print("  No pitch angle metadata found, using sin(α) only.\n")

    # Build grid
    rho_edges = np.linspace(0, args.rho_max, args.bins + 1)
    z_edges   = np.linspace(-args.z_max, args.z_max, args.bins + 1)
    flux_2d   = np.zeros((args.bins, args.bins), dtype=np.float64)

    n_used = 0
    n_skipped = 0
    energies_used = set()
    pitches_used = set()
    L_shells_used = set()

    for i, path in enumerate(h5_paths):
        fname = os.path.basename(path)
        print(f"  [{i+1:>4d}/{len(h5_paths)}] {fname}...", end=" ", flush=True)

        hist, meta, success = accumulate_dwell_time(
            path, rho_edges, z_edges, chunk_size=args.chunk_size,
            exclude_lost=args.exclude_lost)

        if not success or hist is None:
            if meta.get("hit_atmosphere") and args.exclude_lost:
                print("hit atmosphere, excluded")
            else:
                print("no PS data, skipped")
            n_skipped += 1
            continue

        E = meta.get("energy_eV")
        pitch = meta.get("pitch_deg")

        # Apply energy filter
        if args.E_min is not None and E is not None and E < args.E_min:
            print(f"E={E/1e6:.0f} MeV < threshold, skipped")
            n_skipped += 1
            continue
        if args.E_max is not None and E is not None and E > args.E_max:
            print(f"E={E/1e6:.0f} MeV > threshold, skipped")
            n_skipped += 1
            continue

        # Skip if dwell time is all zeros (empty trajectory or immediate crash)
        if hist.sum() == 0:
            print("empty trajectory, skipped")
            n_skipped += 1
            continue

        # Compute weights
        # Spectrum: j(E) ∝ E^(-γ)
        w_spectrum = spectrum_weight(E, args.spectral_index) if E else 1.0
        # Pitch angle: sin(α) × Δα for proper omnidirectional integration
        delta_a = pitch_bin_widths.get(pitch, None) if pitch else None
        w_pitch = pitch_angle_weight(pitch, delta_a) if pitch else 1.0
        weight = w_spectrum * w_pitch

        # Accumulate
        flux_2d += hist * weight
        n_used += 1

        if E:
            energies_used.add(E)
        if pitch:
            pitches_used.add(pitch)
        if meta.get("L_eff"):
            L_shells_used.add(meta["L_eff"])

        E_str = f"E={E/1e6:.0f}MeV" if E else "E=?"
        L_str = f"L={meta.get('L_eff', '?')}"
        p_str = f"α={pitch:.0f}°" if pitch else "α=?"
        dwell = hist.sum()
        print(f"{E_str}  {L_str}  {p_str}  dwell={dwell:.0e}  w={weight:.2e}")

    print(f"\n{'='*60}")
    print(f"  Orbits used: {n_used}   Skipped: {n_skipped}")
    print(f"  Energies: {sorted(e/1e6 for e in energies_used)} MeV")
    print(f"  Pitch angles: {sorted(pitches_used)}°")
    print(f"  L-shells: {sorted(L_shells_used)}")
    print(f"{'='*60}")

    if n_used == 0:
        print("No valid orbits. Nothing to plot.")
        sys.exit(1)

    # Build title
    E_range = f"{min(energies_used)/1e6:.0f}–{max(energies_used)/1e6:.0f} MeV" if energies_used else ""
    title = (f"PS Meridian Flux Map — Proton > {args.E_min/1e6:.0f} MeV\n"
             f"{n_used} orbits, {E_range}, "
             f"spectral index γ={args.spectral_index}"
             if args.E_min else
             f"PS Meridian Flux Map — Proton ({E_range})\n"
             f"{n_used} orbits, spectral index γ={args.spectral_index}")

    plot_flux_map(flux_2d, rho_edges, z_edges,
                  save_path=args.output, title=title,
                  rescale_peak=args.rescale_peak,
                  vmin=args.vmin, vmax=args.vmax)


if __name__ == "__main__":
    main()
