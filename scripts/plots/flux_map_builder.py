#!/usr/bin/env python3
"""
Dwell Map Builder
=================
Post-processes batch run h5 trajectories into a meridian-plane
dwell-occupancy map vs (ρ, z) in Earth radii.

NOTE on terminology: this builds a TRAJECTORY DWELL-OCCUPANCY map, not a
true omnidirectional particle flux.  Two physics ingredients required for a
true flux map are NOT applied here:
  - The toroidal Jacobian (2π·ρ) that converts a Cartesian bin count into a
    3D volume density;
  - A genuine pitch-angle integration (current pipeline uses one pitch per
    cell).
The output is therefore best read as "where each orbit spends its time,
weighted by an assumed energy spectrum and an L-shell population profile."
That's still informative for showing belt-shape geometry, but it is not a
density and should not be overlaid quantitatively on flux references.

The idea:
  1. For each trapped orbit, compute how much TIME it spends in each (ρ, z) bin
     → this gives a per-orbit "shape" (dwell time per bin, normalized)
  2. Weight each orbit by a power-law energy spectrum j(E) ∝ E^(-γ)
     and by sin(α₀) for pitch-angle isotropy
  3. Sum across all orbits → meridian dwell-occupancy map

Usage:
    # Build from all batch h5 files
    python flux_map_builder.py outputs/outputs_rawdata/*.h5

    # With energy range filter
    python flux_map_builder.py outputs/outputs_rawdata/*.h5 --E-min 10e6

    # Custom grid resolution
    python flux_map_builder.py outputs/outputs_rawdata/*.h5 --bins 100

    # Only process files from a specific output folder
    python flux_map_builder.py outputs/outputs_rawdata/*.h5 --output-dir outputs/dwell_map
"""

import sys
import os
import csv
import json
import glob
import argparse
import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Circle


# ═══════════════════════════════════════════════════════════════════
# Boundary status lookup
# ═══════════════════════════════════════════════════════════════════

def _read_boundary_from_csv(h5_path):
    """
    Read boundary status from the per-cell master_simulation_log.csv
    that sits alongside the _rawdata/ folder containing the h5.

    Returns "CLOSED", "OPEN", or None if no CSV / no boundary column.
    """
    # h5 lives at  .../<run_folder>/_rawdata/<hash>_full.h5
    # CSV lives at .../<run_folder>/master_simulation_log.csv
    rawdata_dir = os.path.dirname(h5_path)
    run_folder  = os.path.dirname(rawdata_dir)
    csv_path    = os.path.join(run_folder, "master_simulation_log.csv")

    if not os.path.isfile(csv_path):
        return None

    try:
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                return row.get("boundary", None)
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════════
# Dwell-time accumulation
# ═══════════════════════════════════════════════════════════════════

def accumulate_dwell_time(h5_path, rho_edges, z_edges, chunk_size=1_000_000,
                          exclude_lost=False, truncate_atm=False,
                          r_atmosphere=1.0):
    """
    Read an h5 trajectory and accumulate dwell time in (ρ, z) bins.

    ρ = sqrt(x² + y²)  (cylindrical radius in R_E)
    z = z               (height above equator in R_E)

    Parameters
    ----------
    truncate_atm : bool
        If True, stop accumulating at the first saved point where
        r < r_atmosphere.  Data after the hit is discarded.  The orbit
        still counts (unlike exclude_lost which skips it entirely).
    r_atmosphere : float
        Atmosphere radius in R_E (default 1.0 = surface).

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
            meta["L_eff"]      = m.get("x_initial", None)   # writer stores "x_initial", not "x0"
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
        hit_boundary = False

        for start in range(0, n_steps, chunk_size):
            end = min(start + chunk_size, n_steps)
            chunk = y_ds[:, start:end]

            x = chunk[0]
            y = chunk[1]
            z = chunk[2]

            # Truncate at first atmosphere hit if requested
            if truncate_atm and hit_atm:
                r_sq = x**2 + y**2 + z**2
                below = np.where(r_sq < r_atmosphere**2)[0]
                if len(below) > 0:
                    cut = below[0]  # first point below atmosphere
                    x = x[:cut]
                    y = y[:cut]
                    z = z[:cut]
                    hit_boundary = True
                    if len(x) == 0:
                        break  # hit on first point of this chunk

            rho = np.sqrt(x**2 + y**2)

            # Read time steps for proper weighting
            n_pts = len(x)
            if has_time:
                t_chunk = f["ps"]["t"][start:start + n_pts]
                if len(t_chunk) > 1:
                    # dt per step (use forward difference, last step gets same dt)
                    dt = np.empty_like(t_chunk)
                    dt[:-1] = t_chunk[1:] - t_chunk[:-1]
                    dt[-1] = dt[-2] if len(dt) > 1 else 1.0
                else:
                    dt = np.ones_like(t_chunk)
            else:
                # No time info → uniform weight (fixed step assumed)
                dt = np.ones(n_pts, dtype=np.float64)

            # Accumulate weighted histogram
            h, _, _ = np.histogram2d(rho, z, bins=[rho_edges, z_edges], weights=dt)
            hist_2d += h

            if hit_boundary:
                break

    # Normalize to fractional occupancy so orbits with different
    # total simulation times contribute shape, not magnitude.
    total = hist_2d.sum()
    if total > 0:
        hist_2d /= total

    return hist_2d, meta, True


def spectrum_weight(energy_eV, spectral_index=2.0, E_ref=10e6):
    """
    Power-law energy spectrum weight: j(E) ∝ (E/E_ref)^(-γ)
    Default γ=2.0 is typical for inner belt protons.
    """
    return (energy_eV / E_ref) ** (-spectral_index)


def pitch_angle_weight(pitch_deg, delta_alpha_deg=None):
    """
    Pitch-angle weight: dΩ = 2π sin(α) dα.

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


def compute_energy_bin_widths(energy_list_eV):
    """
    ΔE (eV) represented by each sampled energy, from log-midpoint bin
    edges.  Used only with --dE-weight: makes  Σ j(E_i)·ΔE_i  a proper
    quadrature of  ∫ j(E) dE  on a non-uniform (log-spaced) energy grid.

    The LOWEST bin's lower edge is clamped to the lowest sampled energy
    (no symmetric extrapolation below it).  This matters for a ">E_min"
    integral map: e.g. with a 10 MeV lowest sample, the bin spans
    [10, mid(10,15)] rather than extrapolating down to ~8 MeV, so the
    map never counts phantom contributions below 10 MeV — energies you neither
    sampled nor intend to include in a ">10 MeV" integral.

    (The highest bin's upper edge is still extrapolated symmetrically,
    representing the "and above" tail; with a steep spectrum its
    contribution is negligible.  Clamp it too if you need a hard upper
    cutoff at the top sampled energy.)
    """
    energies = np.array(sorted(set(energy_list_eV)), dtype=np.float64)
    if len(energies) == 1:
        return {energies[0]: 1.0}

    logE = np.log10(energies)
    mids = 0.5 * (logE[:-1] + logE[1:])
    edges_log = np.concatenate([[logE[0]],   # clamp: no extrapolation below lowest sample
                                mids,
                                [logE[-1] + (logE[-1] - mids[-1])]])
    widths = np.diff(10.0 ** edges_log)
    return dict(zip(energies, widths))


def make_radial_weight(index=0.0, L_ref=1.0, profile_path=None):
    """
    Build a radial population-weight callable  w(L)  that imposes an
    L-dependent abundance the dwell-time maps cannot produce on their own.

    The orbits give the *spatial shape* each shell occupies; the *abundance*
    per shell is an external input (source/loss physics not in the model).
    This applies that input as a per-orbit weight, alongside the spectral
    and pitch-angle weights.

    Two modes
    ---------
    profile_path given : interpolate an empirical (L, value) table — e.g. a
        published omnidirectional-flux-vs-L profile — in log space. Values
        outside the table's L range clamp to the nearest endpoint. Overrides
        the power law.
    else : power law  w(L) = (L / L_ref)^(-index).

    Defaults (index=0.0, no profile) give w(L)=1 for all L, i.e. uniform
    weighting and unchanged behavior.
    """
    if profile_path is not None:
        L_tab, f_tab = [], []
        with open(profile_path) as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.replace(",", " ").split()
                L_tab.append(float(parts[0]))
                f_tab.append(float(parts[1]))
        L_tab = np.asarray(L_tab, dtype=np.float64)
        f_tab = np.asarray(f_tab, dtype=np.float64)
        order = np.argsort(L_tab)
        L_tab, f_tab = L_tab[order], f_tab[order]
        logf = np.log10(np.clip(f_tab, 1e-300, None))

        def _w(L):
            # np.interp clamps to endpoint values outside [L_tab[0], L_tab[-1]]
            return 10.0 ** float(np.interp(L, L_tab, logf))
        return _w

    def _w(L):
        return (L / L_ref) ** (-index)
    return _w


# ═══════════════════════════════════════════════════════════════════
# State persistence  (per-orbit unweighted dwell-time histograms)
# ═══════════════════════════════════════════════════════════════════

def _orbit_key(energy_eV, L, pitch_deg):
    """Canonical string key for one orbit: 'E<eV>_L<L>_P<pitch>'."""
    return f"E{energy_eV:.6g}_L{L}_P{pitch_deg}"


def save_state(path, orbit_store, rho_edges, z_edges):
    """
    Save per-orbit unweighted dwell-time histograms to a .npz file.

    orbit_store : list of dicts, each with keys
        'key', 'energy_eV', 'L', 'pitch_deg', 'hist', 'hit_atmosphere'
    """
    meta_json = json.dumps([
        {k: v for k, v in orb.items() if k != "hist"}
        for orb in orbit_store
    ])
    hists = np.stack([orb["hist"] for orb in orbit_store])  # (N, bins, bins)
    np.savez_compressed(path,
                        rho_edges=rho_edges,
                        z_edges=z_edges,
                        hists=hists,
                        meta_json=meta_json)
    size_mb = os.path.getsize(path) / 1024**2
    print(f"Saved state: {path}  ({len(orbit_store)} orbits, {size_mb:.1f} MB)")


def load_state(path):
    """
    Load a previously saved state file.

    Returns
    -------
    orbit_store : list of dicts  (same schema as save_state input)
    rho_edges   : 1D array
    z_edges     : 1D array
    """
    data = np.load(path, allow_pickle=False)
    rho_edges = data["rho_edges"]
    z_edges   = data["z_edges"]
    hists     = data["hists"]
    meta_list = json.loads(str(data["meta_json"]))

    orbit_store = []
    for i, m in enumerate(meta_list):
        m["hist"] = hists[i]
        orbit_store.append(m)

    print(f"Loaded state: {path}  ({len(orbit_store)} orbits)")
    return orbit_store, rho_edges, z_edges


def build_dwell_from_store(orbit_store, rho_edges, z_edges, spectral_index,
                          E_min=None, E_max=None, E_target=None,
                          radial_weight_fn=None, use_dE=False,
                          exclude_lost=False):
    """
    Apply spectral + pitch-angle (+ optional radial) weights to stored
    unweighted histograms and sum into a single dwell-occupancy map.

    E_target : float or None
        If set, only include orbits matching this energy (1% tolerance).
    radial_weight_fn : callable or None
        If given, w(L) imposing an L-dependent population abundance
        (see make_radial_weight). None → uniform (weight 1.0).

    Returns  (dwell_2d, n_used, energies_used, pitches_used, L_shells_used)
    """
    bins_rho = len(rho_edges) - 1
    bins_z   = len(z_edges) - 1
    dwell_2d  = np.zeros((bins_rho, bins_z), dtype=np.float64)

    # Gather all pitches for bin-width calculation
    all_pitches = set(orb["pitch_deg"] for orb in orbit_store
                      if orb.get("pitch_deg") is not None)
    pitch_bin_widths = compute_pitch_bin_widths(all_pitches) if all_pitches else {}

    # ΔE quadrature widths (only if use_dE), from energies passing filters
    energy_bin_widths = {}
    if use_dE:
        kept = set()
        for orb in orbit_store:
            if exclude_lost and orb.get("hit_atmosphere", False):
                continue
            E = orb.get("energy_eV")
            if E is None:
                continue
            if E_min is not None and E < E_min:
                continue
            if E_max is not None and E > E_max:
                continue
            if E_target is not None and abs(E - E_target) / E_target > 0.01:
                continue
            kept.add(E)
        if kept:
            energy_bin_widths = compute_energy_bin_widths(kept)

    n_used = 0
    energies_used = set()
    pitches_used  = set()
    L_shells_used = set()

    for orb in orbit_store:
        E     = orb.get("energy_eV")
        pitch = orb.get("pitch_deg")
        L     = orb.get("L")

        # Drop orbits that ever hit the atmosphere (stably-trapped-only map)
        if exclude_lost and orb.get("hit_atmosphere", False):
            continue

        # Energy filters
        if E_min is not None and E is not None and E < E_min:
            continue
        if E_max is not None and E is not None and E > E_max:
            continue
        if E_target is not None and E is not None:
            if abs(E - E_target) / E_target > 0.01:
                continue

        hist = orb["hist"]
        if hist.sum() == 0:
            continue

        if E:
            w_spectrum = spectrum_weight(E, spectral_index)
        else:
            w_spectrum = 1.0
        w_dE = energy_bin_widths.get(E, 1.0) if (use_dE and E) else 1.0
        delta_a = pitch_bin_widths.get(pitch, None) if pitch else None
        w_pitch = pitch_angle_weight(pitch, delta_a) if pitch else 1.0
        w_radial = radial_weight_fn(L) if (radial_weight_fn is not None and L) else 1.0
        weight = w_spectrum * w_dE * w_pitch * w_radial

        dwell_2d += hist * weight
        n_used += 1
        if E:     energies_used.add(E)
        if pitch: pitches_used.add(pitch)
        if L:     L_shells_used.add(L)

    return dwell_2d, n_used, energies_used, pitches_used, L_shells_used


# ═══════════════════════════════════════════════════════════════════
# Plotting
# ═══════════════════════════════════════════════════════════════════

def _dwell_colormap():
    """
    Rainbow-style colormap for log-scale dwell-occupancy maps:
    dark violet → blue → cyan → green → yellow → orange → red, with red
    holding a broad band near the top and only a thin white cap at the
    very maximum.  Color stops are placed by position (0–1) so the red
    band is wide and white is compressed into the top ~3%; this keeps
    the bright core red instead of saturating to white/pink.
    """
    from matplotlib.colors import LinearSegmentedColormap
    stops = [
        (0.00, (0.04, 0.0,  0.10)),   # near-black (floor)
        (0.05, (0.16, 0.0,  0.32)),   # dark violet
        (0.11, (0.34, 0.0,  0.62)),   # purple
        (0.18, (0.0,  0.0,  0.85)),   # blue
        (0.30, (0.0,  0.45, 1.0)),    # medium blue
        (0.40, (0.0,  0.80, 1.0)),    # cyan-blue
        (0.50, (0.0,  1.0,  0.80)),   # cyan
        (0.58, (0.0,  0.90, 0.30)),   # green
        (0.66, (0.45, 1.0,  0.0)),    # yellow-green
        (0.74, (0.90, 1.0,  0.0)),    # yellow
        (0.82, (1.0,  0.70, 0.0)),    # orange
        (0.89, (1.0,  0.35, 0.0)),    # red-orange
        (0.97, (1.0,  0.0,  0.0)),    # red  (broad band to the top)
        (1.00, (1.0,  0.95, 0.95)),   # thin white cap at the very peak
    ]
    return LinearSegmentedColormap.from_list("dwell_style", stops, N=512)


def _sep_blur(a, kernel, radius):
    """Separable Gaussian blur (reflect-padded), numpy-only."""
    p = np.pad(a, radius, mode="reflect")
    p = np.apply_along_axis(lambda r: np.convolve(r, kernel, mode="same"), 0, p)
    p = np.apply_along_axis(lambda r: np.convolve(r, kernel, mode="same"), 1, p)
    return p[radius:-radius, radius:-radius]


def gaussian_smooth_2d(arr, sigma, space="log"):
    """
    Separable Gaussian smoothing (numpy-only, no scipy dependency).
    sigma is in units of bins.

    space = "linear"
        Arithmetic Gaussian average of the raw values. Conserves the
        linear total, BUT the average is dominated by the largest value
        in the window — so on a LOG color scale bright cores bleed
        outward and faint structure washes out. Risky for multi-decade
        data displayed logarithmically; kept only for backward
        compatibility.

    space = "log"  (default)
        Smooths log10(value) using only populated neighbours
        (normalized convolution), then exponentiates — a geometric mean.
        No single bright cell dominates, empty regions stay empty
        (no bleed into the background), and it respects the log display.
        This is the appropriate choice for a log-colorbar quantity.
    """
    if sigma <= 0:
        return arr
    radius = max(1, int(np.ceil(3 * sigma)))
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    kernel /= kernel.sum()

    if space == "log":
        mask = (arr > 0).astype(np.float64)
        a = np.where(arr > 0, np.log10(np.where(arr > 0, arr, 1.0)), 0.0)
        num = _sep_blur(a, kernel, radius)        # = blur of (logval on populated cells)
        den = _sep_blur(mask, kernel, radius)     # local populated weight
        with np.errstate(invalid="ignore", divide="ignore"):
            sm = np.where(den > 1e-6, num / den, 0.0)
        # keep a cell only if it had meaningful populated support nearby,
        # so smoothing fills gaps within the belt but never paints the
        # empty background at the colorbar floor.
        return np.where(den > 1e-3, 10.0 ** sm, 0.0)

    return _sep_blur(arr, kernel, radius)          # linear (legacy)


def plot_dwell_map(dwell_2d, rho_edges, z_edges, save_path=None, title="",
                  vmin=None, vmax=None, rescale_peak=None,
                  plot_rho_max=None, plot_z_max=None, r_mask=None,
                  smooth_sigma=0.0, smooth_space="log", contours=False):
    """
    Plot the meridian-plane dwell-occupancy map on a log color scale.

    smooth_sigma : float
        If > 0, apply a Gaussian smoothing of this width (in bins) to
        the map before display.  Cosmetic only — softens discrete
        orbit-sampling streaks.  Linear-space smoothing conserves the
        integrated value; log-space smoothing (geometric mean) does not.
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    if smooth_sigma and smooth_sigma > 0:
        dwell_2d = gaussian_smooth_2d(dwell_2d, smooth_sigma, space=smooth_space)

    # White background
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # Rescale so the peak value matches a target (cosmetic only — does not
    # turn the map into a physical density)
    if rescale_peak is not None and dwell_2d.max() > 0:
        scale_factor = rescale_peak / dwell_2d.max()
        dwell_2d = dwell_2d * scale_factor
        units_label = "Dwell Occupancy (rescaled to specified peak)"
    else:
        units_label = "Dwell Occupancy (arb. units)"

    # Mask bins inside r_mask radius (hide sub-atmosphere noise)
    if r_mask is not None:
        rho_centers = 0.5 * (rho_edges[:-1] + rho_edges[1:])
        z_centers   = 0.5 * (z_edges[:-1] + z_edges[1:])
        rho_grid, z_grid = np.meshgrid(rho_centers, z_centers, indexing="ij")
        r_grid = np.sqrt(rho_grid**2 + z_grid**2)
        dwell_2d = np.where(r_grid < r_mask, 0.0, dwell_2d)

    # Mask zero bins for log scale
    dwell_plot = np.ma.masked_where(dwell_2d <= 0, dwell_2d)

    # Only auto-computed limits get snapped to a clean power of 10;
    # limits passed explicitly (--vmin/--vmax) are used verbatim, so a
    # value like 3e5 stays 3e5 instead of rounding up to 1e6.
    if vmin is None:
        vmin = dwell_plot[dwell_plot > 0].min() if dwell_plot.count() > 0 else 1e-1
        vmin_log = 10 ** np.floor(np.log10(vmin))
    else:
        vmin_log = vmin
    if vmax is None:
        vmax = dwell_plot.max() if dwell_plot.count() > 0 else 1.0
        vmax_log = 10 ** np.ceil(np.log10(vmax))
    else:
        vmax_log = vmax

    dwell_cmap = _dwell_colormap()
    norm = LogNorm(vmin=vmin_log, vmax=vmax_log)

    # pcolormesh: x-axis = rho, y-axis = z
    pcm = ax.pcolormesh(rho_edges, z_edges, dwell_plot.T,
                         norm=norm, cmap=dwell_cmap, shading="flat")

    # Optional black iso-contour lines at each decade
    if contours:
        lo = int(np.floor(np.log10(vmin_log)))
        hi = int(np.ceil(np.log10(vmax_log)))
        levels = [10.0 ** e for e in range(lo, hi + 1)]
        # only keep levels actually spanned by the data
        if dwell_plot.count() > 0:
            dmin, dmax = dwell_plot.min(), dwell_plot.max()
            levels = [L for L in levels if dmin < L < dmax]
        if levels:
            rc = 0.5 * (rho_edges[:-1] + rho_edges[1:])
            zc = 0.5 * (z_edges[:-1] + z_edges[1:])
            cs = ax.contour(rc, zc, dwell_plot.T, levels=levels,
                            colors="black", linewidths=0.6, zorder=8)
            # label each level with its decade exponent (e.g. "3" for 1e3)
            ax.clabel(cs, inline=True, fontsize=7,
                      fmt=lambda v: f"{int(round(np.log10(v)))}")

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
    ax.set_xlim(0, plot_rho_max if plot_rho_max else rho_edges[-1])
    _z_lim = plot_z_max if plot_z_max else abs(z_edges[-1])
    ax.set_ylim(-_z_lim, _z_lim)
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
        description="Build meridian dwell-occupancy map from batch h5 trajectories",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto-discover h5 files from a batch group
  python %(prog)s --group fluxmap_10mev
  python %(prog)s --group fluxmap_all --per-energy

  # Explicit h5 file list
  python %(prog)s outputs/outputs_rawdata/*.h5
  python %(prog)s outputs/outputs_rawdata/*.h5 --E-min 10e6 --bins 80

  # Save state after processing (archive h5 files afterward)
  python %(prog)s --group fluxmap_10mev --save-state fluxmap_10mev.npz

  # Re-plot from saved state (no h5 files needed)
  python %(prog)s --load-state fluxmap_10mev.npz --spectral-index 2.5

  # Add new runs to existing state
  python %(prog)s --load-state fluxmap.npz --group fluxmap_20mev --save-state fluxmap.npz
        """)
    parser.add_argument("h5_files", nargs="*", default=[],
                        help="H5 trajectory files (supports glob). "
                             "Optional if --group is provided.")
    parser.add_argument("--group", type=str, default=None,
                        help="Auto-discover h5 files from data/dipoleb/<group>/ "
                             "(scans */_rawdata/*.h5). Combines with any explicit h5_files.")
    parser.add_argument("--per-energy", action="store_true", default=False, dest="per_energy",
                        help="Produce a separate meridional map for each energy found, "
                             "in addition to the combined map.")
    parser.add_argument("--E-min", type=float, default=None,
                        help="Minimum energy in eV (e.g. 10e6 for >10 MeV)")
    parser.add_argument("--E-max", type=float, default=None,
                        help="Maximum energy in eV")
    parser.add_argument("--bins", type=int, default=80,
                        help="Number of bins per axis (default: 80)")
    parser.add_argument("--rho-max", type=float, default=6.0,
                        help="Maximum ρ in R_E (default: 6.0). With z-max=3 this "
                             "gives square 0.075 R_E bins on an 80-grid.")
    parser.add_argument("--z-max", type=float, default=3.0,
                        help="Maximum |z| in R_E (default: 3.0)")
    parser.add_argument("--spectral-index", type=float, default=0.0,
                        help="Power-law spectral index γ for j(E)∝E^(-γ) (default: 0.0 = equal weight)")
    parser.add_argument("--dE-weight", action="store_true", default=False, dest="dE_weight",
                        help="Weight each orbit by the energy bin width ΔE it represents "
                             "(log-midpoint quadrature). Makes the sum over a non-uniform "
                             "energy grid approximate ∫j(E)dE instead of Σj(E_i). "
                             "Default: off (every energy counts equally).")
    parser.add_argument("--radial-index", type=float, default=0.0, dest="radial_index",
                        help="Impose a radial population weight w(L)=(L/L_ref)^(-index). "
                             "Default 0.0 = uniform (no radial reweighting). The dwell "
                             "maps give spatial shape only; this supplies the abundance "
                             "per L-shell. Inner-belt profiles are steep (try 4-8).")
    parser.add_argument("--radial-L-ref", type=float, default=1.0, dest="radial_L_ref",
                        help="Reference L for the radial power law (default 1.0). Only "
                             "sets overall scale, not shape.")
    parser.add_argument("--radial-profile", type=str, default=None, dest="radial_profile",
                        metavar="FILE",
                        help="Two-column (L, value) table to use as the radial population "
                             "profile instead of a power law. "
                             "Interpolated in log space; overrides --radial-index.")
    parser.add_argument("--chunk-size", type=int, default=1_000_000,
                        help="H5 read chunk size (default: 1M)")
    parser.add_argument("--output", type=str, default="dwell_map_meridian.png",
                        help="Output file path")
    parser.add_argument("--exclude-lost", action="store_true", default=False,
                        help="Exclude orbits that hit the atmosphere (only count trapped particles)")
    parser.add_argument("--truncate-atm", action="store_true", default=False, dest="truncate_atm",
                        help="For orbits that hit atmosphere, only count dwell time "
                             "up to the first impact (discard post-impact trajectory).")
    parser.add_argument("--r-atm", type=float, default=1.0, dest="r_atm",
                        help="Atmosphere radius in R_E for --truncate-atm (default: 1.0).")
    parser.add_argument("--exclude-open", action="store_true", default=True, dest="exclude_open",
                        help="Skip orbits with OPEN boundary status (default: True). "
                             "Open orbits are escape trajectories and should not "
                             "contribute to a trapped-particle dwell map.")
    parser.add_argument("--include-open", action="store_false", dest="exclude_open",
                        help="Include OPEN orbits (overrides --exclude-open).")
    parser.add_argument("--rescale-peak", type=float, default=None,
                        help="Rescale map so peak matches this value (cosmetic only — does "
                             "not turn the map into a physical density).")
    parser.add_argument("--vmin", type=float, default=None,
                        help="Force colorbar minimum (e.g. 1e0)")
    parser.add_argument("--vmax", type=float, default=None,
                        help="Force colorbar maximum (e.g. 1e5)")
    parser.add_argument("--plot-rho-max", type=float, default=6.0, dest="plot_rho_max",
                        help="Crop plot x-axis at this ρ value (default: 6.0). Crops "
                             "display only; for older states built at rho_max=7 this "
                             "shows 0-6. Pass a larger value to see the full grid.")
    parser.add_argument("--plot-z-max", type=float, default=None, dest="plot_z_max",
                        help="Crop plot y-axis to ±this z value (e.g. 2.5)")
    parser.add_argument("--r-mask", type=float, default=None, dest="r_mask",
                        help="Hide bins with r < this value (e.g. 1.1 to blank inside atmosphere)")
    parser.add_argument("--smooth", type=float, default=0.0, dest="smooth",
                        metavar="SIGMA",
                        help="Gaussian-smooth the map before display, SIGMA in bins "
                             "(e.g. 1.0). COSMETIC ONLY — softens discrete orbit-"
                             "sampling streaks. Default: 0 = OFF (no smoothing). "
                             "Verify any conclusions on the unsmoothed map.")
    parser.add_argument("--contours", action="store_true", default=False,
                        help="Overlay black iso-contour lines at each decade, "
                             "labeled by exponent.")
    parser.add_argument("--smooth-space", type=str, default="log", dest="smooth_space",
                        choices=["log", "linear"],
                        help="Smoothing space when --smooth>0. 'log' (default): "
                             "geometric mean of log10(value) over populated cells — "
                             "correct for a log colorbar, no bright-core bleed. "
                             "'linear': arithmetic mean (legacy); biased toward bright "
                             "values on a log scale, use with caution.")
    parser.add_argument("--save-state", type=str, default=None, dest="save_state",
                        metavar="FILE",
                        help="Save per-orbit unweighted dwell-time histograms to a "
                             ".npz file.  This lets you archive the h5 files and "
                             "still re-plot with different weights or add new data.")
    parser.add_argument("--load-state", type=str, default=None, dest="load_state",
                        metavar="FILE",
                        help="Load a previously saved state file.  Any new h5 files "
                             "(from --group or positional args) that aren't already "
                             "in the state are processed and merged in.")
    args = parser.parse_args()

    # ── Load existing state (if any) ──────────────────────────────────
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))

    orbit_store = []          # list of per-orbit dicts
    existing_keys = set()     # keys already in the store

    if args.load_state:
        if not os.path.isfile(args.load_state):
            print(f"State file not found: {args.load_state}")
            sys.exit(1)
        orbit_store, saved_rho, saved_z = load_state(args.load_state)
        existing_keys = {orb["key"] for orb in orbit_store}

        # Use the grid from the saved state (ignore --bins/--rho-max/--z-max)
        rho_edges = saved_rho
        z_edges   = saved_z
        print(f"  Using grid from state: {len(rho_edges)-1} bins, "
              f"ρ=[0, {rho_edges[-1]:.1f}], z=[{z_edges[0]:.1f}, {z_edges[-1]:.1f}]")
    else:
        rho_edges = np.linspace(0, args.rho_max, args.bins + 1)
        z_edges   = np.linspace(-args.z_max, args.z_max, args.bins + 1)

    # ── Auto-discover h5 files from --group ────────────────────────────
    h5_paths = []

    if args.group:
        group_dir = os.path.join(PROJECT_ROOT, "data", "dipoleb", args.group)
        if not os.path.isdir(group_dir):
            print(f"Group directory not found: {group_dir}")
            sys.exit(1)
        pattern = os.path.join(group_dir, "*", "_rawdata", "*.h5")
        discovered = sorted(glob.glob(pattern))
        print(f"Discovered {len(discovered)} h5 files from group '{args.group}'.")
        h5_paths.extend(discovered)

    # Expand explicit glob patterns
    for pattern in args.h5_files:
        expanded = glob.glob(pattern)
        if expanded:
            h5_paths.extend(expanded)
        elif os.path.exists(pattern):
            h5_paths.append(pattern)
    h5_paths = sorted(set(h5_paths))

    # ── Process new h5 files ───────────────────────────────────────────
    # (Skip any whose orbit key is already in the loaded state)

    if h5_paths:
        # Remove bad/unreadable files first
        bad_files = []
        for path in h5_paths:
            try:
                with h5py.File(path, "r") as f:
                    pass   # just test that it opens
            except Exception as e:
                bad_files.append(path)
                print(f"  Warning: could not open {os.path.basename(path)}: {e}")
        if bad_files:
            h5_paths = [p for p in h5_paths if p not in bad_files]
            print(f"  Skipping {len(bad_files)} unreadable file(s).")

        n_new = 0
        n_skipped = 0
        n_open = 0
        n_already = 0

        print(f"\nProcessing {len(h5_paths)} h5 files...")
        for i, path in enumerate(h5_paths):
            fname = os.path.basename(path)
            print(f"  [{i+1:>4d}/{len(h5_paths)}] {fname}...", end=" ", flush=True)

            # Check boundary status before doing expensive dwell-time work
            if args.exclude_open:
                boundary = _read_boundary_from_csv(path)
                if boundary is not None and boundary.upper() == "OPEN":
                    print("OPEN boundary, skipped")
                    n_skipped += 1
                    n_open += 1
                    continue
                if boundary is None:
                    print("no boundary CSV, skipped (may be open)")
                    n_skipped += 1
                    continue

            # Peek at metadata to build the orbit key
            try:
                with h5py.File(path, "r") as f:
                    if "summary_json" not in f.attrs:
                        print("no summary_json, skipped")
                        n_skipped += 1
                        continue
                    smry = json.loads(f.attrs["summary_json"])
                    m = smry.get("meta", {})
                    E     = m.get("energy_eV")
                    pitch = m.get("pitch_deg")
                    L     = m.get("x0")
            except Exception as e:
                print(f"read error: {e}")
                n_skipped += 1
                continue

            key = _orbit_key(E, L, pitch)
            if key in existing_keys:
                print("already in state, skipped")
                n_already += 1
                continue

            # Accumulate dwell time
            hist, meta, success = accumulate_dwell_time(
                path, rho_edges, z_edges, chunk_size=args.chunk_size,
                exclude_lost=args.exclude_lost,
                truncate_atm=args.truncate_atm, r_atmosphere=args.r_atm)

            if not success or hist is None:
                if meta.get("hit_atmosphere") and args.exclude_lost:
                    print("hit atmosphere, excluded")
                else:
                    print("no PS data, skipped")
                n_skipped += 1
                continue

            if hist.sum() == 0:
                print("empty trajectory, skipped")
                n_skipped += 1
                continue

            # Store unweighted histogram
            orbit_store.append({
                "key":            key,
                "energy_eV":      E,
                "L":              L,
                "pitch_deg":      pitch,
                "hit_atmosphere": meta.get("hit_atmosphere", False),
                "hist":           hist,
            })
            existing_keys.add(key)
            n_new += 1

            E_str = f"E={E/1e6:.0f}MeV" if E else "E=?"
            L_str = f"L={L}" if L else "L=?"
            p_str = f"α={pitch:.0f}°" if pitch else "α=?"
            dwell = hist.sum()
            print(f"{E_str}  {L_str}  {p_str}  dwell={dwell:.0e}")

        print(f"\n  New orbits added: {n_new}   Skipped: {n_skipped}", end="")
        if n_open > 0:
            print(f"  (OPEN: {n_open})", end="")
        if n_already > 0:
            print(f"  (already in state: {n_already})", end="")
        print()

    elif not orbit_store:
        print("No h5 files found and no state loaded. Nothing to do.")
        print("Specify --group <name>, h5 file paths, or --load-state <file>.")
        sys.exit(1)

    # ── Save state if requested ────────────────────────────────────────
    if args.save_state:
        if not orbit_store:
            print("No orbits to save.")
        else:
            save_state(args.save_state, orbit_store, rho_edges, z_edges)

    # ── Radial population weight (imposed L-dependent abundance) ───────
    radial_weight_fn = make_radial_weight(index=args.radial_index,
                                          L_ref=args.radial_L_ref,
                                          profile_path=args.radial_profile)
    if args.radial_profile:
        print(f"\nRadial weighting: profile from {args.radial_profile} "
              f"(log-space interpolation).")
    elif args.radial_index != 0.0:
        print(f"\nRadial weighting: power law w(L)=(L/{args.radial_L_ref:g})"
              f"^(-{args.radial_index:g}).")

    # ── Build dwell map from orbit store ───────────────────────────────
    dwell_2d, n_used, energies_used, pitches_used, L_shells_used = \
        build_dwell_from_store(orbit_store, rho_edges, z_edges,
                              args.spectral_index,
                              E_min=args.E_min, E_max=args.E_max,
                              radial_weight_fn=radial_weight_fn,
                              use_dE=args.dE_weight,
                              exclude_lost=args.exclude_lost)

    print(f"\n{'='*60}")
    print(f"  Orbits in store: {len(orbit_store)}   Used for plot: {n_used}")
    print(f"  Energies: {sorted(e/1e6 for e in energies_used)} MeV")
    print(f"  Pitch angles: {sorted(pitches_used)}°")
    print(f"  L-shells: {sorted(L_shells_used)}")
    print(f"{'='*60}")

    if n_used == 0:
        print("No valid orbits. Nothing to plot.")
        sys.exit(1)

    # ── Per-energy maps (--per-energy) ───────────────────────────────
    if args.per_energy and len(energies_used) > 1:
        print(f"\nGenerating per-energy maps for {len(energies_used)} energies...")
        output_base, output_ext = os.path.splitext(args.output)

        for E_target in sorted(energies_used):
            E_MeV = E_target / 1e6
            per_e_dwell, per_e_count, _, _, _ = \
                build_dwell_from_store(orbit_store, rho_edges, z_edges,
                                      args.spectral_index, E_target=E_target,
                                      radial_weight_fn=radial_weight_fn,
                                      use_dE=args.dE_weight,
                                      exclude_lost=args.exclude_lost)

            if per_e_count > 0:
                per_e_path = f"{output_base}_{E_MeV:.0f}MeV{output_ext}"
                title_e = (f"PS Meridian Dwell Map — {E_MeV:.0f} MeV Proton\n"
                           f"{per_e_count} orbits, pitch-angle weighted")
                plot_dwell_map(per_e_dwell, rho_edges, z_edges,
                              save_path=per_e_path, title=title_e,
                              rescale_peak=args.rescale_peak,
                              vmin=args.vmin, vmax=args.vmax,
                              plot_rho_max=args.plot_rho_max,
                              plot_z_max=args.plot_z_max,
                              r_mask=args.r_mask,
                              smooth_sigma=args.smooth,
                              smooth_space=args.smooth_space,
                              contours=args.contours)

    # ── Combined map ─────────────────────────────────────────────────
    E_range = f"{min(energies_used)/1e6:.0f}–{max(energies_used)/1e6:.0f} MeV" if energies_used else ""
    spec_label = f"γ={args.spectral_index}"
    title = (f"PS Meridian Flux Map — Proton > {args.E_min/1e6:.0f} MeV\n"
             f"{n_used} orbits, {E_range}, "
             f"spectrum: {spec_label}"
             if args.E_min else
             f"PS Meridian Flux Map — Proton ({E_range})\n"
             f"{n_used} orbits, spectrum: {spec_label}")

    plot_dwell_map(dwell_2d, rho_edges, z_edges,
                  save_path=args.output, title=title,
                  rescale_peak=args.rescale_peak,
                  vmin=args.vmin, vmax=args.vmax,
                  plot_rho_max=args.plot_rho_max,
                  plot_z_max=args.plot_z_max,
                  r_mask=args.r_mask,
                  smooth_sigma=args.smooth,
                  smooth_space=args.smooth_space,
                  contours=args.contours)


if __name__ == "__main__":
    main()
