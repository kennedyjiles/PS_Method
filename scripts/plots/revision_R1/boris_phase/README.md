# Figure R1 (reply to R1.1): Boris phase error and mu statistics on a REGULAR orbit

Particle (particle_322bba.yml; identical ICs to run 322bba, bitwise-checked against its h5 start_state):
100 MeV proton, alpha_eq = 30 deg, gyrophase 15 deg, x0 = 2.68128 R_E (just inside the Tu et al. 2014 loss onset, L ~ 2.7).
Paper constants (R_E = 6378137 m) and dt rounding (round_decimals = 2) via boris_section.setup_from_yml. 1e5 gyroperiods (~75 min).

Pipeline:
  python runs.py ps 65|130|260        fixed-step PS (ps_integrate, adaptive order)   -> out/ps*.npz
  python runs.py ps 130 pert          PS130 with x0*(1+1e-12)                        -> out/ps130_pert.npz
  python runs.py boris 65|130|260|650 Boris (boris_section.boris_chunk)              -> out/boris*.npz
  python runs.py adaptive <h5> <tag>  crossings from run.py adaptive-step PS (322bba_ps65/130, 2e4 T)
  python gyrophase.py <n> 2000        gyrophase Boris-PS at identical times (same dt) -> out/gyrophase_n*.npz
  python analyze.py                   crossing-by-crossing comparison table
  python plot.py                      -> FigR1_boris_phase.png

Checks (2026-10-18):
- PS converged: PS65/PS130/PS260 and adaptive-step PS65/PS130 agree crossing-by-crossing (17114 crossings) to the
  linear-interpolation noise of the crossing finder (crossing time ~1e-7 T, mu ~1e-3 at 65 steps, ~3e-4 at 130); no growth over 1e5 T.
  No step hit the order cap (max order 22/17/15 at 65/130/260). PS dE/E <= 5e-12, dPphi/Pphi <= 3e-13.
- Regular orbit: PS130 with a 1e-12 perturbation tracks PS130 to the same noise level through 1e5 T.
- Boris (E exact by construction; Pphi range 2.2e-5 / 5.5e-6 / 1.4e-6 / 2.3e-7 at 65/130/260/650):
  * gyrophase error at identical times scales as dt^2 (at 10 T: 0.64, 0.16, 0.042, 0.0069 rad); reaches ~1 rad by
    ~30 T (65), ~100 T (130), ~1e3 T (650).
  * bounce-phase (equatorial-crossing time) error grows linearly and scales as dt^2: 69, 16, 3.9, 0.66 T at 1e5 T.
  * mu at equatorial crossings (normalised to launch mu): Boris settles onto a DIFFERENT torus within the first 1e3 T and stays there:
        run       mean    max    (PS: mean 0.860, max 1.137, min 0.418)
        Boris65   0.688   0.898
        Boris130  0.805   1.072
        Boris260  0.844   1.122
        Boris650  0.858   1.134  (distribution matches PS, but the crossing-by-crossing mu sequence decorrelates after ~1e3 T)

Second regular orbit (particle_a5b47e.yml: x0 = 2.5, gyrophase 0, 2e4 T; out_a5b47e/; run with PYML/POUT env vars):
- PS130/PS260/PS130+1e-12 agree (crossing time 5e-8 T, mu 5e-4 = interpolation noise); regular.
- Boris mu DISTRIBUTION matches PS at all step sizes (mean 0.789-0.791 vs 0.789, max 1.159-1.160 vs 1.159),
  but the crossing-by-crossing mu sequence decorrelates (max |dmu| ~0.7 at every step size within 2e4 T).
=> Gyrophase drift (dt^2) and loss of the physical mu-kick sequence are generic; a biased mu DISTRIBUTION appeared for the
   orbit closer to the loss onset (322bba, L~2.68) but not for the deeper one (a5b47e, L=2.5). Regime-dependent: do not generalise
   the distribution result from one orbit.

Surface of section (section_overlay.py -> FigR1_section_overlay.png), PS260 vs Boris; section coords scaled by arc size (0.03):
  run       median offset from PS curve   fraction of PS curve NOT visited (>1% of arc)
  PS65      7.0e-4 (interp noise)          0%
  PS130     1.4e-4                         0%
  Boris65   4.3e-3                         52%
  Boris130  1.6e-3                         28%
  Boris260  5.5e-4                         45%
  Boris650  2.4e-4                         0%
=> Boris points lie ON the PS curve (offset shrinks with step), but WHICH PART of the curve is visited does not converge
   monotonically (130 better than 260); only at 650 steps does Boris cover the same arc as PS. Same non-monotonic pattern as the
   mu-distribution L1 differences (1.35, 0.71, 0.89, 0.04). Orbit is near-resonant, so coverage is sensitive to Boris's frequency error.

Boris invariants vs coverage (whole-run P_phi range from boris_chunk; flat after ~1e3 T, no secular growth):
  steps/T   dE/E      P_phi range   PS curve not visited
  65        ~1e-13    2.2e-5        52%
  130       ~1e-13    5.5e-6        28%
  260       ~1e-13    1.4e-6        45%
  650       ~1e-13    2.3e-7        0%
=> invariants improve monotonically (~dt^2) while orbit coverage does not (260 worse than 130).
Runtime, single thread, 1e4 T: PS20 0.29 s, PS65 0.44 s, Boris65 0.02 s, Boris260 0.09 s, Boris650 0.23 s
(Boris timing = boris_chunk, crossings only; PS = ps_integrate, which stores full state history) -> Boris650 ~ PS20; no speed claim.

Diagnosability summary (322bba, 1e5 T):
  run       bounce period (T)  rel. err   P_phi range  mu@crossing mean / max   mu-hist L1 vs PS   PS curve not visited
  PS20      11.685913          -          4e-14        0.8603 / 1.1374          (ref)              0%
  PS260     11.685913          -          3e-13        0.8603 / 1.1375          ~0                 0%
  Boris65   11.694040          7.0e-4     2.2e-5       0.688 / 0.898            1.35               52%
  Boris130  11.687798          1.6e-4     5.5e-6       0.805 / 1.072            0.71               28%
  Boris260  11.686364          3.9e-5     1.4e-6       0.844 / 1.122            0.89               45%
  Boris650  11.685990          6.6e-6     2.3e-7       0.858 / 1.134            0.04               0%
=> Boris frequency (bounce-period) and P_phi errors shrink smoothly (~dt^2); mean/max mu converge monotonically;
   but the mu distribution and section coverage do NOT (260 worse than 130). Error size is therefore not a reliable
   indicator of whether the orbit statistics are right; only a step-refinement study of the mu statistics reveals it.
   (Per-bounce gyrophase advance at crossings is incoherent on this orbit (coherence ~0.03), so no clean rotation number.)
