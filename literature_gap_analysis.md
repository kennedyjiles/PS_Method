# Literature Gap Analysis: High-Precision Full-Orbit Tracing for Radiation Belt Physics

*Deep literature scan, July 2026. Five parallel search angles, ~60 distinct queries, primary-source fetches of all scoop-risk papers. Claims below were cross-verified across independent search agents; confidence flags noted where verification was limited by paywalls.*

---

## Executive summary

The proposed thesis contribution — **certify a Parker-Sochacki (PSM) tracer against the Dragt–Finn benchmarks, modernize Dragt's 1965 empirical adiabaticity boundary with quantitative chaos indicators, and publish μ-scattering/diffusion rates for dipole-trapped protons from drift-free long-time integrations** — occupies a genuinely open niche. No located paper does any of the three together, and each individually is either absent from the literature or done only partially with methods weaker than what PSM enables.

The five key verdicts:

1. **Young et al. (2002, 2008) remains the de facto FLC scattering input 24 years on.** Every modern proton-belt model imports it (or the Tu et al. 2014 onset criterion) essentially unmodified. The only systematic test-particle validation is Cai et al. (2023) — Boris pusher, 500 gyroperiods, 200 particles per case, diffusion fits restricted to the first 100 gyroperiods because ⟨Δμ²⟩ departs from linearity.
2. **Dragt's W₀² = 0.012μ² boundary has never been quantitatively re-derived.** No paper applies FLI/SALI/GALI/frequency-map analysis to the Störmer problem; no paper confronts the 0.012 number or Figure 32 directly. Two partial neighbors exist (see Scoop Risk).
3. **No high-order Taylor-series/PSM-class integrator has been applied to magnetospheric tracing by anyone other than Jiles & Weigel (arXiv:2604.20876).** The field runs on RK4/adaptive-RK and Boris; the mathematics community's recent high-accuracy integrator advances (Hairer–Lubich filtered Boris, uniformly accurate schemes) have not been adopted by space-physics tracer codes.
4. **No 2015–2026 paper publishes updated μ-diffusion coefficients for dipole-trapped multi-MeV protons from long-time full-orbit integrations.** Confirmed independently by two agents as a negative search result.
5. **Proton-belt modelers explicitly flag the scattering-rate input as unreliable.** Selesnick et al. (2019) tested five loss mechanisms including FLC and found none viable; Lozinski et al. (2024) hand-tuned the curvature-radius criterion to match Van Allen Probes data; Tu et al. (2014) found cumulative-μ-scattering results in "significant disagreement with theoretical predictions."

---

## Angle 1: FLC / μ-scattering state of the art

**Canonical lineage:** Il'in & Il'ina (1978, analytic exponential μ-jumps) → Birmingham (1984, diffusion form) → Delcourt et al. (1994, 1996, centrifugal impulse model) → Anderson et al. (1997) → **Young, Denton, Anderson & Hudson (2002)** empirical μ-jump model fit to half-bounce test-particle crossings in T89 → **Young et al. (2008)** converts to pitch-angle diffusion coefficients D_αα.

> Citation hygiene note: the 2002/2008 papers are Young, **Denton**, Anderson & Hudson — not "Young, Ukhorskiy." Worth double-checking in the thesis bibliography.

**Modern usage and validation:**

| Paper | Contribution | Integrator / run length |
|---|---|---|
| Cai, Li, Wu & Tao 2023 (JGR, [arXiv:2303.06687](https://arxiv.org/abs/2303.06687)) | First systematic validation; Young-2002 beats Birmingham at α₀≈30–60° | **Boris**, dt = T_g/50, 500 T_g runs, 200 particles/case; fits limited to ≤100 T_g |
| Cai & Zhu 2024 ([arXiv:2403.09825](https://arxiv.org/abs/2403.09825)) | Young model breaks down at dayside off-equatorial (Shabansky) minima | test particles |
| Wei et al. 2025 (JGR [10.1029/2024JA033422](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2024JA033422)) | "Loss Cone Offset Method" — first methodological alternative to Young since 2002 | Lorentz integration in TS05, scheme not stated in accessible text |
| Selesnick & Looper 2023 (JGR [10.1029/2023JA031509](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2023JA031509)) | Proton-belt outer boundary from diffusive FLC loss balance | analytic/parameterized rates in T89+dipole |
| Lozinski et al. 2024 (JGR [10.1029/2023JA032377](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2023JA032377)) | R_c(Dst, L*) onset condition in BAS-PRO | empirically retuned to match data |
| Lozinski et al. 2025 (JGR [10.1029/2025JA033871](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2025JA033871)) | TRIPS code, shock redistribution | **Boris**, ~180 s event runs |

**The integrator-fidelity gap (citable):** No paper in this literature quantifies numerical μ or energy drift of its integrator as a limitation on the FLC-scattering estimates — despite the parameterizations being built directly from measured δμ per equator crossing, where un-quantified numerical μ drift is directly confounding. The only integrator-accuracy sentence found in the whole literature (Cai et al. 2023): *"The Boris algorithm is widely used due to its exceptional long-term accuracy"* — asserted, not measured. Run lengths never exceed a few hundred gyroperiods.

**Stated limitations (verbatim quotes with sources):**
- Cai et al. 2023: *"The ⟨Δμ*²⟩ gradually deviates from a linear function of t after approximately 100 T_g"* — diffusion fits cannot use longer baselines. Also: ε upper threshold set to 1, *"beyond which the multiple jump of μ is regarded as small chaotic scattering and the diffusion approximation can be invalid."*
- Cai & Zhu 2024: Young 2002 *"is appropriate for situations where particles are only scattered once in a half bounce period"* — fails at dual magnetic minima.
- Tu et al. 2014: *"results after cumulative μ scattering can show significant disagreement with theoretical predictions."*

---

## Angle 2: Has Dragt's boundary been modernized? — **No.**

**What exists:**
- **Kuznetsov & Yushkov 2002** (Plasma Phys. Rep. 28, 342, [Springer](https://link.springer.com/article/10.1134/1.1469175)) — the closest prior art. Quasi-adiabatic *reduced mapping* model (not full-orbit chaos indicators): reversible μ fluctuations for adiabaticity parameter χ ≳ 0.01, irreversible exponential jumps for χ ≳ 0.1; stability boundary as a function of χ **and pitch angle**; cites Dragt 1965 directly.
- **Bonfim, Griffiths & Hinkley 2000** (Int. J. Bif. Chaos 10, 265) — Lyapunov spectrum vs energy: quasiperiodic → chaotic → hyperchaotic staging.
- **Xie & Liu 2020** (Chaos 30, [10.1063/5.0028644](https://pubs.aip.org/aip/cha/article-abstract/30/12/123108/1074510), [arXiv:2011.11249](https://arxiv.org/abs/2011.11249)); **Liu et al. 2022** (Chaos 32, 043104); **Pang, Liu & Liu 2023** (CMDA, [arXiv:2302.07075](https://arxiv.org/abs/2302.07075)) — Lyapunov-exponent scans of trapped-orbit phase space; fraction of quasiperiodic orbits vs energy; escape times around the trapping threshold. Framed as dynamical-astronomy orbit taxonomy.

**What does not exist (verified negative results):**
- No application of FLI, SALI/GALI, MEGNO, or frequency-map analysis to the Störmer problem — anywhere.
- No chaos-indicator map over the **(W₀², equatorial pitch angle)** plane for full dipole trapped orbits.
- No paper confronting Dragt's 0.012 value or Figure 32. The Lyapunov-based line (Xie/Liu school) never computes μ along orbits and never connects to the adiabatic-invariant framework; the μ-jump line (Il'in → Kuznetsov–Yushkov) never uses full-orbit chaos indicators. **The two halves have never been joined.**

---

## Angle 3: Integrators in magnetospheric tracing

**What the field uses:** RK4/adaptive RK on guiding-center equations plus Boris (or RK) on full Lorentz orbits. Representative: Kress et al. rbelt (RK4/GC hybrid), Elkington (adaptive RK GC), Sorathia/Ukhorskiy CHIMP (GC with dynamic switch to Boris when ε = ρ_g/L > 10⁻², random gyrophase on switch), Lozinski TRIPS (Boris), geomagnetic-cutoff codes (RK4 at ~T_g/100; a 2024 Russian paper is notable for *only now* moving cutoff tracing from RK4 to Boris).

**Boris/symplectic theory:** Qin et al. 2013 (phase-space volume conservation explains Boris's bounded energy error); He/Sun/Liu/Qin volume-preserving family; Hairer & Lubich 2018 (energy behavior of Boris), Hairer–Lubich–Wang 2020 (filtered Boris for strong fields); Tao 2016 (explicit symplectic). These advances are **largely unadopted** by space-physics tracer codes.

**Taylor-series/PSM:** heyoka (Biscani & Izzo 2021) and Jorba's `taylor` are celestial-mechanics-only. **The only application of PSM to charged-particle motion in magnetic fields found anywhere is Jiles & Weigel, [arXiv:2604.20876](https://arxiv.org/abs/2604.20876)** (4–13 orders of magnitude better KE conservation than RK methods on dipole orbits; RKG failed all electron dipole runs). Nearest misses: a 2018 Groningen BSc thesis (Taylor series for charged-particle propagation, not peer-reviewed); Boris-SDC (Winkel et al. 2015, accelerator context); Tan/Smith/Rackauckas 2026 (modern Taylor tooling in Julia, no charged-particle application).

**Work-precision comparisons:** Ripperda et al. 2018 (ApJS) is the canonical pusher comparison and includes a dipole test case, but targets astrophysical regimes, no Taylor-class method, and no bounce/drift-timescale radiation-belt focus. **No published work-precision study of dipole trapped orbits including a Taylor-class method exists besides the Jiles & Weigel preprint.**

---

## Angle 4: Recent Störmer chaos activity (2018–2026)

- **Dubbers 2026** ([arXiv:2603.29459](https://arxiv.org/abs/2603.29459), posted 31 Mar 2026, Heidelberg) — pedagogical survey of nonadiabatic Störmer transport motivated by neutron-decay spectrometers. Computes no chaos indicators itself (its phase-space chaos map is reproduced from Xie & Liu 2020); explicitly poses *"how much smaller than one should the adiabaticity coefficient be"* as an **open question**. → Citable evidence the gap exists, and evidence the Störmer-chaos framing is drawing fresh attention.
- **Pang, Liu & Liu 2023** — escape times below/above the trapping threshold; periodic-orbit families; self-similar structure. Closest existing "escape above threshold" study.
- **Brizard & Markowski 2022** (Phys. Plasmas, [arXiv:2111.05353](https://arxiv.org/abs/2111.05353)) — guiding-center validity in the dipole tested against full orbits, but **restricted to regular orbits** with adiabatically invariant μ.
- Adjacent: FTLE–escape-time power law in a magnetic bottle (2025, [arXiv:2502.01726](https://arxiv.org/abs/2502.01726)); drift-orbit bifurcation modeling (Huang et al. 2022, 2023); stochastic-dissipative Störmer problem (Harko et al. 2025); lab-dipole chaos (Saitoh & Tanioka 2022).

---

## Angle 5: What proton-belt modelers say they need

- Selesnick et al. 2007: model outer boundary = empirical Dst-dependent trapping limit, not derived scattering physics.
- Selesnick et al. 2019 ([10.1029/2019JA026754](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2019JA026754)): five candidate loss mechanisms tested (incl. FLC), *"none are found viable"* to explain observed low-L proton decay — an open loss-physics problem.
- Lozinski et al. 2024: had to *"refine our expression for R_C to achieve agreement with Van Allen Probes observations"*; residual *"~5 MeV proton source not accounted for."*
- Engel et al. 2015/2016: *"field line curvature scattering by itself is insufficient to explain proton loss in the inner belt."*
- AP9/IRENE official limitations: large proton uncertainties <1000 km altitude and >400 MeV in the inner zone; Ginet et al. 2013 frames flux uncertainty as the direct driver of shielding margins.
- Lejosne & Kollmann 2020: proton D_LL uncertainty → *"several orders of magnitude difference in modeled steady state phase space density."*
- Xu et al. 2026 (new NASA dynamic proton model): explicitly excludes storm-time non-adiabatic outer-edge physics — named as its own limitation.

---

## Scoop-risk assessment

| Threat | Level | Basis |
|---|---|---|
| **Dubbers 2026** (arXiv:2603.29459) | **Low** | No indicators computed, no μ tracking, no diffusion rates; promised follow-up targets spectrometer spectra, not dipole μ-diffusion. Cite it *for* the gap. |
| **Xie/Liu/Pang group** (Chaos 2020, 2022; CMDA 2023) | **Moderate** | Most capable active group; LCE scans + escape times already published. But: no μ computation, no adiabatic-invariant connection, no Dragt engagement, standard-precision tools, orbit-taxonomy framing. The CMDA 2023 escape-time work shows they *could* pivot toward transport rates — timeliness matters. |
| **Kuznetsov & Yushkov 2002** | Prior art, not scoop | Boundary from a reduced quasi-adiabatic map, χ and pitch-angle dependent. Must be cited and compared against; strengthens rather than undercuts the contribution (their mapping prediction vs. your full-orbit measurement). |
| **Wei et al. 2025** | Watch | New FLC methodology (loss-cone offset) in TS05; not dipole-fundamental, not high-precision-focused, but shows the FLC-parameterization space is active again. |
| **Tan/Rackauckas 2026 Taylor tooling** | Watch (methods side) | Modern Taylor integrators being mainstreamed in Julia; domain application to plasma still open. |

---

## Recommended thesis positioning

**The defensible one-paragraph frame:** Diffusion-based proton radiation-belt models require non-adiabatic scattering inputs (boundary criteria and μ/pitch-angle diffusion rates) that trace back to a 1965 eyeball-classified boundary (Dragt) and a 2002 empirical fit (Young et al.) validated only recently, with a Boris pusher, over a few hundred gyroperiods. The modelers themselves report the inputs failing (Selesnick 2019; Lozinski 2024; Tu 2014). Meanwhile, no integrator in use can guarantee invariant fidelity over the 10⁴–10⁶ bounce timescales that μ-diffusion statistics actually require, and no one has quantified that confound. This thesis develops and certifies (via the Dragt–Finn fixed point, eigenvalue, and homoclinic benchmarks) a Parker-Sochacki tracer whose invariant error stays bounded near machine precision over ≥10⁵ Lyapunov times, then uses it to (i) replace Dragt's empirical adiabaticity boundary with a quantitative chaos-indicator map over the (W₀², pitch angle) plane, and (ii) publish μ-jump statistics and diffusion rates for dipole-trapped protons with the numerical confound provably controlled.

**Priority order (impact per effort):**
1. Certification chapter (Dragt–Finn benchmarks — already largely built).
2. Chaos-indicator map over (W₀², α_eq) + direct confrontation with the 0.012μ² line and Kuznetsov–Yushkov's χ-pitch boundary. *Differentiator vs. Xie/Liu: lead with μ and the adiabatic-invariant framework, not orbit taxonomy.*
3. μ-jump statistics / D_αα from long-baseline integrations; compare against Young 2008 and Cai 2023's ≤100 T_g fits — specifically test whether the ⟨Δμ²⟩ nonlinearity Cai saw at 100 T_g is physics or numerics.
4. (Extension) escape-time statistics above W₀² = 1/16, positioned against Pang et al. 2023.

**Defensive requirements a referee will impose:**
- The comparison set must include **Boris and a symplectic/volume-preserving method**, not just RK45 — beating RK alone will be called a strawman (Qin 2013 and Hairer–Lubich give Boris bounded energy error; the honest PSM claim is orders-of-magnitude tighter *per-step accuracy at comparable cost*, plus μ fidelity, which volume preservation does not guarantee).
- Cite and compare against Kuznetsov & Yushkov 2002 — the boundary exists in reduced-map form; the contribution is measuring it from certified full orbits and resolving it in μ.
- Dipole-only results transfer qualitatively; frame stretched/compressed (T89-class) fields as future work, per the Cai & Zhu 2024 Shabansky caveat.
- Timeliness: the Xie/Liu group is active and Dubbers 2026 shows the topic warming. Publishing the certification + boundary papers promptly protects the core claim.

---

*Caveats: Young 2002/2008, Tu 2014, and Selesnick & Looper 2023 are Wiley-paywalled; their integrator details could not be confirmed from accessible text and should be verified via institutional access before asserting them in print. The Sorathia/CHIMP integrator details rest on consistent secondary excerpts. Xie & Liu 2020 is indexed with two article numbers (023108/123108) in different sources; verify against the journal page.*
