"""
Kinetic energy conservation diagnostics shared by constb and hyperb drivers.

    kinetic_energy         — KE from velocity components
    compute_energy_drift   — relative KE drift over time
    extract_v              — pull (vx, vy, vz) from PS solution array
"""

import numpy as np
from ps_method.utils import npfloat, maybe_njit

half = npfloat(0.5)
two  = npfloat(2.0)


@maybe_njit
def kinetic_energy(vx, vy, vz, m=npfloat(1.0)):
    return half * m * (vx**two + vy**two + vz**two)

@maybe_njit
def compute_energy_drift(vx, vy, vz):
    KE = kinetic_energy(vx, vy, vz)
    return np.abs(KE - KE[0]) / KE[0]

@maybe_njit
def extract_v(sol):  # assumes PS output has x, y, z, vx, vy, vz as initial entries
    return sol[3], sol[4], sol[5]
