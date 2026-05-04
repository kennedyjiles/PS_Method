"""
Kinetic energy conservation functions shared by constb and hyperb drivers.

    kinetic_energy         — KE from velocity components
    compute_energy_drift   — relative KE drift over time
    extract_v              — pull (vx, vy, vz) from PS solution array
"""

import numpy as np
from . import utils as ul

half = ul.npfloat(0.5)
two  = ul.npfloat(2.0)

@ul.maybe_njit
def kinetic_energy(vx, vy, vz, m=ul.npfloat(1.0)):
    return half * m * (vx**two + vy**two + vz**two)

@ul.maybe_njit
def compute_energy_drift(vx, vy, vz):
    KE = kinetic_energy(vx, vy, vz)
    return np.abs(KE - KE[0]) / KE[0]

@ul.maybe_njit
def extract_v(sol):  # assumes PS output has x, y, z, vx, vy, vz as initial entries
    return sol[3], sol[4], sol[5]
