"""
Kinetic energy conservation + small math primitives shared by constb / hyperb.

    energy_drift               — KE drift over time (njit, float64 path)
    energy_drift_pure          — KE drift, non-JIT, any dtype (e.g. float128)
    extract_v                  — pull (vx, vy, vz) from PS solution array
    trajectory_error_xy        — XY distance from a reference, normalized by scale
"""

import numpy as np
from . import utils as ul

half = ul.npfloat(0.5)
two  = ul.npfloat(2.0)

@ul.maybe_njit
def energy_drift(vx, vy, vz):
    KE = half * (vx**two + vy**two + vz**two)
    return np.abs(KE - KE[0]) / KE[0]

def energy_drift_pure(vx, vy, vz):
    KE = 0.5 * (vx**2 + vy**2 + vz**2)
    return np.abs(KE - KE[0]) / KE[0]

def extract_v(sol):
    """Pull the velocity rows (vx, vy, vz) from a (>=6, N) solution array."""
    return sol[3], sol[4], sol[5]

def trajectory_error_xy(sol, x_ref, y_ref, scale):
    return np.sqrt((sol[0] - x_ref)**2 + (sol[1] - y_ref)**2) / scale
