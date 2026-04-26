"""
Shared physical constants for the PS_Method project.

npfloat is resolved at import time via builtins (set by project_setup.py).
Falls back to np.float64 if builtins.npfloat has not been set yet.
"""

import builtins
import numpy as np

try:
    npfloat = builtins.npfloat
except AttributeError:
    npfloat = np.float64

# ===== Physical Constants (SI) =====
q_e      = npfloat(-1.602176634e-19)       # electron charge (C)
m_e      = npfloat(9.1093837139e-31)       # electron mass (kg)
m_p      = npfloat(1.67262192595e-27)      # proton mass (kg)
evtoj    = npfloat(1.602176634e-19)        # eV to Joules conversion
spdlight = npfloat(299792458.0)            # speed of light (m/s)
RE       = npfloat(6378137.0)              # Earth radius (m)
B_0      = npfloat(3.12e-5)                # dipole surface field strength (T)
