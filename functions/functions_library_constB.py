import numpy as np
from numba import njit
import json, hashlib, os, h5py
from functions.functions_library_universal import maybe_njit, npfloat

one = npfloat(1.0)

@maybe_njit
def PS_constantB_adaptive(order_max, steps, initial_pos_vel, timedelta, Bfield, qoverm, tol):
    n_total = 6        # x, y, z, v_x, v_y, v_z
    final_coeff_matrix = np.zeros((n_total, steps + 1), dtype=npfloat)  # array to store everything
    final_coeff_matrix[:, 0] = initial_pos_vel          # initialize with intial position velocity
    oip1 = one / (one + np.arange(order_max))
    orders_used = np.zeros(steps + 1, dtype=np.int32)
    
    #Labeling indices to visually track more easily
    x, y, z = 0, 1, 2
    vx, vy, vz = 3, 4, 5
    
    for j in range(1, steps + 1):
        c = np.zeros((n_total, order_max + 1), dtype=npfloat)    # array storage for loop
        c[:, 0] = final_coeff_matrix[:, j - 1]
        power = timedelta
        sum_terms = np.zeros(n_total, dtype=npfloat)
        max_contrib = tol + one
        i = 0

        while max_contrib > tol and i < order_max:
            # Position derivatives from velocity
            c[x, i+1] = oip1[i] * c[vx,i]
            c[y, i+1] = oip1[i] * c[vy,i]
            c[z, i+1] = oip1[i] * c[vz,i]

            # Velocity derivatives from lorentz force
            c[vx, i+1] = oip1[i]*qoverm*(Bfield[2]*c[vy,i]-Bfield[1]*c[vz,i])
            c[vy, i+1] = oip1[i]*qoverm*(Bfield[0]*c[vz,i]-Bfield[2]*c[vx,i])
            c[vz, i+1] = oip1[i]*qoverm*(Bfield[1]*c[vx,i]-Bfield[0]*c[vy,i])

            sum_terms += c[:, i+1]* power # just keeps adding these on until PS prder is reached, final added to permanent matrix below
            max_contrib = np.abs(c[:, i+1]).max()
            power *= timedelta
            i += 1

        final_coeff_matrix[:, j] = final_coeff_matrix[:, j - 1] + sum_terms        
        orders_used[j] = i

    return final_coeff_matrix, orders_used

def analytical_constantB(t, d, Bfield, qoverm):
    x0, y0, z0, vx0, vy0, vz0 = d
    omega = qoverm * Bfield[2]  # Cyclotron frequency
    
    sin_ot = np.sin(omega * t)
    cos_ot = np.cos(omega * t)
    
    x_t = x0 + (vy0 / omega) * (1 - cos_ot) + (vx0 / omega) * sin_ot
    y_t = y0 - (vx0 / omega) * (1 - cos_ot) + (vy0 / omega) * sin_ot
    z_t = z0 + vz0 * t

    vx_t = vx0 * cos_ot - vy0 * sin_ot
    vy_t = vy0 * cos_ot + vx0 * sin_ot
    vz_t = vz0 * np.ones_like(t)

    return np.vstack((x_t, y_t, z_t, vx_t, vy_t, vz_t))

@maybe_njit
def lorentz_force_constB(t, d, Bfield, qoverm):
    x, y, z, vx, vy, vz = d  
    dvx = qoverm * (vy * Bfield[2] - vz * Bfield[1])
    dvy = qoverm * (vz * Bfield[0] - vx * Bfield[2])
    dvz = qoverm * (vx * Bfield[1] - vy * Bfield[0])

    return np.array([vx, vy, vz, dvx, dvy, dvz])

# ====================================
# === Read/Write Functions for hdf ===
# ====================================

def _to_serializable(x):
    """Make numpy / custom scalars json-serializable."""
    import numpy as _np
    if isinstance(x, (_np.floating, _np.float32, _np.float64)):
        return float(x)
    if isinstance(x, (_np.integer,)):
        return int(x)
    if isinstance(x, (_np.ndarray,)):
        return x.tolist()
    return x

def get_run_params(USE_RK45, USE_RK4, KE_particle, rtol_rk45, atol_rk45,
                   mass_si, q_e, B_0,
                   x_initial, y_initial, z_initial,
                   pitch_deg, phi_deg,
                   norm_time, ps_step, rk4_step,
                   PS_order, tol, qoverm):
    """Collect all knobs that define a 'unique' run."""
    return {
        # toggles
        "USE_RK45": bool(USE_RK45),
        "USE_RK4":  bool(USE_RK4),


        # physics & normalization
        "KE_particle": _to_serializable(KE_particle),
        "mass_si": _to_serializable(mass_si),
        "q_e": _to_serializable(q_e),
        "B_0": _to_serializable(B_0),

        # initial conditions 
        "x_initial": _to_serializable(x_initial),
        "y_initial": _to_serializable(y_initial),
        "z_initial": _to_serializable(z_initial),
        "pitch_deg": _to_serializable(pitch_deg),
        "phi_deg": _to_serializable(phi_deg),

        # times / steps
        "norm_time": _to_serializable(norm_time),
        "ps_step": _to_serializable(ps_step),
        "rk4_step": _to_serializable(rk4_step),

        # PS & solver knobs
        "PS_order": int(PS_order),
        "tol": _to_serializable(tol),
        "rtol_rk45": _to_serializable(rtol_rk45),
        "atol_rk45": _to_serializable(atol_rk45),

        # charge/mass normalization used in RHS
        "qoverm": _to_serializable(qoverm),
    }

def run_hash(params: dict) -> str:
    j = json.dumps(params, sort_keys=True, default=_to_serializable, separators=(",",":"))
    return hashlib.sha1(j.encode("utf-8")).hexdigest()[:16]

def h5_path_for(params, output_folder):
    return os.path.join(output_folder, f"run_{run_hash(params)}.h5")

def save_results_h5(h5_path, params, results):
    with h5py.File(h5_path, "w") as f:
        # store params as a single JSON attribute on root
        f.attrs["params_json"] = json.dumps(params, sort_keys=True, default=_to_serializable)

        for k in ("ps","rk4","rk45"):
            if k in results and results[k] is not None:
                grp = f.create_group(k)
                for name, arr in results[k].items():
                    if arr is None: 
                        continue
                    grp.create_dataset(name, data=arr, compression="gzip", compression_opts=2)

        # meta info
        meta = results.get("meta", {})
        gmeta = f.create_group("meta")
        # store timing dict as attrs
        for mk, mv in meta.get("timing", {}).items():
            gmeta.attrs[f"timing_{mk}"] = float(mv)
        # scalar attrs
        for sk in ("physical_time","norm_time","percent_c","particle_label"):
            if sk in meta:
                gmeta.attrs[sk] = meta[sk]

def load_results_h5(h5_path):
    with h5py.File(h5_path, "r") as f:
        loaded = {"meta": {"timing": {}}}
        # params
        loaded["params"] = json.loads(f.attrs["params_json"])

        # helper to pull groups
        def _read_grp(name):
            if name not in f: return None
            g = f[name]
            out = {}
            for ds in g:
                out[ds] = g[ds][...]
            return out

        for k in ("ps","rk4","rk45"):
            loaded[k] = _read_grp(k)

        # meta attrs
        gmeta = f["meta"]
        for a in gmeta.attrs:
            if a.startswith("timing_"):
                loaded["meta"]["timing"][a.replace("timing_","")] = gmeta.attrs[a]
            else:
                loaded["meta"][a] = gmeta.attrs[a]

        return loaded
