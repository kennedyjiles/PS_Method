import numpy as np
import builtins
import os
import json, hashlib, h5py
from numba import njit
from .functions_library_universal import cauchy_sum, maybe_njit, npfloat

@maybe_njit
def lorentz_force_hyperB(t, y, gamma, qoverm):
    y = y.astype(npfloat)
    gamma = npfloat(gamma)

    Bz = np.tanh(gamma * y[1])
    ax = qoverm * y[4] * Bz
    ay = - qoverm * y[3] * Bz
    az = npfloat(0.0)

    return np.array([y[3], y[4], y[5], ax, ay, az], dtype=npfloat)

@maybe_njit
def PS_hyperB(PS_order, steps_ps, initial_pos_vel, timedelta, gamma, qoverm, tol):
    n_total = 9        # x, y, z, v_x, v_y, v_z, sinh, cosh, Bz 
    final_coeff_matrix = np.zeros((n_total, steps_ps + 1), dtype=npfloat)          # array to store everything
    
    #Labeling indices for sanity tracking
    x, y, z = 0, 1, 2
    vx, vy, vz = 3, 4, 5
    sinh_aux, cosh_aux = 6, 7
    Bz_aux  = 8

    # Seeting up Initial conditions
    final_coeff_matrix[0:6, 0] = initial_pos_vel
    y0 = initial_pos_vel[1]
    
    final_coeff_matrix[sinh_aux, 0] = np.sinh(gamma * y0)         # initiated analytically for start
    final_coeff_matrix[cosh_aux, 0] = np.cosh(gamma * y0) 
    final_coeff_matrix[Bz_aux, 0] = np.tanh(gamma * y0)

    Bz_series = np.zeros(PS_order, dtype=npfloat)
    orders_used = np.zeros(steps_ps + 1, dtype=np.int32)
    oip1 = 1.0 / (1.0 + np.arange(PS_order))

    for j in range(1, steps_ps + 1):              # indexing over time steps, start @1 b/c 0 is determined above
        c = np.zeros((n_total, PS_order + 1), dtype=npfloat)    # array storage for loop
        c[:, 0] = final_coeff_matrix[:, j - 1]   # applies previous step calcs as initial start in temp array

        sum_terms = np.zeros(n_total, dtype=npfloat)
        power = timedelta
        max_contrib = tol + npfloat(1.0)
        i = 0

        while max_contrib > tol and i < PS_order:
            # Core expansions occur in this loop 
            # First, update Bz[i] using the division recurrence: Bz = sinh / cosh= (1/B_0)(A_n-\sum_k=1^n B_k f_(n-k))
            if i == 0: # caucy division recurrence is definited for n>0, 0 called from initial conditions
                Bz_series[0] = c[sinh_aux, 0] / c[cosh_aux, 0] # this is just using the analytical values to start
            else:
                s = c[sinh_aux, i]
                for k in range(1, i + 1):
                    s -= c[cosh_aux, k] * Bz_series[i - k]
                Bz_series[i] = s / c[cosh_aux, 0]
            
            # cauchy product of v_yB_z and v_xB_z, must stay abouve the velocity calculations
            vyBz = cauchy_sum(c[vy], Bz_series, i)
            vxBz = cauchy_sum(c[vx], Bz_series, i)

            c[x, i+1] = oip1[i] * c[vx, i] 
            c[y, i+1] = oip1[i] * c[vy, i]
            c[z, i+1] = oip1[i] * c[vz, i]
            c[vx, i+1] = qoverm * oip1[i] * vyBz  
            c[vy, i+1] = - qoverm * oip1[i] * vxBz
            c[vz, i+1] = 0.0
            
            # Aux calculations, analytical initial values but the remaining summations are pure expansions
            c[sinh_aux, i+1] =  oip1[i] * gamma * cauchy_sum(c[cosh_aux], c[vy], i)
            c[cosh_aux, i+1] =  oip1[i] * gamma * cauchy_sum(c[sinh_aux], c[vy], i)

            # Bz is already computed in Bz_series[i], this is just adding it to the temp matrix (not Neo's)
            c[Bz_aux, i+1] = Bz_series[i]

            sum_terms += c[:, i+1]*power # just keeps adding these on until PS prder is reached, final added to permanent matrix below
            max_contrib = np.abs(c[:, i+1]).max()
            power *= timedelta
            i += 1
            
        final_coeff_matrix[:, j] = final_coeff_matrix[:, j - 1] + sum_terms        

        # tethering
        y_now = final_coeff_matrix[y, j]
        sinh_now = np.sinh(gamma * y_now)
        cosh_now = np.cosh(gamma * y_now)
        Bz_now = sinh_now / cosh_now

        final_coeff_matrix[sinh_aux, j] = sinh_now
        final_coeff_matrix[cosh_aux, j] = cosh_now
        final_coeff_matrix[Bz_aux, j] = Bz_now


        orders_used[j] = i

    return final_coeff_matrix, orders_used

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
                   mass_si, q_e, B_0, delta,
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
        "delta": _to_serializable(delta),

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

        for k in ("ps","rk4","rk45","rkg"):
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

        for k in ("ps","rk4","rk45","rkg"):
            loaded[k] = _read_grp(k)

        # meta attrs
        gmeta = f["meta"]
        for a in gmeta.attrs:
            if a.startswith("timing_"):
                loaded["meta"]["timing"][a.replace("timing_","")] = gmeta.attrs[a]
            else:
                loaded["meta"][a] = gmeta.attrs[a]

        return loaded
