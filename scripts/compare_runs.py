#!/usr/bin/env python
"""Compare two dipoleb PS h5 files (single-file vs segmented VDS).

Usage:
    python compare_runs.py  <baseline_full.h5>  <segmented_full.h5>

Reads ps/y and ps/orders in column blocks (so it never loads a multi-TB
dataset into RAM) and reports the max absolute difference plus whether the
two runs are bit-identical.
"""
import sys, numpy as np, h5py

def main(a, b, block=200_000):
    with h5py.File(a, "r") as fa, h5py.File(b, "r") as fb:
        ya, yb = fa["ps/y"], fb["ps/y"]
        oa, ob = fa["ps/orders"], fb["ps/orders"]
        print(f"baseline : y{ya.shape} orders{oa.shape}  virtual={ya.is_virtual}")
        print(f"segmented: y{yb.shape} orders{ob.shape}  virtual={yb.is_virtual}")
        if ya.shape != yb.shape or oa.shape != ob.shape:
            print("SHAPE MISMATCH — different number of saved points"); return 1
        n = ya.shape[1]
        max_abs = 0.0
        bit_ident = True
        orders_ident = True
        for j0 in range(0, n, block):
            j1 = min(j0 + block, n)
            ca, cb = ya[:, j0:j1], yb[:, j0:j1]
            d = float(np.max(np.abs(ca - cb))) if j1 > j0 else 0.0
            max_abs = max(max_abs, d)
            bit_ident &= np.array_equal(ca, cb)
            orders_ident &= np.array_equal(oa[j0:j1], ob[j0:j1])
        print(f"columns compared        : {n:,}")
        print(f"max |y_base - y_seg|    : {max_abs:.3e}")
        print(f"orders bit-identical    : {orders_ident}")
        print(f"y bit-identical         : {bit_ident}")
        print("\nRESULT:", "IDENTICAL ✓" if (bit_ident and orders_ident)
              else "DIFFERENCES FOUND ✗")
        return 0 if (bit_ident and orders_ident) else 1

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__); sys.exit(2)
    sys.exit(main(sys.argv[1], sys.argv[2]))
