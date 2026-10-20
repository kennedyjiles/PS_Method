#!/usr/bin/env python
"""Compare ps/y and ps/orders of two h5 runs (e.g. single-file vs stitched VDS).

Usage:
    python scripts/compare_runs.py <a.h5> <b.h5>

Streams column blocks so it never loads the full dataset. Prints the max
absolute difference for information; the verdict and exit code (0 identical,
1 differences or shape mismatch, 2 usage) are strict bit-identity. VDS
sources are checked first so a dangling stitch is reported rather than
compared as zeros.
"""
import os, sys, numpy as np, h5py
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from ps_method import writers as wr

def main(a, b, block=200_000):
    for path in (a, b):
        wr.check_vds_readable(path)
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
            d = float(np.max(np.abs(ca - cb)))
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
