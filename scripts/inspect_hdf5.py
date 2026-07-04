"""
Quick HDF5 inspector — prints the group/dataset tree with shapes, dtypes,
and attributes for a run h5 file. Read-only.

    inspect_hdf5 — open a file and recursively print its structure
"""
import h5py
import os
import sys


# to use: python scripts/inspect_hdf5.py data/<config-name>/_rawdata/run_xxxxxxxx.h5

def _vds_tag(dset, folder):
    """For a Virtual Dataset, report how many backing source files are present.

    Sources are stored as basenames (folder-relative), so a stitched _full.h5
    reads its _segNNN.h5 files from the same directory. A VDS silently returns
    fill-value (0) for absent sources, so a missing count flags likely-corrupt
    (zero-filled) reads — something the shape/dtype alone can't reveal.
    """
    if not dset.is_virtual:
        return ""
    srcs = dset.virtual_sources()
    present = 0
    for vs in srcs:
        fn = vs.file_name
        if (os.path.isabs(fn) and os.path.exists(fn)) or \
           os.path.exists(os.path.join(folder, os.path.basename(fn))):
            present += 1
    total = len(srcs)
    warn = "" if present == total else "  ⚠ MISSING SOURCES → reads as zeros"
    return f"  [VIRTUAL: {present}/{total} sources present{warn}]"


def inspect_hdf5(path):
    folder = os.path.dirname(os.path.abspath(path))
    with h5py.File(path, "r") as f:
        print(f"\n📂 File: {path}")
        print("Root attributes:")
        for k, v in f.attrs.items():
            print(f"   {k}: {v}")

        def walk(name, obj):
            if isinstance(obj, h5py.Group):
                print(f"\n📁 Group: {name}")
                for k, v in obj.attrs.items():
                    print(f"   (attr) {k}: {v}")
            elif isinstance(obj, h5py.Dataset):
                print(f"  📄 Dataset: {name}  shape={obj.shape} "
                      f"dtype={obj.dtype}{_vds_tag(obj, folder)}")

        f.visititems(walk)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inspect_hdf5.py path/to/file.h5")
    else:
        inspect_hdf5(sys.argv[1])
