"""
Print the group/dataset tree of an h5 file with shapes, dtypes and attrs.

Usage:
    python scripts/inspect_hdf5.py data/dipoleb/<run>/_rawdata/<hash>_full.h5

Read-only. Virtual datasets are tagged with how many backing _segNNN.h5
sources resolve (same search order as HDF5: $HDF5_VDS_PREFIX, the file's
folder, cwd); an unresolved VDS reads as zeros with no error, which this
flags.
"""
import h5py
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from ps_method import writers as wr


def _vds_tag(dset, path):
    """Tag a virtual dataset with its resolvable-source count (writers decides)."""
    if not dset.is_virtual:
        return ""
    total = len(dset.virtual_sources())
    group, _, name = dset.name.strip("/").rpartition("/")
    missing = len(wr.missing_vds_sources(path, group=group, dset=name))
    present = total - missing
    warn = "" if missing == 0 else "  ⚠ MISSING SOURCES → reads as zeros"
    return f"  [VIRTUAL: {present}/{total} sources present{warn}]"


def inspect_hdf5(path):
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
                      f"dtype={obj.dtype}{_vds_tag(obj, path)}")

        f.visititems(walk)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/inspect_hdf5.py path/to/file.h5")
        sys.exit(2)
    inspect_hdf5(sys.argv[1])
