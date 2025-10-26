import h5py
import sys


# to use python inspect_hdf5.py dipole_outputs/run_xxxxxxxx.h5

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
                print(f"  📄 Dataset: {name}  shape={obj.shape} dtype={obj.dtype}")

        f.visititems(walk)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inspect_hdf5.py path/to/file.h5")
    else:
        inspect_hdf5(sys.argv[1])
