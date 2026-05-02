"""
Quick benchmark: compare h5 compression strategies on an existing file.

Usage:
    python scripts/benchmark_compression.py path/to/your_file.h5

Reads all datasets from the file, recompresses each with different
settings, and reports file size and write time.
"""

import sys
import os
import time
import tempfile
import h5py
import numpy as np


def collect_datasets(group, prefix=""):
    """Recursively collect all datasets from an h5 group."""
    datasets = {}
    for key in group:
        path = f"{prefix}/{key}" if prefix else key
        if isinstance(group[key], h5py.Dataset):
            datasets[path] = group[key][:]
        elif isinstance(group[key], h5py.Group):
            datasets.update(collect_datasets(group[key], path))
    return datasets


def benchmark_write(datasets, compression, shuffle, label):
    """Write all datasets to a temp file with given settings and measure."""
    tmp = tempfile.NamedTemporaryFile(suffix=".h5", delete=False)
    tmp.close()

    t0 = time.perf_counter()
    with h5py.File(tmp.name, "w") as f:
        for name, data in datasets.items():
            f.create_dataset(
                name, data=data,
                compression=compression,
                compression_opts=compression if isinstance(compression, int) else None,
                shuffle=shuffle,
            )
    elapsed = time.perf_counter() - t0
    size_mb = os.path.getsize(tmp.name) / (1024 * 1024)

    # Also measure read time
    t0 = time.perf_counter()
    with h5py.File(tmp.name, "r") as f:
        for name in datasets:
            _ = f[name][:]
    read_time = time.perf_counter() - t0

    os.unlink(tmp.name)
    return size_mb, elapsed, read_time


def main():
    if len(sys.argv) < 2:
        print("Usage: python benchmark_compression.py <path_to_h5_file>")
        sys.exit(1)

    src_path = sys.argv[1]
    src_size_mb = os.path.getsize(src_path) / (1024 * 1024)

    print(f"Source file: {src_path}")
    print(f"Source size: {src_size_mb:.2f} MB\n")

    with h5py.File(src_path, "r") as f:
        datasets = collect_datasets(f)

    total_elements = sum(d.size for d in datasets.values())
    raw_mb = sum(d.nbytes for d in datasets.values()) / (1024 * 1024)
    print(f"Datasets: {len(datasets)}, Total elements: {total_elements:,}")
    print(f"Raw (uncompressed) size: {raw_mb:.2f} MB\n")

    configs = [
        ("none",              None,  False),
        ("gzip=1",            1,     False),
        ("gzip=2 (current)",  2,     False),
        ("gzip=4",            4,     False),
        ("gzip=9",            9,     False),
        ("gzip=1 + shuffle",  1,     True),
        ("gzip=2 + shuffle",  2,     True),
        ("gzip=4 + shuffle",  4,     True),
        ("gzip=9 + shuffle",  9,     True),
    ]

    print(f"{'Config':<22s} {'Size MB':>8s} {'Write s':>8s} {'Read s':>8s} {'vs current':>10s}")
    print("-" * 60)

    baseline_size = None
    for label, comp, shuf in configs:
        comp_arg = "gzip" if comp is not None else None
        size, wtime, rtime = benchmark_write(datasets, comp_arg, shuf, label)
        if "current" in label:
            baseline_size = size

        vs = ""
        if baseline_size and baseline_size > 0:
            pct = ((size - baseline_size) / baseline_size) * 100
            vs = f"{pct:+.1f}%"

        # Fix compression_opts passing
        print(f"{label:<22s} {size:>8.2f} {wtime:>8.3f} {rtime:>8.3f} {vs:>10s}")


def benchmark_write(datasets, compression, shuffle, label):
    """Write all datasets to a temp file with given settings and measure."""
    tmp = tempfile.NamedTemporaryFile(suffix=".h5", delete=False)
    tmp.close()

    # Determine gzip level from label
    comp_opts = None
    if compression == "gzip":
        for part in label.split("="):
            for word in part.split():
                try:
                    comp_opts = int(word)
                    break
                except ValueError:
                    continue
            if comp_opts is not None:
                break

    t0 = time.perf_counter()
    with h5py.File(tmp.name, "w") as f:
        for name, data in datasets.items():
            kwargs = {"shuffle": shuffle}
            if compression is not None:
                kwargs["compression"] = compression
                kwargs["compression_opts"] = comp_opts
            f.create_dataset(name, data=data, **kwargs)
    elapsed = time.perf_counter() - t0
    size_mb = os.path.getsize(tmp.name) / (1024 * 1024)

    t0 = time.perf_counter()
    with h5py.File(tmp.name, "r") as f:
        for name in datasets:
            _ = f[name][:]
    read_time = time.perf_counter() - t0

    os.unlink(tmp.name)
    return size_mb, elapsed, read_time


if __name__ == "__main__":
    main()
