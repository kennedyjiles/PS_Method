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
import datetime
import h5py


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


def main():
    if len(sys.argv) < 2:
        print("Usage: python benchmark_compression.py <path_to_h5_file>")
        sys.exit(1)

    src_path = sys.argv[1]
    src_size_mb = os.path.getsize(src_path) / (1024 * 1024)

    # --- Open a tee-style log next to the source h5 so the choice of
    #     compression setting is auditable later. Reruns overwrite. ---
    src_dir = os.path.dirname(os.path.abspath(src_path)) or "."
    src_stem = os.path.splitext(os.path.basename(src_path))[0]
    log_path = os.path.join(src_dir, f"{src_stem}_compression_benchmark.log")
    log_file = open(log_path, "w")

    def log_print(msg=""):
        print(msg)
        log_file.write(msg + "\n")
        log_file.flush()

    log_print(f"# Compression benchmark — {datetime.datetime.now().isoformat(timespec='seconds')}")
    log_print(f"# Command: {' '.join(sys.argv)}")
    log_print("")
    log_print(f"Source file: {src_path}")
    log_print(f"Source size: {src_size_mb:.2f} MB")
    log_print("")

    with h5py.File(src_path, "r") as f:
        datasets = collect_datasets(f)

    total_elements = sum(d.size for d in datasets.values())
    raw_mb = sum(d.nbytes for d in datasets.values()) / (1024 * 1024)
    log_print(f"Datasets: {len(datasets)}, Total elements: {total_elements:,}")
    log_print(f"Raw (uncompressed) size: {raw_mb:.2f} MB")
    log_print("")

    configs = [
        ("none",              None,  False),
        ("gzip=1",                      1,     False),
        ("gzip=2",                      2,     False),
        ("gzip=4",                      4,     False),
        ("gzip=9",                      9,     False),
        ("gzip=1 + shuffle",   1,     True),
        ("gzip=2 + shuffle",   2,     True),
        ("gzip=4 + shuffle",   4,     True),
        ("gzip=9 + shuffle",   9,     True),
    ]

    log_print(f"{'Config':<20s} {'Size MB':>8s} {'size %':>8s} {'Write s':>8s} {'Read s':>8s}")
    log_print(f"# size % is vs the 'none' (uncompressed) baseline")
    log_print("-" * 57)

    baseline_size = None
    for label, comp, shuf in configs:
        comp_arg = "gzip" if comp is not None else None
        size, wtime, rtime = benchmark_write(datasets, comp_arg, shuf, label)
        if label == "none":
            baseline_size = size

        size_vs = ""
        if baseline_size and baseline_size > 0:
            size_vs = f"{((size - baseline_size) / baseline_size) * 100:+.1f}%"

        log_print(f"{label:<20s} {size:>8.2f} {size_vs:>8s} {wtime:>8.3f} {rtime:>8.3f}")

    log_print("")
    log_print(f"# Log saved to: {log_path}")
    log_file.close()
    print(f"\nLog saved to: {log_path}")


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
