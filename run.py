"""
run.py — single entry point for the constb / hyperb / dipoleb drivers.

Usage:
    python run.py configs/<field>/<name>.yml [--replot]   # full run (or replot)
    python run.py <field> <name> [--replot]               # shorthand for the above
    python run.py <any path containing data/<field>/>.yml # replot from a saved run

<field> is one of constb, hyperb, dipoleb and is inferred from the path. A yml
anywhere under a data/<field>/ folder (the provenance copy, or a trim_h5
companion, relative or absolute) always means replot; --replot forces replot
for a configs/ path. The driver is called as main(yaml_path, replot=...); no
other flags are passed through.
"""

import os
import re
import sys


# Field keywords used both as the driver name and as the configs/ + data/
# subdirectory name (they're identical after the lowercase rename).
_FIELDS = {"constb", "hyperb", "dipoleb"}


def _resolve(args):
    """Return (field_key, yaml_path, replot) from the command-line arguments.

    Supports these calling conventions:
        run.py configs/dipoleb/demo.yml           → full run
        run.py configs/dipoleb/demo.yml --replot  → replot only
        run.py data/dipoleb/demo/demo.yml         → replot only (auto-detected)
        run.py dipoleb demo                       → full run (shorthand)
        run.py dipoleb configs/dipoleb/demo.yml   → full run (explicit)
    """
    # Strip --replot flag from args before positional parsing
    force_replot = "--replot" in args
    args = [a for a in args if a != "--replot"]

    if len(args) == 1:
        path = args[0]

        # --- Detect replot mode: a data/<field>/ folder anywhere in the path
        #     (relative, ./, ../, absolute, or under an output_root prefix) ---
        m = re.search(r"(?:^|/)data/([^/]+)/", path.replace(os.sep, "/"))
        if m:
            data_dir = m.group(1).lower()
            if data_dir not in _FIELDS:
                raise ValueError(
                    f"Cannot infer field type from data path: {path}\n"
                    f"Expected data/<field>/... where field is one of {sorted(_FIELDS)}"
                )
            if not os.path.isfile(path):
                raise FileNotFoundError(f"Config file not found: {path}")
            return data_dir, path, True  # replot = True

        # --- Normal run: path contains configs/<field>/ ---
        for field_key in _FIELDS:
            if f"configs/{field_key}/" in path or f"configs{os.sep}{field_key}{os.sep}" in path:
                if not os.path.isfile(path):
                    raise FileNotFoundError(f"Config file not found: {path}")
                return field_key, path, force_replot

        raise ValueError(
            f"Cannot infer field type from path: {path}\n"
            f"Expected path under configs/ (full run) or data/ (replot)"
        )

    elif len(args) == 2:
        field_raw, run_name = args
        field_key = field_raw.lower().rstrip("/")
        if field_key not in _FIELDS:
            raise ValueError(
                f"Unknown field type: '{field_raw}'\n"
                f"Expected one of: {sorted(_FIELDS)}"
            )
        configs_dir = os.path.join(os.path.dirname(__file__), "configs", field_key)

        # If run_name is already a full path
        if run_name.endswith((".yml", ".yaml")) and os.path.isfile(run_name):
            replot = force_replot or "data/" in run_name or f"data{os.sep}" in run_name
            return field_key, run_name, replot

        # Otherwise treat it as a config name
        yaml_path = os.path.join(configs_dir, f"{run_name}.yml")
        if not os.path.isfile(yaml_path):
            available = [f.replace(".yml", "") for f in os.listdir(configs_dir)
                         if f.endswith(".yml") and f != "base.yml"]
            raise FileNotFoundError(
                f"No config found: {yaml_path}\n"
                f"Available configs for {field_raw}: {available}"
            )
        return field_key, yaml_path, force_replot

    else:
        raise SystemExit(
            "Usage:\n"
            "  python run.py configs/dipoleb/demo.yml      # full run\n"
            "  python run.py data/dipoleb/demo/demo.yml    # replot only\n"
            "  python run.py dipoleb demo                  # shorthand"
        )


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    field_key, yaml_path, replot = _resolve(sys.argv[1:])

    driver_label = field_key
    mode = "replot (post-processing only)" if replot else "full run"

    print(f"Field type : {driver_label}")
    print(f"Config     : {yaml_path}")
    print(f"Mode       : {mode}\n")

    if field_key == "constb":
        from constb import main as driver_main
    elif field_key == "hyperb":
        from hyperb import main as driver_main
    elif field_key == "dipoleb":
        from dipoleb import main as driver_main

    driver_main(yaml_path, replot=replot)


if __name__ == "__main__":
    main()
