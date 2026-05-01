"""
run.py — Unified entry point for all PS Method drivers.

Infers the field type (constB, hyperB, dipoleB) from the config path
and dispatches to the appropriate driver's main() function.

Full run (solvers + post-processing):
    python run.py configs/constB/demo.yml
    python run.py configs/hyper/demo.yml
    python run.py configs/dipole/demo.yml

Post-processing only (replot from cached h5 data):
    python run.py data/constB/demo/demo.yml
    python run.py data/hyperB/demo/demo.yml
    python run.py data/dipoleB/demo/demo.yml

Shorthand (field type + config name):
    python run.py constB demo
    python run.py hyperB demo
    python run.py dipoleB paper1
"""

import os
import sys


# Map field keywords → (config subdirectory, data subdirectory)
_DRIVERS = {
    "constb":  ("constB",  "constB"),
    "hyperb":  ("hyper",   "hyperB"),
    "dipoleb": ("dipole",  "dipoleB"),
}

# Map directory names → field keyword (for path-based inference)
_DIR_TO_FIELD = {
    # configs/ subdirectories
    "constB":  "constb",
    "constb":  "constb",
    "hyper":   "hyperb",
    "dipole":  "dipoleb",
    # data/ subdirectories
    "dipoleB": "dipoleb",
    "hyperB":  "hyperb",
}


def _resolve(args):
    """Return (field_key, yaml_path, replot) from the command-line arguments.

    Supports these calling conventions:
        run.py configs/dipole/demo.yml          → full run
        run.py data/dipoleB/demo/demo.yml       → replot only
        run.py dipoleB demo                     → full run (shorthand)
        run.py dipoleB configs/dipole/demo.yml  → full run (explicit)
    """
    if len(args) == 1:
        path = args[0]

        # --- Detect replot mode: path starts with data/ ---
        if path.startswith("data/") or path.startswith("data" + os.sep):
            # Infer field type from data/<field>/ prefix
            parts = path.replace(os.sep, "/").split("/")
            # parts: ["data", "<field>", "<config>", "config.yml"]
            if len(parts) >= 3:
                data_dir = parts[1]
                if data_dir.lower() in _DIR_TO_FIELD:
                    field_key = _DIR_TO_FIELD[data_dir]
                elif data_dir.lower() in [k for k in _DRIVERS]:
                    field_key = data_dir.lower()
                else:
                    raise ValueError(
                        f"Cannot infer field type from data path: {path}\n"
                        f"Expected data/<field>/... where field is constB, hyperB, or dipoleB"
                    )
                if not os.path.isfile(path):
                    raise FileNotFoundError(f"Config file not found: {path}")
                return field_key, path, True  # replot = True

        # --- Normal run: path contains configs/<subdir>/ ---
        for dirname, field_key in _DIR_TO_FIELD.items():
            if f"configs/{dirname}/" in path or f"configs{os.sep}{dirname}{os.sep}" in path:
                if not os.path.isfile(path):
                    raise FileNotFoundError(f"Config file not found: {path}")
                return field_key, path, False  # replot = False

        raise ValueError(
            f"Cannot infer field type from path: {path}\n"
            f"Expected path under configs/ (full run) or data/ (replot)"
        )

    elif len(args) == 2:
        field_raw, run_name = args
        field_key = field_raw.lower().rstrip("/")
        if field_key not in _DRIVERS:
            raise ValueError(
                f"Unknown field type: '{field_raw}'\n"
                f"Expected one of: constB, hyperB, dipoleB"
            )
        config_subdir = _DRIVERS[field_key][0]
        configs_dir = os.path.join(os.path.dirname(__file__), "configs", config_subdir)

        # If run_name is already a full path
        if run_name.endswith((".yml", ".yaml")) and os.path.isfile(run_name):
            replot = "data/" in run_name or f"data{os.sep}" in run_name
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
        return field_key, yaml_path, False

    else:
        raise SystemExit(
            "Usage:\n"
            "  python run.py configs/dipole/demo.yml       # full run\n"
            "  python run.py data/dipoleB/demo/demo.yml    # replot only\n"
            "  python run.py dipoleB demo                  # shorthand"
        )


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    field_key, yaml_path, replot = _resolve(sys.argv[1:])

    driver_label = _DRIVERS[field_key][1]
    mode = "replot (post-processing only)" if replot else "full run"

    print(f"Field type : {driver_label}")
    print(f"Config     : {yaml_path}")
    print(f"Mode       : {mode}\n")

    if field_key == "constb":
        from constB import main as driver_main
    elif field_key == "hyperb":
        from hyperB import main as driver_main
    elif field_key == "dipoleb":
        from dipoleB import main as driver_main

    driver_main(yaml_path, replot=replot)


if __name__ == "__main__":
    main()
