"""
Logging and debug diagnostics for dipole trajectory runs.

    setup_logger       — create a file-backed logger
    redirect_logger    — move logger output to a new path
    check_time_grids   — validate step sizes and build time arrays
"""

import os
import logging
import shutil


def setup_logger(name="dipole_logger", filename="dipole_run.log", level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)

    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    formatter = logging.Formatter('%(levelname)s — %(message)s')

    file_handler = logging.FileHandler(filename, mode="w")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def redirect_logger(logger, new_path):
    """Move a logger's file output to a new path.

    Copies any content already written to the original log file, then
    replaces the file handler so subsequent messages go to new_path.
    """
    old_path = None
    formatter = None
    for h in logger.handlers:
        if isinstance(h, logging.FileHandler):
            old_path = h.baseFilename
            formatter = h.formatter
            break

    if old_path is None:
        return

    # Flush and close the old handler
    for h in logger.handlers[:]:
        if isinstance(h, logging.FileHandler):
            h.flush()
            h.close()
            logger.removeHandler(h)

    # Copy early log content to the new location
    if os.path.exists(old_path) and old_path != os.path.abspath(new_path):
        shutil.copy2(old_path, new_path)
        os.remove(old_path)

    # Attach new handler (append so copied content is preserved)
    new_handler = logging.FileHandler(new_path, mode="a")
    if formatter:
        new_handler.setFormatter(formatter)
    logger.addHandler(new_handler)


def check_time_grids(norm_time, ps_step=None, steps_ps=None,
                     rk4_step=None, steps_rk4=None,
                     rkg_step=None, steps_rkg=None,
                     rk45_t=None, rtol=1e-12):
    """Report each enabled method's time grid and flag drift from norm_time.

    For each enabled method the function reports step / steps / final_time
    alongside norm_time, and tags the line ``[OK]`` if the method's final
    time matches norm_time within ``rtol * |norm_time|``, or
    ``[DRIFT: Δ=...]`` with the signed difference if it doesn't.

    PS always matches by construction (the driver forces
    ``norm_time = steps_ps * ps_step``). RK4 / RKG with a step size that
    doesn't divide norm_time evenly will land slightly short or long —
    that's the kind of drift this check is meant to surface.
    """
    lines = []
    threshold = rtol * abs(norm_time)

    def _flag(final_t):
        diff = float(final_t) - float(norm_time)
        if abs(diff) <= threshold:
            return "[OK]"
        return f"[DRIFT: Δ={diff:+.3e}]"

    def _report(label, step, steps):
        final_t = step * steps
        lines.append(
            f"{label}: step={step:.3e}, steps={steps}, "
            f"final_time={final_t:.3e}, norm_time={norm_time:.3e} {_flag(final_t)}"
        )

    if ps_step is not None and steps_ps is not None:
        _report("PS", ps_step, steps_ps)
    if rk4_step is not None and steps_rk4 is not None:
        _report("RK4", rk4_step, steps_rk4)
    if rkg_step is not None and steps_rkg is not None:
        _report("RKG", rkg_step, steps_rkg)
    if rk45_t is not None:
        final_t = rk45_t[-1]
        lines.append(
            f"RK45: final_time={final_t:.3e}, "
            f"norm_time={norm_time:.3e} {_flag(final_t)}"
        )

    return "\n".join(lines)
