import logging
import os

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
