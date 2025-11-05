# log.py
import logging
import sys

def setup_logger():
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True  # ensures all existing configs are reset
    )

def get_logger(name: str):
    return logging.getLogger(name)
