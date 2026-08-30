"""
utils.py
Shared helpers: logging, reproducibility, and simple timing.
"""

import logging
import random
import sys
import time
from contextlib import contextmanager

import numpy as np


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)


def get_logger(name: str = "rag_router") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            fmt="[%(asctime)s] %(levelname)s - %(name)s - %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


@contextmanager
def timer():
    """Usage: with timer() as t: ... ; print(t.elapsed_ms)"""
    class _T:
        elapsed_ms = 0.0
    t = _T()
    start = time.perf_counter()
    yield t
    t.elapsed_ms = (time.perf_counter() - start) * 1000
