"""
GPU/CPU backend abstraction.

All numeric code should import from here rather than directly importing cupy.
When GPU is unavailable (tests, CPU machines), falls back transparently to numpy.

Usage:
    from src.utils.backend import xp, GPU_AVAILABLE, to_numpy, from_numpy
"""
from __future__ import annotations

import os
import numpy as np

# Allow forcing CPU via environment variable (useful for tests and CI)
_FORCE_CPU = os.getenv("TOPARB_FORCE_CPU", "0").lower() in ("1", "true", "yes")

try:
    if _FORCE_CPU:
        raise ImportError("CPU mode forced by TOPARB_FORCE_CPU")
    import cupy as xp  # type: ignore[import]
    GPU_AVAILABLE = True
except ImportError:
    xp = np  # type: ignore[assignment]
    GPU_AVAILABLE = False


def to_numpy(arr) -> np.ndarray:
    """Convert cupy or numpy array to numpy."""
    if GPU_AVAILABLE and hasattr(arr, "get"):
        return arr.get()
    return np.asarray(arr)


def from_numpy(arr: np.ndarray):
    """Convert numpy array to backend array (cupy or numpy)."""
    if GPU_AVAILABLE:
        return xp.asarray(arr)
    return arr


def asnumpy(arr) -> np.ndarray:
    """Alias for to_numpy for cupy compatibility."""
    return to_numpy(arr)