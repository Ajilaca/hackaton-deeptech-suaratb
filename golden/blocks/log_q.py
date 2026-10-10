"""log2 fixed-point: leading-one + 8 bit mantissa -> LUT 256 entri. Keluaran Q5.8 (0..8191).

L = (p << 8) + LUT[m], p = posisi leading-one (0..31), m = 8 bit tepat di bawahnya.
E = 0 -> L = 0. LUT[m] = min(255, round(256*log2(1 + (m+0.5)/256))) (titik tengah mengurangi bias).
Padanan RTL: log_q.sv (priority encoder 32 bit + barrel shift + ROM 256x8).
"""
from __future__ import annotations
import numpy as np
from .. import params
from .fixed import bit_length


def log_lut() -> np.ndarray:
    m = np.arange(1 << params.LOG_LUT_BITS)
    v = np.floor(256.0 * np.log2(1.0 + (m + 0.5) / (1 << params.LOG_LUT_BITS)) + 0.5)
    return np.minimum(v, 255).astype(np.int64)


def log2_q(e) -> np.ndarray:
    e = np.asarray(e, dtype=np.int64)
    p = np.maximum(bit_length(e) - 1, 0)
    mb = params.LOG_LUT_BITS
    left = np.where(p >= mb, p - mb, 0)
    right = np.where(p < mb, mb - p, 0)
    m = ((e >> left) << right) & ((1 << mb) - 1)
    out = (p << params.LOG_FRAC) + log_lut()[m]
    return np.where(e > 0, out, 0)
