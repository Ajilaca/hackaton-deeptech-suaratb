"""Pre-emphasis y[n] = x[n] - a*x[n-1], a = PREEMPH_Q15/2^15.

Masukan int16, keluaran 16+frac bit (saturasi). frac>0 menghindari pembulatan ke int16 (K-08). `prev` = sampel terakhir blok sebelumnya (0 saat reset).
Padanan RTL: preemph.sv (Fase 2). Latensi RTL target: 1 sampel (2 siklus bila pengali berstage).
"""
from __future__ import annotations
import numpy as np
from .. import params
from .fixed import rshift_round, sat


def preemph(x, coef_q15: int | None = None, prev: int = 0, frac: int = 0) -> np.ndarray:
    """frac = jumlah bit pecahan tambahan pada keluaran (0 -> int16; params.PREEMPH_FRAC -> 16+frac bit).
    y = (x << frac) - round(a * x_prev / 2^(15-frac)); saturasi ke 16+frac bit."""
    a = params.PREEMPH_Q15 if coef_q15 is None else coef_q15
    assert 0 <= frac <= 15
    x = np.asarray(x, dtype=np.int64)
    xp = np.concatenate(([prev], x[:-1]))
    y = (x << frac) - rshift_round(a * xp, 15 - frac)
    return sat(y, 16 + frac)
