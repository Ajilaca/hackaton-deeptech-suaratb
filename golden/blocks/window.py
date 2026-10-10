"""Hamming 400 titik (Q1.15), zero-pad ke 512, keluaran W bit (pecahan tambahan G = W-16 bit).

windowed = round(y * w / 2^(15-G)): hasil kali Q1.15 dipertahankan G bit pecahan lebih
banyak daripada sampel int16, jadi lebar FFT > 16 benar-benar menambah presisi.
Padanan RTL: window.sv (ROM 200 entri, simetris).
"""
from __future__ import annotations
import numpy as np
from .. import params
from .fixed import rshift_round, sat


def hamming_q15(n: int | None = None, frac: int | None = None) -> np.ndarray:
    n = params.FRAME_LEN if n is None else n
    frac = params.WIN_FRAC if frac is None else frac
    k = np.arange(n)
    w = 0.54 - 0.46 * np.cos(2 * np.pi * k / (n - 1))
    return np.clip(np.floor(w * (1 << frac) + 0.5), 0, (1 << frac) - 1).astype(np.int64)


def apply_window(frames, W: int | None = None, in_frac: int = 0) -> np.ndarray:
    """frames: (..., 400) dengan in_frac bit pecahan (0 = int16) -> (..., 512) int64 (W bit bertanda)."""
    W = params.FFT_W if W is None else W
    g = W - 16 - in_frac
    assert 0 <= g <= params.WIN_FRAC - 1
    f = np.asarray(frames, dtype=np.int64)
    w = hamming_q15()
    y = rshift_round(f * w, params.WIN_FRAC - g)
    y = sat(y, W)
    pad = np.zeros(f.shape[:-1] + (params.NFFT - params.FRAME_LEN,), dtype=np.int64)
    return np.concatenate([y, pad], axis=-1)
