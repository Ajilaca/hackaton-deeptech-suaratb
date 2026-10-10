"""Spektrum daya P[k] = re^2 + im^2, k = 0..256 (tanpa penskalaan; lebar 2W-1 bit tak bertanda).
Padanan RTL: power.sv (2 pengali atau 1 pengali berbagi waktu)."""
from __future__ import annotations
import numpy as np
from .. import params


def power(re, im) -> np.ndarray:
    re = np.asarray(re, dtype=np.int64)[..., : params.N_BINS]
    im = np.asarray(im, dtype=np.int64)[..., : params.N_BINS]
    return re * re + im * im
