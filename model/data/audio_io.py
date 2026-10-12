"""Muat WAV -> mono int16 16 kHz (resample_poly), skala asli dipertahankan (tanpa normalisasi)."""
from __future__ import annotations
from math import gcd
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


def load_16k(path: str) -> np.ndarray:
    x, fs = sf.read(path, dtype="float64", always_2d=True)
    x = x.mean(axis=1)
    if fs != 16000:
        g = gcd(int(fs), 16000)
        x = resample_poly(x, 16000 // g, int(fs) // g)
    return np.clip(np.rint(x * 32768.0), -32768, 32767).astype(np.int16)
