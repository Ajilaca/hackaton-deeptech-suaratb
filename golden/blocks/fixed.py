"""Aritmetika bilangan bulat bersama (semantik = perangkat keras: tambah setengah lalu geser aritmetik)."""
from __future__ import annotations
import numpy as np


def rshift_round(v, sh: int):
    """floor((v + 2^(sh-1)) / 2^sh) pada int64; sh=0 -> v."""
    v = np.asarray(v, dtype=np.int64)
    if sh <= 0:
        return v << (-sh)
    return (v + (1 << (sh - 1))) >> sh


def sat(v, bits: int, signed: bool = True):
    v = np.asarray(v, dtype=np.int64)
    if signed:
        return np.clip(v, -(1 << (bits - 1)), (1 << (bits - 1)) - 1)
    return np.clip(v, 0, (1 << bits) - 1)


_POW2 = np.array([1 << k for k in range(63)], dtype=np.int64)


def bit_length(v):
    """bit_length elemen-demi-elemen untuk int64 non-negatif (0 -> 0); eksak sampai 2^62."""
    v = np.asarray(v, dtype=np.int64)
    return np.searchsorted(_POW2, v, side="right").astype(np.int64)
