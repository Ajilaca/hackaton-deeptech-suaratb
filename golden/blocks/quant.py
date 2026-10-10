"""Kuantisasi fitur log-mel Q5.8 -> int8: q = sat8(round((L - OFFSET) / 2^SHIFT)).
Padanan RTL: quant.sv (pengurang, geser, saturasi)."""
from __future__ import annotations
import numpy as np
from .. import params
from .fixed import rshift_round, sat


def quant_feat(L, offset: int | None = None, shift: int | None = None) -> np.ndarray:
    o = params.FEAT_OFFSET if offset is None else offset
    s = params.FEAT_SHIFT if shift is None else shift
    return sat(rshift_round(np.asarray(L, dtype=np.int64) - o, s), 8)
