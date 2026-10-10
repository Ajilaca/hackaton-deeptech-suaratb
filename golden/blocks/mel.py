"""Filterbank mel 40 band (HTK, 50-8000 Hz), bobot Q0.12 sparse (maks. 2 band per bin).

E_b = min((sum_k P[k]*w_b[k]) >> MEL_SHIFT, 2^32-1).
Padanan RTL: mel.sv (ROM sparse berisi {band, bobot, akhir-bin}, 40 akumulator).
"""
from __future__ import annotations
import numpy as np
from .. import params


def _hz2mel(f):
    return 2595.0 * np.log10(1.0 + np.asarray(f, dtype=np.float64) / 700.0)


def _mel2hz(m):
    return 700.0 * (10.0 ** (np.asarray(m, dtype=np.float64) / 2595.0) - 1.0)


def mel_filterbank_float() -> np.ndarray:
    """(N_MEL, N_BINS) float, segitiga puncak 1 di domain frekuensi bin."""
    nb = params.N_MEL
    edges = _mel2hz(np.linspace(_hz2mel(params.MEL_FMIN), _hz2mel(params.MEL_FMAX), nb + 2))
    fk = np.arange(params.N_BINS) * params.FS / params.NFFT
    fb = np.zeros((nb, params.N_BINS))
    for b in range(nb):
        lo, c, hi = edges[b], edges[b + 1], edges[b + 2]
        up = (fk - lo) / (c - lo)
        dn = (hi - fk) / (hi - c)
        fb[b] = np.clip(np.minimum(up, dn), 0.0, None)
    return fb


def mel_weights_q() -> np.ndarray:
    top = (1 << params.MEL_W_FRAC) - 1
    q = np.floor(mel_filterbank_float() * (1 << params.MEL_W_FRAC) + 0.5)
    return np.clip(q, 0, top).astype(np.int64)


def mel_sparse():
    """Daftar (bin, band, bobot) bobot>0, urut bin lalu band."""
    w = mel_weights_q()
    return [(k, b, int(w[b, k])) for k in range(params.N_BINS) for b in range(params.N_MEL) if w[b, k] > 0]


def mel_energy(p, shift: int | None = None, in_bits: int | None = None) -> np.ndarray:
    shift = params.MEL_SHIFT if shift is None else shift
    in_bits = params.LOG_IN_BITS if in_bits is None else in_bits
    acc = np.asarray(p, dtype=np.int64) @ mel_weights_q().T
    return np.minimum(acc >> shift, (1 << in_bits) - 1)


def mel_acc(p) -> np.ndarray:
    return np.asarray(p, dtype=np.int64) @ mel_weights_q().T
