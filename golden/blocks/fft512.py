"""FFT 512 radix-2 DIT, masukan real, data W bit bertanda, twiddle Q1.15, skala 1/2 per stage.

Per butterfly (satu pembulatan, ditambah-setengah lalu geser):
    t = b * w                       (Q1.15, hasil 2W-ish bit)
    out_a = (a*2^15 + t + 2^(s+14)) >> (15+s)
    out_b = (a*2^15 - t + 2^(s+14)) >> (15+s)         s = FFT_STAGE_SHIFT[stage] (0/1)
Dengan s=1 di semua stage besaran kompleks tidak pernah melebihi masukan, jadi tidak ada
overflow (dijamin analitik) dan keluaran = FFT/512 * 2^G. Hitungan saturasi tetap dilaporkan.
Padanan RTL: fft512.sv (1 butterfly, RAM in-place, ROM twiddle 256 entri cos+sin).
"""
from __future__ import annotations
import numpy as np
from .. import params
from .fixed import sat


def twiddle_q15(n: int | None = None, frac: int | None = None):
    """(cos[k], sin[k]) untuk W_N^k = cos - j*sin, k=0..N/2-1, Q1.15, dipotong ke +-(2^15-1)."""
    n = params.NFFT if n is None else n
    frac = params.TW_FRAC if frac is None else frac
    k = np.arange(n // 2)
    lim = (1 << frac) - 1
    c = np.clip(np.floor(np.cos(2 * np.pi * k / n) * (1 << frac) + 0.5), -lim, lim).astype(np.int64)
    s = np.clip(np.floor(np.sin(2 * np.pi * k / n) * (1 << frac) + 0.5), -lim, lim).astype(np.int64)
    return c, s


def _bitrev(n: int) -> np.ndarray:
    bits = n.bit_length() - 1
    idx = np.arange(n)
    r = np.zeros(n, dtype=np.int64)
    for b in range(bits):
        r |= ((idx >> b) & 1) << (bits - 1 - b)
    return r


def fft512(xr, xi=None, W: int | None = None, stage_shift=None):
    """xr (...,512) int. Keluaran (re, im, n_sat) urutan alami."""
    W = params.FFT_W if W is None else W
    ss = params.FFT_STAGE_SHIFT if stage_shift is None else stage_shift
    n = params.NFFT
    stages = n.bit_length() - 1
    assert len(ss) == stages
    xr = np.asarray(xr, dtype=np.int64)
    xi = np.zeros_like(xr) if xi is None else np.asarray(xi, dtype=np.int64)
    perm = _bitrev(n)
    re, im = xr[..., perm].copy(), xi[..., perm].copy()
    lead = re.shape[:-1]
    cw, sw = twiddle_q15()
    n_sat = 0
    for st in range(stages):
        half = 1 << st
        m = half * 2
        sh = 15 + ss[st]
        rnd = 1 << (sh - 1)
        k = np.arange(half) * (n // m)
        wr, wi = cw[k], -sw[k]          # W = cos - j sin
        re4 = re.reshape(lead + (n // m, 2, half))
        im4 = im.reshape(lead + (n // m, 2, half))
        ar, br = re4[..., 0, :], re4[..., 1, :]
        ai, bi = im4[..., 0, :], im4[..., 1, :]
        tr = br * wr - bi * wi
        ti = br * wi + bi * wr
        ar15, ai15 = ar << 15, ai << 15
        o = np.empty_like(re4), np.empty_like(im4)
        o[0][..., 0, :] = (ar15 + tr + rnd) >> sh
        o[0][..., 1, :] = (ar15 - tr + rnd) >> sh
        o[1][..., 0, :] = (ai15 + ti + rnd) >> sh
        o[1][..., 1, :] = (ai15 - ti + rnd) >> sh
        lim = 1 << (W - 1)
        n_sat += int(np.count_nonzero((o[0] >= lim) | (o[0] < -lim) | (o[1] >= lim) | (o[1] < -lim)))
        re = sat(o[0], W).reshape(lead + (n,))
        im = sat(o[1], W).reshape(lead + (n,))
    return re, im, n_sat
