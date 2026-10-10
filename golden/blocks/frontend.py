"""Rangkaian front-end golden: sampel int16 -> fitur log-mel int8 (frame, 40).

Fungsi murni; semua parameter default dari params, bisa ditimpa untuk eksperimen Fase 0.
"""
from __future__ import annotations
import numpy as np
from .. import params
from .preemph import preemph
from .window import apply_window
from .fft512 import fft512
from .power import power
from .mel import mel_energy, mel_filterbank_float
from .log_q import log2_q
from .quant import quant_feat
from .fixed import rshift_round


def frame_signal(x: np.ndarray, n_frames: int | None = None) -> np.ndarray:
    nf = (len(x) - params.FRAME_LEN) // params.HOP + 1 if n_frames is None else n_frames
    idx = np.arange(nf)[:, None] * params.HOP + np.arange(params.FRAME_LEN)[None, :]
    x = np.asarray(x)
    return (x if x.dtype.kind == "f" else x.astype(np.int64))[idx]


def pad_clip(clip: np.ndarray) -> np.ndarray:
    """Klip 0,5 s -> 8240 sampel (nol di ekor) agar tepat N_FRAMES frame."""
    x = np.zeros(params.CLIP_SAMPLES_PADDED, dtype=np.int64)
    n = min(len(clip), len(x))
    x[:n] = np.asarray(clip, dtype=np.int64)[:n]
    return x


def log_mel_q(x, W=None, mel_shift=None, stage_shift=None, with_internal: bool = False, in_bits=None):
    """x int16 (aliran). Mengembalikan L (frame,40) Q6.8 [+ dict internal]."""
    pf = (params.FFT_W if W is None else W) - 16          # bit pecahan pre-emphasis = guard FFT (K-08)
    x_pe = preemph(x, frac=pf)
    fr = frame_signal(x_pe)
    win = apply_window(fr, W, in_frac=pf)
    re, im, nsat = fft512(win, W=W, stage_shift=stage_shift)
    p = power(re, im)
    e = mel_energy(p, mel_shift, in_bits)
    L = log2_q(e)
    if with_internal:
        return L, dict(x_pe=x_pe, x_pe16=rshift_round(x_pe, pf), win=win, re=re, im=im, p=p, e=e, n_sat=nsat)
    return L


def features_q(x, W=None, mel_shift=None, offset=None, shift=None, stage_shift=None, in_bits=None) -> np.ndarray:
    return quant_feat(log_mel_q(x, W, mel_shift, stage_shift, in_bits=in_bits), offset, shift).astype(np.int8)


def log_mel_ref(x, W=None, mel_shift=None, stage_shift=None, x_pe=None) -> np.ndarray:
    """Referensi float64 dengan konvensi skala yang sama (FFT/2^sum(stage_shift) * 2^G, / 2^MEL_SHIFT).
    Mengembalikan log2(E) * 256 (satuan L), tanpa pembulatan; E<0,5 dipotong ke -256.
    x_pe: bila diberikan, dipakai sebagai keluaran pre-emphasis (mengisolasi galat lebar FFT)."""
    W = params.FFT_W if W is None else W
    ms = params.MEL_SHIFT if mel_shift is None else mel_shift
    xf = np.asarray(x, dtype=np.float64)
    if x_pe is None:
        pe = xf.copy()
        pe[1:] -= (params.PREEMPH_Q15 / 32768.0) * xf[:-1]
    else:                      # isolasi pengaruh lebar FFT: pakai keluaran preemph integer golden
        pe = np.asarray(x_pe, dtype=np.float64)
    fr = frame_signal(pe)
    k = np.arange(params.FRAME_LEN)
    w = 0.54 - 0.46 * np.cos(2 * np.pi * k / (params.FRAME_LEN - 1))
    ss = params.FFT_STAGE_SHIFT if stage_shift is None else stage_shift
    X = np.fft.rfft(fr * w, n=params.NFFT, axis=-1) / (1 << sum(ss)) * (1 << (W - 16))
    pw = X.real ** 2 + X.imag ** 2
    e = (pw @ mel_filterbank_float().T) * (1 << params.MEL_W_FRAC) / (1 << ms)
    return 256.0 * np.log2(np.maximum(e, 2.0 ** -1))


def input_floor_L(W=None) -> np.ndarray:
    """L (log2*256, skala yang sama dengan log_mel_ref) per band dari derau kuantisasi masukan int16
    (putih, var 1/12 LSB^2, setelah pre-emphasis dan window). Di bawah ini fitur hanya derau kuantisasi."""
    W = params.FFT_W if W is None else W
    a = params.PREEMPH_Q15 / 32768.0
    var = (1.0 / 12.0) * (1.0 + a * a)
    k = np.arange(params.FRAME_LEN)
    w = 0.54 - 0.46 * np.cos(2 * np.pi * k / (params.FRAME_LEN - 1))
    ss = sum(params.FFT_STAGE_SHIFT)
    pk = var * np.sum(w * w) / (1 << (2 * ss)) * (1 << (2 * (W - 16)))     # E|X_k|^2 (rata-rata, semua bin)
    e = pk * mel_filterbank_float().sum(axis=1) * (1 << params.MEL_W_FRAC) / (1 << params.MEL_SHIFT)
    return 256.0 * np.log2(e)


def segment(L_or_q: np.ndarray, start: int, n: int | None = None) -> np.ndarray:
    """Potong n_frames mulai `start` (nol bila di luar rentang). Hasil (n, 40)."""
    n = params.N_FRAMES if n is None else n
    out = np.zeros((n,) + L_or_q.shape[1:], dtype=L_or_q.dtype)
    lo, hi = max(start, 0), min(start + n, L_or_q.shape[0])
    if hi > lo:
        out[lo - start: hi - start] = L_or_q[lo:hi]
    return out


def to_cnn_input(feat_q: np.ndarray) -> np.ndarray:
    """(50 frame, 40 mel) -> (1, 40 mel, 50 frame) = (C, H, W)."""
    return feat_q.T[None, :, :].astype(np.int64)
