"""Detektor batuk berbasis energi (EMA noise floor + rasio ambang + hold).

Per frame f (tiap HOP sampel): E[f] = sum(x_pe[n]^2 >> DET_E_SHIFT) atas 160 sampel TERBARU frame f.
Noise floor (pecahan DET_EMA_SHIFT bit): Nq += ((E << S) - Nq) >> S ; N = max(Nq >> S, N_FLOOR).
Trigger: E > K*N selama >= MIN_HOPS frame berurutan -> onset = frame pertama di atas ambang.
Segmen fitur = frame [onset - PRE_ROLL, onset + POST_ONSET - 1]; N dibekukan selama segmen
direkam; setelah itu refractory HOLD_HOPS frame. (Tafsiran 'tahan 300 ms' = refractory: K-07.)
Padanan RTL: cough_det.sv (Fase 2).
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .. import params
from .fixed import sat


def hop_energy(x_pe: np.ndarray, n_frames: int | None = None, e_shift: int | None = None) -> np.ndarray:
    """E[f] atas sampel [f*HOP + FRAME_LEN - HOP, f*HOP + FRAME_LEN). x_pe int16 (pra-emphasis)."""
    es = params.DET_E_SHIFT if e_shift is None else e_shift
    x = np.asarray(x_pe, dtype=np.int64)
    nf = (len(x) - params.FRAME_LEN) // params.HOP + 1 if n_frames is None else n_frames
    out = np.zeros(max(nf, 0), dtype=np.int64)
    for f in range(nf):
        s = f * params.HOP + params.FRAME_LEN - params.HOP
        seg = x[s: s + params.HOP]
        out[f] = int(np.sum(seg * seg)) >> es
    return out


@dataclass
class Event:
    onset: int            # frame onset
    start: int            # frame awal segmen (onset - PRE_ROLL, bisa < 0 -> dipotong ke 0)
    end: int              # frame akhir inklusif


def cough_det(E, k: int | None = None, ema_shift: int | None = None, min_hops: int | None = None,
              hold: int | None = None, n_floor: int | None = None) -> list[Event]:
    k = params.DET_K if k is None else k
    S = params.DET_EMA_SHIFT if ema_shift is None else ema_shift
    mh = params.DET_MIN_HOPS if min_hops is None else min_hops
    hold = params.DET_HOLD_HOPS if hold is None else hold
    floor = params.DET_N_FLOOR if n_floor is None else n_floor
    E = [int(v) for v in E]
    ev: list[Event] = []
    nq = None
    run = 0
    cap_end = -1        # frame terakhir yang masih direkam
    dead_until = -1
    for f, e in enumerate(E):
        if nq is None:
            nq = e << S
        n = max(nq >> S, floor)
        recording = f <= cap_end
        if not recording:
            nq += ((e << S) - nq) >> S      # EMA hanya saat idle/refractory
        above = e > k * n
        if recording or f <= dead_until:
            run = 0
            continue
        run = run + 1 if above else 0
        if run >= mh:
            onset = f - mh + 1
            ev.append(Event(onset, max(onset - params.PRE_ROLL, 0), onset + params.POST_ONSET - 1))
            cap_end = onset + params.POST_ONSET - 1
            dead_until = cap_end + hold
            run = 0
    return ev
