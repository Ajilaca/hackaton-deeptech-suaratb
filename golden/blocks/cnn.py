"""CNN int8 bit-exact: 3x [Conv3x3(same) -> requant -> ReLU -> MaxPool2x2], GAP, FC->ReLU, FC.

Requant: y = (acc * M0 + 2^(s-1)) >> s, M0 Q15 tak bertanda (0..32767), s = shift;
         ReLU+saturasi ke [0,127] (int8) pada layer tersembunyi; FC akhir: saturasi int16, tanpa ReLU.
Akumulator ACC_BITS bertanda; `assert` tanpa overflow (akumulator + bias).
GAP: (sum * round(2^15/(H*W)) + 2^14) >> 15.
Padanan RTL: pe_array / cnn_ctrl / requant / gap_fc (Fase 4).
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from .. import params
from .fixed import rshift_round, sat


@dataclass
class QLayer:
    name: str
    kind: str                 # 'conv' | 'fc'
    w: np.ndarray             # conv: (cout,cin,3,3) int8; fc: (cout,cin) int8
    b: np.ndarray             # (cout,) int32 (muat ACC_BITS)
    m0: int                   # Q15
    shift: int
    pool: bool = False
    final: bool = False       # FC terakhir: int16 tanpa ReLU
    gap_hw: int = 0           # >0: layer ini didahului GAP atas gap_hw elemen


@dataclass
class QModel:
    layers: list[QLayer] = field(default_factory=list)


class AccOverflow(AssertionError):
    pass


def check_acc(acc: np.ndarray, bits: int | None = None, where: str = "") -> int:
    bits = params.ACC_BITS if bits is None else bits
    lo, hi = -(1 << (bits - 1)), (1 << (bits - 1)) - 1
    mx = int(np.max(np.abs(acc))) if acc.size else 0
    if acc.size and (acc.min() < lo or acc.max() > hi):
        raise AccOverflow(f"akumulator {bits} bit overflow di {where}: min={acc.min()} max={acc.max()}")
    return mx


def requant(acc, m0: int, shift: int, relu: bool, out_bits: int = 8):
    assert 0 <= m0 < (1 << 15) and 1 <= shift <= params.REQ_SHIFT_MAX
    y = (np.asarray(acc, dtype=np.int64) * m0 + (1 << (shift - 1))) >> shift
    if relu:
        return np.clip(y, 0, (1 << (out_bits - 1)) - 1)
    return sat(y, out_bits)


def conv3x3(x: np.ndarray, w: np.ndarray, b: np.ndarray) -> np.ndarray:
    """x (cin,H,W) int; w (cout,cin,3,3); zero padding 'same'. Keluaran akumulator (cout,H,W) int64."""
    cin, h, wd = x.shape
    xp = np.pad(x.astype(np.int64), ((0, 0), (1, 1), (1, 1)))
    win = np.lib.stride_tricks.sliding_window_view(xp, (3, 3), axis=(1, 2))  # (cin,H,W,3,3)
    acc = np.einsum("chwij,ocij->ohw", win, w.astype(np.int64))
    return acc + b.astype(np.int64)[:, None, None]


def maxpool2(x: np.ndarray) -> np.ndarray:
    c, h, w = x.shape
    h2, w2 = h // 2, w // 2
    t = x[:, : h2 * 2, : w2 * 2].reshape(c, h2, 2, w2, 2)
    return t.max(axis=(2, 4))


def gap(x: np.ndarray) -> np.ndarray:
    c, h, w = x.shape
    recip = int(np.floor((1 << params.GAP_RECIP_FRAC) / (h * w) + 0.5))
    s = x.astype(np.int64).sum(axis=(1, 2))
    return (s * recip + (1 << (params.GAP_RECIP_FRAC - 1))) >> params.GAP_RECIP_FRAC


def cnn_forward(x: np.ndarray, model: QModel, stats: dict | None = None) -> np.ndarray:
    """x (1,40,50) int8 -> logit int16 (skalar array). stats: peak |acc| per layer."""
    a = np.asarray(x, dtype=np.int64)
    for L in model.layers:
        if L.gap_hw:
            a = gap(a)
            assert a.shape[0] == L.w.shape[1]
        if L.kind == "conv":
            acc = conv3x3(a, L.w, L.b)
        else:
            acc = L.w.astype(np.int64) @ a.reshape(-1) + L.b.astype(np.int64)
        mx = check_acc(acc, where=L.name)
        if stats is not None:
            stats[L.name] = max(stats.get(L.name, 0), mx)
        if L.final:
            a = requant(acc, L.m0, L.shift, relu=False, out_bits=params.LOGIT_BITS)
        else:
            a = requant(acc, L.m0, L.shift, relu=True)
            if L.pool:
                a = maxpool2(a)
    return a


def worst_case_acc(cin_k: int, a_max: int = 128, w_max: int = 128) -> int:
    """Batas atas |acc| tanpa bias: n_masukan * |a|max * |w|max."""
    return cin_k * a_max * w_max


def random_model(seed: int = 0, c1=None, c2=None, c3=None, hidden=None, w_scale: float = 24.0) -> QModel:
    """Model int8 ACAK (hanya untuk uji golden/vektor; BUKAN bobot terlatih)."""
    rng = np.random.default_rng(seed)
    kw = dict(c1=c1 or params.C1, c2=c2 or params.C2, c3=c3 or params.C3, hidden=hidden or params.FC_HIDDEN)
    m = QModel()
    for d in params.layer_descs(**kw):
        shp = (d["cout"], d["cin"], 3, 3) if d["kind"] == "conv" else (d["cout"], d["cin"])
        w = np.clip(np.rint(rng.normal(0, w_scale, shp)), -128, 127).astype(np.int8)
        b = np.rint(rng.normal(0, 200, d["cout"])).astype(np.int32)
        m.layers.append(QLayer(d["name"], d["kind"], w, b, m0=1 << 14, shift=15 + 8,
                               pool=d["pool"], final=(d["name"] == "fc2"), gap_hw=d.get("gap_hw", 0)))
    return m


def m0_shift_from_scale(m: float) -> tuple[int, int]:
    """m real > 0 -> (M0 in [2^14, 2^15), shift) dengan M0 * 2^-shift ~= m."""
    assert m > 0
    s = 15
    while m * (1 << s) < (1 << 14):
        s += 1
    while m * (1 << s) >= (1 << 15):
        s -= 1
    return int(np.floor(m * (1 << s) + 0.5)), s


def calibrate_requant(model: QModel, inputs: list[np.ndarray], target: int = 100, pct: float = 99.9) -> None:
    """Pilih (M0, shift) per layer agar persentil `pct` aktivasi pra-requant -> `target`. Dijalankan berurutan."""
    for li, L in enumerate(model.layers):
        accs = []
        for x in inputs:
            a = np.asarray(x, dtype=np.int64)
            for K in model.layers[:li]:
                if K.gap_hw:
                    a = gap(a)
                acc = conv3x3(a, K.w, K.b) if K.kind == "conv" else K.w.astype(np.int64) @ a.reshape(-1) + K.b
                a = requant(acc, K.m0, K.shift, relu=not K.final, out_bits=params.LOGIT_BITS if K.final else 8)
                if K.pool:
                    a = maxpool2(a)
            if L.gap_hw:
                a = gap(a)
            acc = conv3x3(a, L.w, L.b) if L.kind == "conv" else L.w.astype(np.int64) @ a.reshape(-1) + L.b
            accs.append(np.abs(acc).ravel())
        p = max(float(np.percentile(np.concatenate(accs), pct)), 1.0)
        tgt = target if not L.final else 2000
        L.m0, L.shift = m0_shift_from_scale(tgt / p)


def save_qmodel(model: QModel, path) -> None:
    """Simpan QModel ke .npz (bobot int8, bias int32, M0, shift, flag layer)."""
    d = {"n": np.array(len(model.layers))}
    for i, L in enumerate(model.layers):
        d[f"w{i}"] = L.w.astype(np.int8); d[f"b{i}"] = L.b.astype(np.int32)
        d[f"meta{i}"] = np.array([L.m0, L.shift, int(L.pool), int(L.final), L.gap_hw, int(L.kind == "conv")])
        d[f"name{i}"] = np.array(L.name)
    np.savez(path, **d)


def load_qmodel(path) -> QModel:
    z = np.load(path)
    m = QModel()
    for i in range(int(z["n"])):
        m0, sh, pool, final, gap_hw, is_conv = (int(v) for v in z[f"meta{i}"])
        m.layers.append(QLayer(str(z[f"name{i}"]), "conv" if is_conv else "fc", z[f"w{i}"].astype(np.int8),
                               z[f"b{i}"].astype(np.int32), m0, sh, bool(pool), bool(final), gap_hw))
    return m
