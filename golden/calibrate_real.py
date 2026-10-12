"""Kalibrasi Fase 0 pada klip batuk NYATA (hasil segmentasi detektor golden). Tidak mengubah params.py.

    python -m golden.calibrate_real [--n-rec 150] [--out docs/vault/kalibrasi-data-nyata.md]

Masukan: $SUARATB_DATA/processed/clips/*/*.npz (dari model.data.prep_audio). Tanpa label klinis di sini;
AUROC ada di model/baseline.py. Detektor: tanpa ground truth batuk, hanya statistik laju kejadian.
"""
from __future__ import annotations
import argparse
import pathlib
import numpy as np
import pandas as pd
from . import params, paths
from .blocks import frontend as F
from .blocks.preemph import preemph
from .blocks.fft512 import fft512
from .blocks.window import apply_window
from .blocks.cough_det import cough_det
from .blocks.mel import mel_acc


def load_clips(n_rec: int, seed: int = 0):
    idx = pd.read_csv(paths.processed_dir() / "clip_index.csv")
    idx = idx[(idx.status.isin(["ok", "cached"])) & (idx.n_clips > 0)].drop_duplicates("hash")
    idx = idx.sample(min(n_rec, len(idx)), random_state=seed)
    clips, recs = [], []
    for r in idx.itertuples():
        z = np.load(r.out)
        clips.append(z["clips"]); recs.append(z["hop_rms"])
    return np.concatenate(clips).astype(np.int64), recs


def err_octave(L, ref, floor_l, margin_db=10.0):
    m = ref >= (floor_l[None, :] + 256.0 * np.log2(10 ** (margin_db / 10.0)))
    if not m.any():
        return float("nan"), 0.0
    return float(np.sqrt(np.mean(((L - ref)[m] / 256.0) ** 2))), float(m.mean())


def width_table(clips, widths=(16, 18, 20, 22, 24), n=200):
    rows = []
    sub = clips[:: max(1, len(clips) // n)][:n]
    for W in widths:
        errs, fr, snrs, sats = [], [], [], 0
        fl = F.input_floor_L(W)
        for x in sub:
            L = F.log_mel_q(x, W=W, mel_shift=0, in_bits=60)
            ref = F.log_mel_ref(x, W=W, mel_shift=0, x_pe=preemph(x, frac=W - 16) / float(1 << (W - 16)))
            e, f = err_octave(L, ref, fl)
            errs.append(e); fr.append(f)
            fr_ = F.frame_signal(preemph(x, frac=W - 16)); win = apply_window(fr_, W, in_frac=W - 16)
            re, im, ns = fft512(win, W=W); sats += ns
            r = np.fft.fft(win.astype(float), axis=-1) / 512
            snrs.append(10 * np.log10(np.sum(np.abs(r) ** 2) / max(np.sum((re - r.real) ** 2 + (im - r.imag) ** 2), 1e-12)))
        rows.append((W, float(np.nanmean(errs)), float(np.nanpercentile(errs, 90)), float(np.mean(fr)),
                     float(np.mean(snrs)), float(np.percentile(snrs, 10)), sats))
    return rows


def schedule_table(clips, W=24, top=60):
    peaks = np.abs(clips).max(axis=1)
    sub = clips[np.argsort(-peaks)[:top]]
    sch = {"semua 1/2": (1,) * 9, "1 stage tanpa skala": (0,) + (1,) * 8, "2 stage": (0, 0) + (1,) * 7,
           "3 stage": (0, 0, 0) + (1,) * 6}
    out = []
    for nm, ss in sch.items():
        sat = 0; clips_sat = 0
        for x in sub:
            fr = F.frame_signal(preemph(x, frac=W - 16)); win = apply_window(fr, W, in_frac=W - 16)
            _, _, ns = fft512(win, W=W, stage_shift=ss); sat += ns; clips_sat += ns > 0
        out.append((nm, sat, clips_sat))
    return out, int(peaks.max()), int(np.percentile(peaks, 50))


def quant_ranges(clips, W=24, n=400):
    sub = clips[:: max(1, len(clips) // n)][:n]
    Ls = np.concatenate([F.log_mel_q(x, W=W, mel_shift=0, in_bits=60).ravel() for x in sub])
    lo, hi = np.percentile(Ls[Ls > 0], 0.5), np.percentile(Ls[Ls > 0], 99.5)
    shift = int(np.ceil(np.log2(max(hi - lo, 1) / 254.0)))
    off = int(np.rint((lo + hi) / 2))
    q = np.clip(np.floor((Ls - off) / 2 ** shift + 0.5), -128, 127)
    cur = np.clip(np.floor((Ls - params.FEAT_OFFSET) / 2 ** params.FEAT_SHIFT + 0.5), -128, 127)
    pc = {p: float(np.percentile(Ls[Ls > 0], p)) for p in (0.5, 1, 5, 50, 95, 99, 99.5)}
    return dict(pct=pc, offset=off, shift=shift, sat_new=float(np.mean(np.abs(q) >= 127)),
                sat_cur=float(np.mean((cur >= 127) | (cur <= -128))), zero=float(np.mean(Ls == 0)))


def mel_bits(clips, n=100):
    sub = clips[:: max(1, len(clips) // n)][:n]
    mx = 0
    for x in sub:
        _, i = F.log_mel_q(x, mel_shift=0, in_bits=60, with_internal=True)
        mx = max(mx, int(mel_acc(i["p"]).max()))
    return mx.bit_length()


def detector_rates(recs, ks=(4, 6, 8, 12, 16), ss=(5, 6, 7)):
    rows = []
    for k in ks:
        for s in ss:
            ne, mins = [], 0.0
            for hr in recs:
                E = (hr.astype(np.float64) ** 2 * params.HOP).astype(np.int64)
                ne.append(len(cough_det(E, k=k, ema_shift=s)))
                mins += len(hr) * params.HOP / params.FS / 60.0
            rows.append((k, s, float(np.median(ne)), float(np.sum(ne) / mins), float(np.mean(np.array(ne) == 0))))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-rec", type=int, default=150)
    ap.add_argument("--out", default="docs/vault/kalibrasi-data-nyata.md")
    a = ap.parse_args()
    clips, recs = load_clips(a.n_rec)
    wt = width_table(clips)
    st, pk_max, pk_med = schedule_table(clips)
    qr = quant_ranges(clips)
    mb = mel_bits(clips)
    dr = detector_rates(recs)
    hr = np.concatenate(recs)
    o = ["# Kalibrasi Fase 0 pada audio NYATA (perekam, Zambia CIDRZ)", "",
         f"Dihasilkan `python -m golden.calibrate_real` dari {len(clips)} klip batuk (hasil detektor golden) pada "
         f"{len(recs)} rekaman unik. **params.py tidak diubah**; angka di bawah adalah masukan keputusan.", "",
         "## Tingkat sinyal",
         f"RMS per hop (LSB int16): p10/p50/p90/p99 = " + "/".join(f"{np.percentile(hr, p):.0f}" for p in (10, 50, 90, 99)),
         f"Puncak klip: median {pk_med}, maks {pk_max} (skala penuh 32767). Batas atas akumulator mel teramati: {mb} bit (batas analitik {params.MEL_ACC_BITS}).", "",
         "## Lebar FFT: galat fitur vs float (oktaf, elemen ≥ lantai derau kuantisasi + 10 dB) dan SNR FFT",
         "| W | galat rata-rata | galat p90 klip | fraksi elemen dinilai | SNR FFT rata-rata (dB) | SNR p10 (dB) | saturasi |",
         "|---|---|---|---|---|---|---|"]
    for W, e, e90, fr, s, s10, sat in wt:
        o.append(f"| {W} | {e:.3f} | {e90:.3f} | {fr:.2f} | {s:.1f} | {s10:.1f} | {sat} |")
    o += ["", f"Ambang keputusan K-04: galat ≤ 0,0625 oktaf (0,5 langkah int8).", "",
          "## Jadwal skala (W=24) pada 60 klip berpuncak tertinggi", "| jadwal | sampel jenuh | klip jenuh |", "|---|---|---|"]
    for nm, sat, cs in st:
        o.append(f"| {nm} | {sat} | {cs} |")
    p = qr["pct"]
    o += ["", "## Kuantisasi fitur (W=24, L Q6.8)",
          "persentil L (0,5/1/5/50/95/99/99,5): " + " / ".join(f"{p[k]:.0f}" for k in (0.5, 1, 5, 50, 95, 99, 99.5)),
          f"- nilai berjalan (sintetis): OFFSET {params.FEAT_OFFSET}, SHIFT {params.FEAT_SHIFT}: jenuh int8 {100*qr['sat_cur']:.2f}%",
          f"- usulan dari data nyata: **OFFSET {qr['offset']}, SHIFT {qr['shift']}**: jenuh int8 {100*qr['sat_new']:.2f}%; elemen nol {100*qr['zero']:.3f}%", "",
          "## Detektor: laju kejadian pada rekaman utuh (tanpa ground truth)",
          "| K | EMA S | median kejadian/rekaman | kejadian/menit | rekaman tanpa kejadian |", "|---|---|---|---|---|"]
    for k, s, med, rate, z in dr:
        o.append(f"| {k} | {s} | {med:.0f} | {rate:.1f} | {100*z:.0f}% |")
    pathlib.Path(a.out).write_text("\n".join(o) + "\n")
    print("\n".join(o))


if __name__ == "__main__":
    main()
