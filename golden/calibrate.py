"""Fase 0: eksperimen penentuan lebar FFT, skala, ambang detektor, dan konstanta kuantisasi.

    python -m golden.calibrate                 # audio sintetis (default sampai data/raw ada)
    python -m golden.calibrate --data data/raw # audio nyata (tanpa label: hanya statistik fitur)

Hasil ditulis ke golden/out/phase0_calibration.json (di .gitignore). Tidak mengubah params.py;
nilai beku disalin manual ke params.py dan dicatat di docs/vault/keputusan.md.
"""
from __future__ import annotations
import argparse
import json
import pathlib
import numpy as np
from . import params, synth
from .blocks import frontend as F
from .blocks.preemph import preemph
from .blocks.cough_det import hop_energy, cough_det
from .blocks.fft512 import fft512
from .blocks.window import apply_window

OUT = pathlib.Path(__file__).parent / "out"


def load_audio_dir(path: str, limit: int = 200):
    import soundfile as sf
    from math import gcd
    files = sorted(p for p in pathlib.Path(path).rglob("*") if p.suffix.lower() in (".wav", ".flac"))[:limit]
    out = []
    for p in files:
        x, fs = sf.read(str(p), dtype="float64", always_2d=True)
        x = x.mean(axis=1)
        if fs != params.FS:
            from scipy.signal import resample_poly
            g = gcd(int(fs), params.FS)
            x = resample_poly(x, params.FS // g, int(fs) // g)
        out.append(np.clip(np.rint(x * 32767), -32768, 32767).astype(np.int16))
    return out


def ensemble(seeds=(0, 1, 2)):
    """Klip sintetis: derau latar sigma 1,5..100 LSB, batuk 10/20 dB, derau saja, bicara."""
    clips = []
    for sg in (1.5, 3.0, 10.0, 30.0, 100.0):
        for s in seeds:
            for kind, snr in (("cough", 10.0), ("cough", 25.0), ("noise", 0.0), ("speech", 15.0)):
                clips.append((f"{kind}@sg{sg}/snr{snr}", sg, synth.clip(1000 * s + int(sg * 10), sg, snr, kind)))
    return clips


def _err_octave(L, ref, floor_l, margin_db: float = 10.0):
    """RMS galat (oktaf) hanya pada elemen >= lantai derau kuantisasi masukan + margin.
    Di bawahnya fitur hanyalah derau kuantisasi int16 masukan itu sendiri (tak ada informasi)."""
    m = ref >= (floor_l[None, :] + 256.0 * np.log2(10 ** (margin_db / 10.0)))
    if not m.any():
        return float("nan"), 0.0
    d = (L.astype(np.float64) - ref)[m] / 256.0
    return float(np.sqrt(np.mean(d ** 2))), float(m.mean())


def fft_snr_db(x, W, stage_shift=None):
    ss = params.FFT_STAGE_SHIFT if stage_shift is None else stage_shift
    xp = preemph(x, frac=W - 16)
    fr = F.frame_signal(xp)
    win = apply_window(fr, W, in_frac=W - 16)
    re, im, ns = fft512(win, W=W, stage_shift=stage_shift)
    ref = np.fft.fft(win.astype(np.float64), axis=-1) / (1 << sum(ss))
    num = np.sum(ref.real ** 2 + ref.imag ** 2)
    den = np.sum((re - ref.real) ** 2 + (im - ref.imag) ** 2)
    return float(10 * np.log10(num / max(den, 1e-30))), ns


def exp_fft_width(clips, widths=(16, 18, 20, 22, 23, 24, 25)):
    """Galat fitur vs referensi float pada elemen >= lantai derau kuantisasi masukan + 10 dB.
    'fft' = referensi memakai preemph golden (mengisolasi lebar FFT/window); 'e2e' = float murni."""
    rows = []
    for W in widths:
        by = {}
        for name, sg, x in clips:
            xp = F.pad_clip(x)
            snr, ns = fft_snr_db(xp, W)
            L = F.log_mel_q(xp, W=W, mel_shift=0, in_bits=60)
            fl = F.input_floor_L(W)
            e_fft, frac = _err_octave(L, F.log_mel_ref(xp, W=W, mel_shift=0, x_pe=preemph(xp, frac=W - 16) / float(1 << (W - 16))), fl)
            e_e2e, _ = _err_octave(L, F.log_mel_ref(xp, W=W, mel_shift=0), fl)
            b = by.setdefault(sg, dict(snr=[], fft=[], e2e=[], frac=[], sat=0))
            b["snr"].append(snr); b["fft"].append(e_fft); b["e2e"].append(e_e2e); b["frac"].append(frac); b["sat"] += ns
        rows.append(dict(W=W, per_sigma={str(sg): dict(fft_snr_db=float(np.mean(v["snr"])),
                                                       feat_err_octave_fft=float(np.nanmean(v["fft"])),
                                                       feat_err_octave_e2e=float(np.nanmean(v["e2e"])),
                                                       frac_above_inputfloor=float(np.mean(v["frac"])),
                                                       fft_saturations=v["sat"]) for sg, v in by.items()}))
    return rows


def exp_stage_schedule(widths=(18, 20, 22, 24)):
    """Jadwal skala per stage x lebar: SNR FFT pada derau tenang dan saturasi pada batuk keras."""
    scheds = {"all_1": (1,) * 9, "first1_0": (0,) + (1,) * 8, "first2_0": (0, 0) + (1,) * 7,
              "first3_0": (0, 0, 0) + (1,) * 6}
    levels = {"q_sg3": synth.clip(3, 3.0, 15.0, "cough"),               # ruang tenang
              "mid_rms1000": synth.clip(5, 30.0, 30.5, "cough"),        # ~ -30 dBFS
              "loud_rms3000": synth.clip(6, 30.0, 40.0, "cough"),       # ~ -20 dBFS
              "loud_rms10000": synth.clip(7, 30.0, 50.4, "cough")}      # ~ -10 dBFS
    rows = []
    for W in widths:
        for nm, ss in scheds.items():
            r = dict(W=W, schedule=nm, total_shift=sum(ss))
            for ln, x in levels.items():
                xp = F.pad_clip(x)
                snr, ns = fft_snr_db(xp, W, ss)
                r[ln] = dict(fft_snr_db=round(snr, 1), saturations=ns, peak=int(np.max(np.abs(x))))
            rows.append(r)
    return rows


def exp_quant_ranges(clips, W):
    Ls = np.concatenate([F.log_mel_q(F.pad_clip(x), W=W, mel_shift=0, in_bits=60).ravel() for _, _, x in clips])
    nz = Ls[Ls > 0]
    lo, hi = np.percentile(nz, 0.5), np.percentile(nz, 99.5)
    shift = int(np.ceil(np.log2(max(hi - lo, 1) / 254.0)))
    off = int(np.rint((lo + hi) / 2))
    return dict(L_p0_5=float(lo), L_p99_5=float(hi), L_max=int(Ls.max()), shift=shift, offset=off,
                step_octave=2 ** shift / 256.0, frac_zero=float(np.mean(Ls == 0)))


def exp_mel_shift(clips, W):
    """Opsi A (Q5.8, masukan 32 bit): kisaran MEL_SHIFT agar tidak saturasi vs lantai tak hilang."""
    from .blocks.mel import mel_acc
    peak, floor = 0, None
    for _, sg, x in clips:
        _, i = F.log_mel_q(F.pad_clip(x), W=W, mel_shift=0, with_internal=True)
        acc = mel_acc(i["p"])
        peak = max(peak, int(acc.max()))
        q = np.percentile(acc[acc > 0], 1)
        floor = q if floor is None else min(floor, q)
    need = max(int(peak).bit_length() - params.LOG_IN_BITS, 0)
    return dict(peak_acc_bits=int(peak).bit_length(), p1_floor_bits=int(floor).bit_length(),
                shift_no_saturation_on_ensemble=need)


# ---- detektor -----------------------------------------------------------------

def _det_streams():
    pos_streams, neg_streams = [], []
    for sg in (3.0, 10.0, 30.0):
        for snr in (6.0, 10.0, 15.0, 25.0):
            x, pos = synth.stream(int(sg * 7 + snr), 20.0, sg, 6, snr)
            pos_streams.append((sg, snr, x, pos))
        neg_streams.append((f"noise sg{sg}", synth.stream(int(sg) + 500, 60.0, sg, 0, 0.0)[0], 60.0))
        neg_streams.append((f"click sg{sg}", synth.stream(int(sg) + 600, 60.0, sg, 0, 20.0,
                                                          distract=("click",) * 10)[0], 60.0))
        neg_streams.append((f"tone sg{sg}", synth.stream(int(sg) + 700, 60.0, sg, 0, 15.0,
                                                         distract=("tone",) * 8)[0], 60.0))
        neg_streams.append((f"step x4 sg{sg}", synth.stream(int(sg) + 800, 60.0, sg, 0, 0.0, level_step=4.0)[0], 60.0))
    return pos_streams, neg_streams


def exp_detector():
    pos, neg = _det_streams()
    Epos = [(sg, snr, hop_energy(preemph(x)), p) for sg, snr, x, p in pos]  # detektor: int16
    Eneg = [(nm, hop_energy(preemph(x)), sec) for nm, x, sec in neg]
    rows = []
    for k in (4, 6, 8, 12, 16):
        for S in (4, 5, 6, 7):
            for mh in (2, 3):
                hit = {}
                for sg, snr, E, ps in Epos:
                    ev = cough_det(E, k=k, ema_shift=S, min_hops=mh)
                    for p in ps:
                        f = p // params.HOP
                        ok = any(f - 3 <= e.onset <= f + 14 for e in ev)
                        hit.setdefault(snr, []).append(ok)
                rec = {str(s): float(np.mean(v)) for s, v in sorted(hit.items())}
                cnt, mins = {}, {}
                for nm, E, sec in Eneg:
                    cat = nm.split()[0]
                    cnt[cat] = cnt.get(cat, 0) + len(cough_det(E, k=k, ema_shift=S, min_hops=mh))
                    mins[cat] = mins.get(cat, 0.0) + sec / 60.0
                fpc = {c: cnt[c] / mins[c] for c in cnt}
                # tone = bunyi keras non-batuk: tak dapat ditolak detektor energi -> dilaporkan terpisah
                fp = sum(cnt[c] for c in cnt if c != "tone") / sum(mins[c] for c in mins if c != "tone")
                rows.append(dict(k=k, ema_shift=S, min_hops=mh, recall=rec, fp_per_min=fp, fp_by_cat=fpc))
    return rows


def choose_detector(rows, fp_max=0.5, snr_gate="10.0"):
    ok = [r for r in rows if r["fp_per_min"] <= fp_max]
    if not ok:
        return None
    best = max(r["recall"][snr_gate] for r in ok)
    cand = [r for r in ok if r["recall"][snr_gate] >= best - 1e-9]
    return min(cand, key=lambda r: (abs(np.log2(r["k"] / 8)) + abs(r["ema_shift"] - 6) + (r["min_hops"] - 2)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None, help="folder audio nyata (mis. $SUARATB_DATA/raw)")
    a = ap.parse_args()
    res = {"source": "synthetic"}
    clips = ensemble()
    if a.data and pathlib.Path(a.data).exists() and any(pathlib.Path(a.data).rglob("*.wav")):
        real = load_audio_dir(a.data)
        clips = [(f"real{i}", 0, c[: params.CLIP_SAMPLES]) for i, c in enumerate(real)]
        res["source"] = "data/raw"
    res["fft_width"] = exp_fft_width(clips)
    res["stage_schedule"] = exp_stage_schedule()
    res["quant_W18"] = exp_quant_ranges(clips, 18)
    res["quant_W24"] = exp_quant_ranges(clips, 24)
    res["mel_shift_W18"] = exp_mel_shift(clips, 18)
    res["mel_shift_W24"] = exp_mel_shift(clips, 24)
    det = exp_detector()
    res["detector"] = det
    res["detector_choice"] = choose_detector(det)
    OUT.mkdir(exist_ok=True)
    (OUT / "phase0_calibration.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "detector"}, indent=1))


if __name__ == "__main__":
    main()
