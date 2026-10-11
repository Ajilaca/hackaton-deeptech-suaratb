"""Fitur kualitas/kemiripan-batuk per klip (heuristik DITETAPKAN SEBELUM melihat label) + topeng.

    python -m model.clipq --tag v2

Per klip (8240 sampel, onset di frame PRE_ROLL=5): snr_db (puncak RMS vs median RMS frame pra-onset 0..3),
attack_db (kenaikan energi frame 4 -> 7), dur_fr (frame dalam 12 dB dari puncak), harm (autokorelasi ternormalisasi
maks 60-400 Hz pada frame terkeras 4..9), hf_ratio (energi >2 kHz / total pada frame terkeras).
Topeng 'cough_like': snr_db >= 10, attack_db >= 6, 3 <= dur_fr <= 30, harm < 0.6. Ambang tidak disetel pada label.
Keluaran: $SUARATB_DATA/features/clipq_<tag>.npz (selaras dengan meta_<tag>.csv).
"""
from __future__ import annotations
import argparse
import numpy as np
import pandas as pd
from golden import params, paths

FR, HOP = params.FRAME_LEN, params.HOP


def clip_feats(x: np.ndarray) -> dict:
    x = x.astype(np.float64)
    nf = (len(x) - FR) // HOP + 1
    idx = np.arange(nf)[:, None] * HOP + np.arange(FR)[None, :]
    fr = x[idx]
    rms = np.sqrt(np.mean(fr ** 2, axis=1)) + 1e-3
    noise = np.median(rms[0:4])
    pk = int(np.argmax(rms[4:10])) + 4
    snr = 20 * np.log10(rms[pk] / noise)
    e_db = 20 * np.log10(rms)
    attack = e_db[7] - e_db[4]
    dur = int(np.sum(e_db >= e_db[pk] - 12.0))
    seg = fr[pk] - fr[pk].mean()
    ac = np.correlate(seg, seg, "full")[FR - 1:]
    ac = ac / (ac[0] + 1e-9)
    lo, hi = int(params.FS / 400), int(params.FS / 60)
    harm = float(ac[lo:hi].max())
    sp = np.abs(np.fft.rfft(seg * np.hamming(FR), 512)) ** 2
    f = np.fft.rfftfreq(512, 1 / params.FS)
    hf = float(sp[f >= 2000].sum() / (sp.sum() + 1e-9))
    return dict(snr_db=snr, attack_db=attack, dur_fr=dur, harm=harm, hf_ratio=hf, peak=float(np.abs(x).max()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="v2")
    a = ap.parse_args()
    fd = paths.features_dir()
    meta = pd.read_csv(fd / f"meta_{a.tag}.csv")
    ci = pd.read_csv(paths.processed_dir() / "clip_index.csv").set_index("path")
    meta["k"] = meta.groupby("path").cumcount()
    rows = []
    cache = {}
    for r in meta.itertuples():
        if r.path not in cache:
            cache.clear()
            cache[r.path] = np.load(ci.loc[r.path, "out"])["clips"]
        rows.append(clip_feats(cache[r.path][r.k]))
    q = pd.DataFrame(rows)
    q["cough_like"] = (q.snr_db >= 10) & (q.attack_db >= 6) & (q.dur_fr >= 3) & (q.dur_fr <= 30) & (q.harm < 0.6)
    np.savez(fd / f"clipq_{a.tag}.npz", **{c: q[c].values for c in q.columns})
    print(q.describe().T[["mean", "25%", "50%", "75%"]].round(2).to_string())
    print("cough_like:", int(q.cough_like.sum()), "dari", len(q), f"({100 * q.cough_like.mean():.0f}%)")
    print(meta.assign(cl=q.cough_like.values).groupby("device").cl.mean().round(2).to_dict())
    print("peserta dengan >=3 klip lolos:", int((meta.assign(cl=q.cough_like.values).groupby("barcode").cl.sum() >= 3).sum()),
          "dari", meta.barcode.nunique())


if __name__ == "__main__":
    main()
