"""TBscreen (Nairobi, UW, CC BY 4.0): klip 1 s -> segmen 8240 sampel -> fitur golden int8, selaras format Zambia.

    python -m model.data.tbscreen build [--cap 60] [--workers 4]
    python -m model.data.tbscreen zambia_active [--tag v2]

Klip TBscreen sudah dipotong satu batuk per berkas (1 s, nol di ekor/tengah), jadi detektor golden tidak dipakai:
onset = frame pertama dengan energi hop > puncak/16 (-12 dB), segmen mulai di (onset - PRE_ROLL) frame, nol-isi di ekor.
Keluaran ($SUARATB_DATA/features): feats_tbs.npz (x_w24 int8 (N,50,40), active bool (N,50), peak int16),
meta_tbs.csv (barcode=subject, y, device, kind, cohort, sex_m, age, path).
`active`: frame dengan energi hop > puncak/32 (dipakai sama untuk Zambia agar statistik tidak didominasi diam/nol).
Catatan label: Label TB vs NTB (penyakit paru lain); cohort A = rekrutmen TB, C = rekrutmen non-TB (PERANCU kuat).
"""
from __future__ import annotations
import argparse
import multiprocessing as mp
import os
import numpy as np
import pandas as pd
from golden import params, paths
from golden.blocks import frontend as F
from golden.blocks.cough_det import hop_energy
from golden.blocks.preemph import preemph
from golden.blocks.quant import quant_feat
from model.data.audio_io import load_16k

ROOT = paths.data_root() / "raw_tbscreen" / "TBscreen_Dataset"
N = params.CLIP_SAMPLES_PADDED
OFFSET, SHIFT = params.FEAT_OFFSET, params.FEAT_SHIFT


def segment(x: np.ndarray) -> np.ndarray:
    """x int16 16 kHz (1 s) -> 8240 sampel, onset di frame PRE_ROLL."""
    pe = preemph(np.pad(x, (0, max(0, params.FRAME_LEN - len(x)))))
    E = hop_energy(pe)
    thr = E.max() / 16 if E.max() > 0 else 1
    on = int(np.argmax(E > thr))
    s = max(0, on - params.PRE_ROLL) * params.HOP
    seg = np.zeros(N, dtype=np.int16)
    part = x[s:s + N]
    seg[:len(part)] = part
    return seg


def active_mask(seg: np.ndarray) -> np.ndarray:
    E = hop_energy(preemph(seg))
    return (E > E.max() / 32) if E.max() > 0 else np.zeros(len(E), bool)


def _job(a):
    path = a
    seg = segment(load_16k(path))
    L = F.log_mel_q(seg.astype(np.int64), W=24, mel_shift=0, in_bits=60)
    return quant_feat(L, OFFSET, SHIFT).astype(np.int8), active_mask(seg), seg


def build(cap: int, workers: int, seed: int = 0):
    rows = []
    for kind, csv, sub in (("passive", "Passive_coughs/Passive_coughs.csv", "Passive_coughs"),
                           ("forced", "Forced_coughs/Forced_coughs.csv", "Forced_coughs")):
        d = pd.read_csv(ROOT / csv, low_memory=False)
        have = set(os.listdir(ROOT / sub / "Audio_files"))
        d = d[d.path.map(lambda p: p + ".wav" in have)].copy()
        d["wav"] = d.path.map(lambda p: str(ROOT / sub / "Audio_files" / (p + ".wav")))
        d["kind"] = kind
        rows.append(d)
    d = pd.concat(rows, ignore_index=True)
    d = d[d.Label.isin(["TB", "NTB"])]
    if cap:   # tutup klip pasif per subjek (acak tetap) agar subjek berklip banyak tidak mendominasi
        keep = []
        for (_, k), g in d.groupby(["subject", "kind"]):
            keep.append(g if k == "forced" or len(g) <= cap else g.sample(cap, random_state=seed))
        d = pd.concat(keep)
    d = d.sort_values("path").reset_index(drop=True)
    print(f"{len(d)} klip, {d.subject.nunique()} subjek, TB {d.drop_duplicates('subject').Label.eq('TB').sum()}", flush=True)
    with mp.Pool(workers) as p:
        res = p.map(_job, list(d.wav), chunksize=32)
    x = np.stack([r[0] for r in res]); act = np.stack([r[1] for r in res])
    clips = np.stack([r[2] for r in res])
    m = pd.DataFrame(dict(barcode=d.subject, y=(d.Label == "TB").astype(int), device=d.device, kind=d.kind,
                          cohort=d.cohort, sex_m=(d.gndr_a == 0).astype(int), age=d.age, path=d.path))
    fd = paths.features_dir(); fd.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(fd / "feats_tbs.npz", x_w24=x, active=act, clips=clips)
    m.to_csv(fd / "meta_tbs.csv", index=False)
    print(m.groupby(["kind", "device"]).agg(klip=("y", "size"), subjek=("barcode", "nunique")).to_string())
    print("jenuh %.2f%%" % (100 * np.mean(np.abs(x.astype(int)) >= 127)))


def zambia_active(tag: str = "v2"):
    """Topeng frame aktif untuk klip Zambia (urutan sama dengan meta_<tag>.csv)."""
    fd = paths.features_dir()
    meta = pd.read_csv(fd / f"meta_{tag}.csv")
    ci = pd.read_csv(paths.processed_dir() / "clip_index.csv").set_index("path")
    meta["k"] = meta.groupby("path").cumcount()
    out, cache = [], {}
    for r in meta.itertuples():
        if r.path not in cache:
            cache.clear()
            cache[r.path] = np.load(ci.loc[r.path, "out"])["clips"]
        out.append(active_mask(cache[r.path][r.k]))
    np.savez_compressed(fd / f"active_{tag}.npz", active=np.stack(out))
    print(len(out), "klip; frame aktif median", int(np.median(np.stack(out).sum(1))))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "zambia_active"])
    ap.add_argument("--cap", type=int, default=60)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--tag", default="v2")
    a = ap.parse_args()
    build(a.cap, a.workers) if a.cmd == "build" else zambia_active(a.tag)
