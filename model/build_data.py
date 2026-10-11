"""Bangun tabel klip berlabel, fitur golden (beberapa lebar FFT) + fitur float-ref, dan split per peserta.

    python -m model.build_data [--workers 3] [--offset 8265] [--shift 5]

Keluaran ($SUARATB_DATA/features): feats_v1.npz (x_w24, x_w22, x_w20, x_w18, x_ref: int8 (N,50,40)),
meta_v1.csv (barcode,y,facility,device,path,onset_idx); ($SUARATB_DATA/processed): splits_v1.csv (barcode,fold).
OFFSET diberikan untuk W=24; untuk W lain digeser 512*(W-24) (energi ~ 2^(2G)) sehingga pemetaan nyatanya sama.
Tidak mengubah golden/params.py.
"""
from __future__ import annotations
import argparse
import multiprocessing as mp
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from golden import params, paths
from golden.blocks import frontend as F
from golden.blocks.preemph import preemph
from golden.blocks.quant import quant_feat
from model import dataset

WIDTHS = (24, 22, 20, 18)


def _feat(args):
    clip, off24, shift = args
    out = []
    for W in WIDTHS:
        L = F.log_mel_q(clip, W=W, mel_shift=0, in_bits=60)
        out.append(quant_feat(L, off24 + 512 * (W - 24), shift).astype(np.int8))
    ref = F.log_mel_ref(clip, W=24, mel_shift=0)
    out.append(np.clip(np.floor((ref - off24) / 2 ** shift + 0.5), -128, 127).astype(np.int8))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--offset", type=int, default=8265)
    ap.add_argument("--shift", type=int, default=5)
    ap.add_argument("--tag", default="v1")
    ap.add_argument("--repeats", type=int, default=1, help="jumlah pengulangan split 5-lipat (kolom fold0..)")
    ap.add_argument("--devices", nargs="*", default=None, help="batasi perangkat (mis. recorder phonea)")
    a = ap.parse_args()
    idx = pd.read_csv(paths.processed_dir() / "clip_index.csv")
    idx = idx[idx.status.isin(["ok", "cached"]) & (idx.n_clips > 0)].drop_duplicates("hash")
    meta = dataset.load_meta()[["barcode", "y", "facility_code"]]
    if a.devices:
        idx = idx[idx.device.isin(a.devices)]
    idx = idx.merge(meta, on="barcode", how="inner")          # buang rekaman tanpa label
    clips, rows = [], []
    for r in idx.itertuples():
        z = np.load(r.out)
        for k in range(len(z["onsets"])):
            clips.append(z["clips"][k]); rows.append((r.barcode, int(r.y), r.facility_code, r.device, r.path, int(z["onsets"][k])))
    m = pd.DataFrame(rows, columns=["barcode", "y", "facility", "device", "path", "onset"])
    print(f"{len(m)} klip, {m.barcode.nunique()} peserta, positif peserta {m.drop_duplicates('barcode').y.sum()}")
    with mp.Pool(a.workers) as p:
        res = p.map(_feat, [(c.astype(np.int64), a.offset, a.shift) for c in clips], chunksize=16)
    names = [f"x_w{w}" for w in WIDTHS] + ["x_ref"]
    arrs = {n: np.stack([r[i] for r in res]) for i, n in enumerate(names)}
    fd = paths.features_dir(); fd.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(fd / f"feats_{a.tag}.npz", **arrs, offset=a.offset, shift=a.shift)
    m.to_csv(fd / f"meta_{a.tag}.csv", index=False)
    pt = m.drop_duplicates("barcode")[["barcode", "y"]].reset_index(drop=True)
    fac = m.drop_duplicates("barcode").facility.values
    strat = pt.y.astype(str) + "_" + fac               # stratifikasi y x klinik
    for r in range(a.repeats):
        skf = StratifiedKFold(5, shuffle=True, random_state=r)
        col = "fold" if (r == 0 and a.repeats == 1) else f"fold{r}"
        pt[col] = -1
        for f, (_, te) in enumerate(skf.split(pt, strat)):
            pt.loc[te, col] = f
    if a.repeats > 1:
        pt["fold"] = pt["fold0"]
    pt.to_csv(paths.processed_dir() / f"splits_{a.tag}.csv", index=False)
    print(pt.groupby("fold").y.agg(["size", "sum"]).T.to_string())
    print(m.groupby("device").agg(klip=("y", "size"), peserta=("barcode", "nunique")).to_string())
    for n, v in arrs.items():
        print(n, v.shape, "jenuh %.2f%%" % (100 * np.mean(np.abs(v.astype(int)) >= 127)))


if __name__ == "__main__":
    main()
