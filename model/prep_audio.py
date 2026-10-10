"""Segmentasi batuk memakai DETEKTOR GOLDEN (sama dengan yang akan ada di hardware).

    python -m model.prep_audio --devices recorder phonea [--workers 3]

Keluaran di $SUARATB_DATA/processed/clips/<device>/<barcode>.npz:
  clips (n, 8240) int16 mulai onset-PRE_ROLL frame; onsets (n,) frame; hop_rms (nf,) float32 (RMS per hop)
dan $SUARATB_DATA/processed/clip_index.csv. Tidak ada label di berkas klip; label diambil dari metadata saat melatih.
"""
from __future__ import annotations
import argparse
import hashlib
import multiprocessing as mp
import pathlib
import numpy as np
import pandas as pd
from golden import params, paths
from golden.blocks.preemph import preemph
from golden.blocks.cough_det import hop_energy, cough_det
from model.audio_io import load_16k
from model import dataset


def process(args):
    rel, device, barcode, out_dir = args
    out = pathlib.Path(out_dir) / device / f"{barcode}.npz"
    # beberapa peserta punya >1 berkas: akhiri nama dengan stem berkas bila bentrok
    out = out.with_name(f"{pathlib.Path(rel).stem.replace(' ', '_')}.npz")
    if out.exists():
        z = np.load(out)
        return dict(path=rel, out=str(out), n_clips=len(z["onsets"]), dur=float(z["dur"]), hash=str(z["hash"]), status="cached")
    try:
        x = load_16k(str(paths.raw_dir() / rel))
    except Exception as e:
        return dict(path=rel, out=None, n_clips=0, dur=0.0, hash="", status=f"ERR {type(e).__name__}")
    pe = preemph(x)
    E = hop_energy(pe)
    ev = cough_det(E)
    n = params.CLIP_SAMPLES_PADDED
    clips, ons = [], []
    for e in ev:
        s = e.start * params.HOP
        seg = np.zeros(n, dtype=np.int16)
        part = x[s:s + n]
        seg[:len(part)] = part
        clips.append(seg); ons.append(e.onset)
    h = hashlib.md5(x[: 16000 * 20].tobytes()).hexdigest()
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, clips=np.array(clips, dtype=np.int16).reshape(-1, n), onsets=np.array(ons, dtype=np.int32),
                        hop_rms=np.sqrt(E / params.HOP).astype(np.float32), dur=len(x) / 16000.0, hash=h)
    return dict(path=rel, out=str(out), n_clips=len(ev), dur=len(x) / 16000.0, hash=h, status="ok")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--devices", nargs="+", default=["recorder"])
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    idx = dataset.index_audio(with_header=False)
    idx = idx[idx.device.isin(a.devices) & idx.barcode.notna() & idx.name.str.lower().str.endswith(".wav")]
    out_dir = paths.processed_dir() / "clips"
    jobs = [(r.path, r.device, r.barcode, str(out_dir)) for r in idx.itertuples()]
    print(f"{len(jobs)} rekaman")
    with mp.Pool(a.workers) as p:
        res = p.map(process, jobs, chunksize=4)
    df = pd.DataFrame(res).merge(idx[["path", "device", "barcode"]], on="path")
    prev = paths.processed_dir() / "clip_index.csv"
    if prev.exists():
        df = pd.concat([pd.read_csv(prev), df]).drop_duplicates("path", keep="last")
    df.to_csv(prev, index=False)
    print(df.groupby("device").agg(n=("path", "size"), clips=("n_clips", "sum"), err=("status", lambda s: int(s.str.startswith("ERR").sum()))))
    print("klip per rekaman: median", df.n_clips.median(), "min", df.n_clips.min(), "maks", df.n_clips.max())
    print("hash duplikat:", int(df.duplicated("hash").sum()))


if __name__ == "__main__":
    main()
