"""E3: embedding OPERA (CE, CT) per rekaman dan evaluasi dengan protokol E0.

    python -m model.embed_pretrained extract --model operaCE        # -> features/emb_operaCE.npz
    python -m model.embed_pretrained eval --models operaCE operaCT   # -> reports/embedding-pralatih.md

Resep OPERA (apa adanya): rekaman utuh, trim hening (frame 100 ms), mel 64 pita 50-8000 Hz (n_fft 1024, hop 512) dinormalisasi
min-max, masuk encoder tanpa dipotong. Bobot dimuat dengan torch.load(weights_only=True) dari $SUARATB_DATA/ext/opera_ck.
Embedding per rekaman, dirata-rata per peserta (semua perangkat). Hanya barcode asli/pola/tengah (audit E1).
"""
from __future__ import annotations
import argparse
import sys
import warnings
import numpy as np
import pandas as pd
import torch
from golden import paths

warnings.filterwarnings("ignore")
EXT = paths.data_root() / "ext"
CFG = {"operaCE": ("efficientnet", "encoder-operaCE.ckpt"), "operaCT": ("htsat", "encoder-operaCT.ckpt")}


def load_model(name: str):
    sys.path.insert(0, str(EXT / "opera"))
    from src.model.models_cola import Cola
    enc, ck = CFG[name]
    m = Cola(encoder=enc)
    z = torch.load(EXT / "opera_ck" / ck, weights_only=True, map_location="cpu")
    r = m.load_state_dict(z["state_dict"], strict=False)
    assert not r.missing_keys and not r.unexpected_keys, (r.missing_keys[:3], r.unexpected_keys[:3])
    return m.eval()


def spectrogram(x: np.ndarray, sr: int = 16000, min_sec: float = 8.0) -> np.ndarray:
    import librosa
    sys.path.insert(0, str(EXT / "opera"))
    from src.util import split_pad_sample, pre_process_audio_mel_t
    y, _ = librosa.effects.trim(x.astype(np.float32), frame_length=sr // 10, hop_length=sr // 20)
    if len(y) < sr // 4:                       # hampir hening: pakai seluruh rekaman
        y = x.astype(np.float32)
    if len(y) / sr < min_sec:
        y = split_pad_sample([y, 0, 0], min_sec, sr, "repeat")[0][0]
    return pre_process_audio_mel_t(y.squeeze(), f_max=8000)


def extract(name: str):
    from model.data.audio_io import load_16k
    fx = pd.read_csv(paths.processed_dir() / "barcode_fix.csv")
    fx = fx[fx.tier.isin(["asli", "pola", "tengah"]) & fx.path.str.lower().str.endswith(".wav")].reset_index(drop=True)
    torch.set_num_threads(4)
    m = load_model(name)
    dim = m.dim_fea
    E = np.zeros((len(fx), dim), np.float32)
    for i, r in enumerate(fx.itertuples()):
        x = load_16k(str(paths.raw_dir() / r.path))
        s = spectrogram(x)
        with torch.no_grad():
            if name == "operaCT":      # HTSAT gagal pada masukan panjang: potong per 8 s (251 frame), rata-rata
                ch = [s[j:j + 251] for j in range(0, max(len(s) - 125, 1), 251)]
                ch = [c for c in ch if len(c) >= 125]
                E[i] = np.mean([m.extract_feature(torch.tensor(np.pad(c, ((0, 251 - len(c)), (0, 0)), mode="edge")[None],
                                                                dtype=torch.float), dim).numpy()[0] for c in ch], axis=0)
            else:
                E[i] = m.extract_feature(torch.tensor(s[None], dtype=torch.float), dim).numpy()[0]
        if i % 100 == 0:
            print(name, i, len(fx), flush=True)
    np.savez(paths.features_dir() / f"emb_{name}.npz", emb=E, path=fx.path.values, barcode=fx.barcode_fix.values, device=fx.device.values)


def extract_hear(max_clips: int = 6, pref=("phonec", "phoneb", "phonea", "recorder"), tag: str = "hear", threads: int = 6):
    """HeAR (google/hear-pytorch): jendela 2 s mulai 0,25 s sebelum onset detektor golden; per peserta maks max_clips klip
    paling keras (energi puncak hop, tanpa label) dari SATU perangkat (urutan pref). Embedding = pooler_output (512)."""
    import importlib.util
    import os
    from transformers import AutoModel
    from golden import params
    from golden.blocks.cough_det import hop_energy, cough_det
    from golden.blocks.preemph import preemph
    from model.data.audio_io import load_16k
    spec = importlib.util.spec_from_file_location("au", str(EXT / "hear/python/data_processing/audio_utils.py"))
    au = importlib.util.module_from_spec(spec); spec.loader.exec_module(au)
    os.environ.setdefault("HF_HOME", str(paths.data_root() / "hf"))
    torch.set_num_threads(threads)
    m = AutoModel.from_pretrained("google/hear-pytorch").eval()
    fx = pd.read_csv(paths.processed_dir() / "barcode_fix.csv")
    fx = fx[fx.tier.isin(["asli", "pola", "tengah"]) & fx.path.str.lower().str.endswith(".wav")]
    pick = []
    for b, g in fx.groupby("barcode_fix"):
        for d in pref:
            gd = g[g.device == d]
            if len(gd):
                pick.append((b, d, gd.path.tolist()))
                break
    rows, embs = [], []
    for n, (b, d, ps) in enumerate(pick):
        cands = []
        for pth in ps:
            x = load_16k(str(paths.raw_dir() / pth)).astype(np.int64)
            E = hop_energy(preemph(x))
            for e in cough_det(E):
                lo = e.onset * params.HOP - 4000
                cands.append((float(E[e.onset:e.onset + 12].max()), pth, x, max(lo, 0)))
        cands.sort(key=lambda t: -t[0])
        for _, pth, x, s0 in cands[:max_clips]:
            w = x[s0:s0 + 32000].astype(np.float32) / 32768.0
            w = np.pad(w, (0, 32000 - len(w)))
            with torch.no_grad():
                o = m(au.preprocess_audio(torch.tensor(w)[None]), return_dict=True).pooler_output[0].numpy()
            embs.append(o); rows.append((pth, b, d))
        if n % 20 == 0:
            print("hear", n, len(pick), len(embs), flush=True)
    r = pd.DataFrame(rows, columns=["path", "barcode", "device"])
    np.savez(paths.features_dir() / f"emb_{tag}.npz", emb=np.stack(embs), path=r.path.values, barcode=r.barcode.values, device=r.device.values)


def participant_matrix(name: str, u: pd.DataFrame, device: str | None = None) -> np.ndarray:
    z = np.load(paths.features_dir() / f"emb_{name}.npz", allow_pickle=True)
    d = pd.DataFrame(z["emb"]); d["b"] = z["barcode"]; d["dev"] = z["device"]
    if device:
        d = d[d.dev == device]
    g = d.drop(columns="dev").groupby("b").mean()
    return g.reindex(u.barcode.values).values


def evaluate(models, n_perm=40, reps=10, out=None):
    from model.evaluation import protocol as P
    u = P.universe(); f = P.make_folds(u, reps)
    o = ["# Embedding pralatih (E3) dengan protokol E0", "", f"Semesta {len(u)} peserta, {int(u.y.sum())} positif. Resep OPERA: rekaman utuh, "
         "dirata-rata per peserta. Kriteria sinyal: AUROC strat. > p95 nol dan batas bawah CI Δ > 0.", ""] + P.HEAD + P.reference_rows(u)
    for nm in models:
        F = participant_matrix(nm, u)
        ok = ~np.isnan(F).any(1)
        us = u[ok].reset_index(drop=True)
        fs = P.make_folds(us, reps)
        o += P.evaluate(F[ok], us, fs, f"{nm} (n={len(us)}, positif {int(us.y.sum())})", n_perm)
        print(o[-1], flush=True)
    path = out or str(paths.runs_dir() / "reports" / "embedding-pralatih.md")
    import pathlib
    pathlib.Path(path).write_text("\n".join(o) + "\n")
    print("\n".join(o))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["extract", "eval", "hear"])
    ap.add_argument("--model", default="operaCE")
    ap.add_argument("--models", nargs="+", default=["operaCE", "operaCT"])
    ap.add_argument("--perm", type=int, default=40)
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default=None, help="hear: pakai hanya perangkat ini; keluaran emb_hear_<device>.npz")
    a = ap.parse_args()
    if a.cmd == "extract":
        extract(a.model)
    elif a.cmd == "hear":
        extract_hear(pref=(a.device,), tag=f"hear_{a.device}", threads=8) if a.device else extract_hear()
    else:
        evaluate(a.models, a.perm, a.reps, a.out)
