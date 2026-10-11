"""Cek lintas-perangkat pada peserta Zambia yang sama: perekam, Phone A, B, C.

    python -m model.device_eval [--out $SUARATB_DATA/runs/reports/lintas-perangkat.md]

Fitur peserta = mean+std log-mel (frame aktif) dari klip batuk perangkat itu; model = LR (C via CV dalam).
Split 5-lipat PER PESERTA (peserta yang sama di perangkat lain tidak pernah ada di latih dan uji sekaligus).
Dilaporkan: AUROC dalam-perangkat, latih perangkat X -> uji perangkat Y, dan latih semua perangkat.
"""
from __future__ import annotations
import argparse
import pathlib
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from golden import params, paths
from golden.blocks.cough_det import hop_energy
from golden.blocks.preemph import preemph
from model.tbs_eval import clip_vec, mk, ci

warnings.filterwarnings("ignore")


def active(clip: np.ndarray) -> np.ndarray:
    E = hop_energy(preemph(clip.astype(np.int64)))
    return (E > E.max() / 32) if E.max() > 0 else np.zeros(len(E), bool)


def vectors_bc():
    """feats_v3bc.npz: urutan baris = urutan idx build_data -> klip diambil ulang dari berkas hasil prep_audio."""
    fd = paths.features_dir()
    x = np.load(fd / "feats_v3bc.npz")["x_w24"]
    meta = pd.read_csv(fd / "meta_v3bc.csv")
    idx = pd.read_csv(paths.processed_dir() / "clip_index.csv")
    idx = idx[idx.status.isin(["ok", "cached"]) & (idx.n_clips > 0)].drop_duplicates("hash")
    idx = idx[idx.device.isin(["phoneb", "phonec"])]
    paths_ok = set(meta.path)
    idx = idx[idx.path.isin(paths_ok)]
    acts = []
    for r in idx.itertuples():
        z = np.load(r.out)
        for c in z["clips"]:
            acts.append(active(c))
    return x, np.stack(acts), meta


def per_participant(x, act, meta):
    V = np.stack([clip_vec(x[i].astype(np.float32), act[i]) for i in range(len(x))])
    out = {}
    for dev, g in meta.groupby("device"):
        ix = g.index.values
        u, inv = np.unique(g.barcode.values, return_inverse=True)
        P = np.stack([V[ix[inv == k]].mean(0) for k in range(len(u))])
        y = g.drop_duplicates("barcode").set_index("barcode").loc[u].y.values.astype(int)
        out[dev] = (u, P, y)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(paths.runs_dir() / "reports" / "lintas-perangkat.md"))
    ap.add_argument("--reps", type=int, default=3)
    a = ap.parse_args()
    fd = paths.features_dir()
    # perekam + Phone A (v2) dan B/C (v3bc)
    x2 = np.load(fd / "feats_v2.npz")["x_w24"]; a2 = np.load(fd / "active_v2.npz")["active"]
    m2 = pd.read_csv(fd / "meta_v2.csv")
    xb, ab, mb = vectors_bc()
    D = per_participant(x2, a2, m2)
    D.update(per_participant(xb, ab, mb))
    devs = [d for d in ("recorder", "phonea", "phoneb", "phonec") if d in D]
    allp = sorted(set().union(*[set(D[d][0]) for d in devs]))
    ylab = {}
    for d in devs:
        for b, yy in zip(D[d][0], D[d][2]):
            ylab[b] = yy
    allp = np.array(allp); yall = np.array([ylab[b] for b in allp])
    res_in = {d: [] for d in devs}
    res_x = {(s, t): [] for s in devs for t in devs if s != t}
    res_all = {d: [] for d in devs}
    oof_in = {d: np.full((a.reps, len(allp)), np.nan) for d in devs}
    for r in range(a.reps):
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=r).split(allp, yall):
            btr, bte = set(allp[tr]), set(allp[te])
            sel = {}
            for d in devs:
                u, P, y = D[d]
                mtr = np.array([b in btr for b in u]); mte = np.array([b in bte for b in u])
                sel[d] = (P[mtr], y[mtr], P[mte], y[mte], u[mte])
            for s in devs:
                if len(set(sel[s][1])) < 2:
                    continue
                m = mk(); m.fit(sel[s][0], sel[s][1])
                for t in devs:
                    Xte, yte = sel[t][2], sel[t][3]
                    if len(Xte):
                        sc = m.decision_function(Xte)
                        if s == t:
                            for b, v in zip(sel[t][4], sc):
                                oof_in[t][r, np.where(allp == b)[0][0]] = v
                        else:
                            res_x[(s, t)].append((yte, sc))
            Xa = np.vstack([sel[d][0] for d in devs]); ya = np.concatenate([sel[d][1] for d in devs])
            m = mk(); m.fit(Xa, ya)
            for d in devs:
                if len(sel[d][2]):
                    res_all[d].append((sel[d][3], m.decision_function(sel[d][2])))

    def pooled(lst):
        y = np.concatenate([p[0] for p in lst]); s = np.concatenate([p[1] for p in lst])
        return y, s

    o = ["# Cek lintas-perangkat (peserta Zambia yang sama)", "",
         "Fitur peserta = mean+std log-mel frame aktif. LR, split 5-lipat per peserta (peserta yang sama tidak muncul di latih dan "
         f"uji). {a.reps} ulangan; AUROC dari skor terkumpul. Perangkat: {', '.join(devs)}. "
         "Dalam-perangkat = latih dan uji perangkat sama; silang = latih satu perangkat, uji perangkat lain pada peserta uji. "
         "Catatan: tabel silang dan 'semua perangkat' menumpuk skor 3 ulangan (n = peserta x ulangan, bukan independen) sehingga CI-nya "
         "terlalu sempit; baca sebagai indikasi.", "",
         "## Dalam-perangkat", "| perangkat | peserta | positif | AUROC | CI 95% |", "|---|---|---|---|---|"]
    for d in devs:
        ok = ~np.isnan(oof_in[d]).all(0)
        yk = yall[ok]; sk = np.nanmean(oof_in[d][:, ok], 0)
        lo, hi = ci(yk, sk)
        o.append(f"| {d} | {len(yk)} | {int(yk.sum())} | {roc_auc_score(yk, sk):.3f} | [{lo:.3f}, {hi:.3f}] |")
    o += ["", "## Silang-perangkat (latih -> uji)", "| latih | uji | n uji / positif | AUROC | CI 95% |", "|---|---|---|---|---|"]
    for (s, t), lst in res_x.items():
        if not lst:
            continue
        y, sc = pooled(lst)
        lo, hi = ci(y, sc)
        o.append(f"| {s} | {t} | {len(y)} / {int(y.sum())} | {roc_auc_score(y, sc):.3f} | [{lo:.3f}, {hi:.3f}] |")
    o += ["", "## Latih semua perangkat -> uji per perangkat", "| uji | n / positif | AUROC | CI 95% |", "|---|---|---|---|"]
    for d, lst in res_all.items():
        y, sc = pooled(lst)
        lo, hi = ci(y, sc)
        o.append(f"| {d} | {len(y)} / {int(y.sum())} | {roc_auc_score(y, sc):.3f} | [{lo:.3f}, {hi:.3f}] |")
    out = pathlib.Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(o) + "\n")
    print("\n".join(o))


if __name__ == "__main__":
    main()
