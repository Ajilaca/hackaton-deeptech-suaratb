"""Langkah 1: evaluasi TBscreen dalam-dataset dan lintas-dataset (Zambia <-> TBscreen) dengan kontrol perancu.

    python -m model.tbs_eval [--out $SUARATB_DATA/runs/reports/tbscreen.md]

Fitur peserta (80): rata-rata antar klip dari [mean, std] per band log-mel int8 HANYA pada frame aktif
(energi hop > puncak/32; sama di kedua dataset). Model: LR (C via CV dalam). Kontrol: cohort (A=rekrutmen TB, C=non-TB),
jenis kelamin, usia; permutasi label dalam cohort; per perangkat; dalam jenis kelamin.
Klaim yang diperbolehkan: lihat CLAUDE.md (tidak ada klaim klinis/generalisasi).
"""
from __future__ import annotations
import argparse
import pathlib
import warnings
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from golden import paths
from model import dataset

warnings.filterwarnings("ignore")


def clip_vec(c: np.ndarray, act: np.ndarray) -> np.ndarray:
    a = c[act] if act.sum() >= 3 else c
    return np.concatenate([a.mean(0), a.std(0)])


def participant_table(x, act, ids) -> tuple[np.ndarray, np.ndarray]:
    V = np.stack([clip_vec(x[i].astype(np.float32), act[i]) for i in range(len(x))])
    u, inv = np.unique(ids, return_inverse=True)
    P = np.zeros((len(u), V.shape[1]))
    for k in range(len(u)):
        P[k] = V[inv == k].mean(0)
    return u, P


def load_zambia(tag="v2"):
    fd = paths.features_dir()
    x = np.load(fd / f"feats_{tag}.npz")["x_w24"]
    act = np.load(fd / f"active_{tag}.npz")["active"]
    meta = pd.read_csv(fd / f"meta_{tag}.csv")
    u, P = participant_table(x, act, meta.barcode.values)
    g = dataset.load_meta().set_index("barcode").loc[u]
    d = pd.DataFrame(dict(id=u, y=g.y.values, sex_m=(g.sex.astype(str).str.lower() == "m").astype(int).values,
                          age=g.age.values.astype(float), group=g.facility_code.values))
    return d, P


def load_tbs(kind="passive", device=None, cap=None):
    fd = paths.features_dir()
    z = np.load(fd / "feats_tbs.npz")
    meta = pd.read_csv(fd / "meta_tbs.csv")
    sel = (meta.kind == kind).to_numpy().copy()
    if device:
        sel = sel & (meta.device == device).to_numpy()
    u, P = participant_table(z["x_w24"][sel], z["active"][sel], meta.barcode.values[sel])
    g = meta[sel].drop_duplicates("barcode").set_index("barcode").loc[u]
    d = pd.DataFrame(dict(id=u, y=g.y.values, sex_m=g.sex_m.values, age=g.age.values.astype(float),
                          group=g.cohort.values))
    d["age"] = d.age.fillna(d.age.median())
    return d, P


def mk():
    return make_pipeline(SimpleImputer(), StandardScaler(),
                         LogisticRegressionCV(Cs=np.logspace(-3, 0, 7), cv=3, class_weight="balanced",
                                              scoring="roc_auc", max_iter=2000))


def cv_auc(X, y, strat, reps=5):
    a, oof = [], []
    for r in range(reps):
        s = np.zeros(len(y))
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=r).split(X, strat):
            m = mk(); m.fit(X[tr], y[tr]); s[te] = m.decision_function(X[te])
        a.append(roc_auc_score(y, s)); oof.append(s)
    return float(np.mean(a)), float(np.std(a)), np.mean(oof, axis=0)


def ci(y, s, n=2000, seed=0):
    rng = np.random.default_rng(seed); v = []
    for _ in range(n):
        j = rng.choice(len(y), len(y))
        if y[j].min() != y[j].max():
            v.append(roc_auc_score(y[j], s[j]))
    return np.percentile(v, 2.5), np.percentile(v, 97.5)


def fit_apply(Xtr, ytr, Xte, znorm: bool):
    if znorm:
        Xtr = (Xtr - Xtr.mean(0)) / (Xtr.std(0) + 1e-6); Xte = (Xte - Xte.mean(0)) / (Xte.std(0) + 1e-6)
    m = mk(); m.fit(Xtr, ytr)
    return m.decision_function(Xte)


def row(name, y, s, note=""):
    lo, hi = ci(y, s)
    return f"| {name} | {len(y)} / {int(y.sum())} | {roc_auc_score(y, s):.3f} | [{lo:.3f}, {hi:.3f}] | {note} |"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(paths.runs_dir() / "reports" / "tbscreen.md"))
    ap.add_argument("--perm", type=int, default=30)
    a = ap.parse_args()
    dz, Pz = load_zambia()
    dt, Pt = load_tbs("passive")
    rng = np.random.default_rng(0)
    o = ["# TBscreen: evaluasi dalam-dataset dan lintas-dataset", "",
         "Sumber: Sharma dkk. 2024 (Nairobi, CC BY 4.0), klip 1 s, tiga perangkat (Pixel, boundary mic, Yeti). "
         "Pembanding: Zambia CIDRZ (360 peserta). Fitur peserta: mean/std log-mel int8 pada frame aktif, sama di kedua dataset. "
         "Kolom peserta/positif = n / TB.", "",
         "## Komposisi TBscreen (klip pasif, berizin audio)"]
    tab = pd.crosstab([dt.group, dt.y], dt.sex_m)
    o += [f"- subjek: {len(dt)}; TB {int(dt.y.sum())}, NTB {int((1 - dt.y).sum())}; "
          f"cohort A (rekrutmen TB) {int((dt.group == 'A').sum())}, cohort C (rekrutmen non-TB) {int((dt.group == 'C').sum())}",
          f"- TB per cohort: A: TB {int(((dt.group == 'A') & (dt.y == 1)).sum())}, NTB {int(((dt.group == 'A') & (dt.y == 0)).sum())}; "
          f"C: TB {int(((dt.group == 'C') & (dt.y == 1)).sum())}, NTB {int(((dt.group == 'C') & (dt.y == 0)).sum())}",
          "- **Cohort hampir identik dengan label** (A = TB hampir seluruhnya, C = non-TB seluruhnya): kondisi rekaman, tanggal, dan "
          "tim rekrutmen tidak dapat dipisahkan dari TB. AUROC dalam-TBscreen karenanya BUKAN bukti penanda akustik TB.", ""]
    # 1. perancu tanpa audio
    o += ["## Perancu tanpa audio (TBscreen)", "| prediktor | n / TB | AUROC | CI 95% | |", "|---|---|---|---|---|"]
    y = dt.y.values
    o.append(row("cohort saja (A=1)", y, (dt.group == "A").astype(float).values, "label hampir = cohort"))
    o.append(row("jenis kelamin laki-laki", y, dt.sex_m.values.astype(float)))
    o.append(row("usia", y, -dt.age.values))
    # 2. dalam-dataset
    o += ["", "## Dalam-dataset (CV 5-lipat x 5, per subjek)", "| model | n / TB | AUROC | CI 95% | catatan |", "|---|---|---|---|---|"]
    strat = np.array([f"{a_}_{b}" for a_, b in zip(dt.y, dt.group)])
    m_, sd, oof = cv_auc(Pt, y, strat)
    o.append(row("audio (semua perangkat, pasif)", y, oof, f"±{sd:.3f} antar ulangan"))
    null = []
    for _ in range(a.perm):
        yp = y.copy()
        for g in dt.group.unique():
            i = np.where(dt.group.values == g)[0]; yp[i] = rng.permutation(y[i])
        sp = np.array([f"{a_}_{b}" for a_, b in zip(yp, dt.group)])
        null.append(cv_auc(Pt, yp, sp, reps=1)[0])
    o.append(f"| nol permutasi (label diacak dalam cohort) | - | rata-rata {np.mean(null):.3f} | p95 {np.percentile(null, 95):.3f} | maks {np.max(null):.3f} |")
    for dev in ("pixel", "codec", "yeti"):
        d2, P2 = load_tbs("passive", dev)
        s2 = np.array([f"{a_}_{b}" for a_, b in zip(d2.y, d2.group)])
        o.append(row(f"audio, perangkat {dev}", d2.y.values, cv_auc(P2, d2.y.values, s2)[2]))
    for sx, nm in ((1, "laki-laki"), (0, "perempuan")):
        mk_ = (dt.sex_m == sx).values
        if len(set(y[mk_])) == 2 and y[mk_].sum() >= 8:
            s2 = np.array([f"{a_}_{b}" for a_, b in zip(dt.y[mk_], dt.group[mk_])])
            o.append(row(f"audio, hanya {nm}", y[mk_], cv_auc(Pt[mk_], y[mk_], s2)[2], "jenis kelamin terkendali"))
    # 3. lintas-dataset
    o += ["", "## Lintas-dataset (latih satu, uji lain; tanpa penyetelan pada data uji)",
          "| arah | n uji / TB | AUROC | CI 95% | catatan |", "|---|---|---|---|---|"]
    for zn, zl in ((False, "fitur mentah"), (True, "z-norm per dataset")):
        s = fit_apply(Pz, dz.y.values, Pt, zn)
        o.append(row(f"Zambia → TBscreen ({zl})", y, s))
    for zn, zl in ((False, "fitur mentah"), (True, "z-norm per dataset")):
        s = fit_apply(Pt, y, Pz, zn)
        o.append(row(f"TBscreen → Zambia ({zl})", dz.y.values, s))
    # 4. fitur menangkap jenis kelamin lintas dataset?
    s = fit_apply(Pz, dz.sex_m.values, Pt, True)
    o.append(row("audio → jenis kelamin: Zambia → TBscreen (z-norm)", dt.sex_m.values, s, "kontrol: fitur menangkap sex"))
    s = fit_apply(Pz, dz.y.values, Pt, True)
    for sx, nm in ((1, "laki-laki"), (0, "perempuan")):
        mk_ = (dt.sex_m == sx).values
        if len(set(y[mk_])) == 2:
            o.append(row(f"Zambia → TBscreen, hanya {nm} (z-norm)", y[mk_], s[mk_]))
    o += ["", "## Interpretasi (diisi setelah angka dibaca)", "- (lihat ringkasan di status-fase.md)"]
    pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True); pathlib.Path(a.out).write_text("\n".join(o) + "\n")
    print("\n".join(o))


if __name__ == "__main__":
    main()
