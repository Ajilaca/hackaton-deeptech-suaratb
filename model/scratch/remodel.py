"""Butir 1: pengklasifikasi tingkat peserta (akumulator statistik log-mel + regresi logistik) dengan kontrol ketat.

    python -m model.scratch.remodel --tag v2 [--perm 30]

Varian fitur: klip {semua, cough_like} x normalisasi {mentah, kurangi derau pra-onset} x perangkat {semua}.
C dipilih dengan CV dalam (LogisticRegressionCV). Evaluasi: 5-lipat stratified (klinik x label) x 5 ulangan;
nol = permutasi label DALAM klinik. Laporan: $SUARATB_DATA/runs/reports/remodel-lr.md.
"""
from __future__ import annotations
import argparse
import pathlib
import warnings
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from golden import paths
from model.clinical_models import fusion
from model.data import dataset

warnings.filterwarnings("ignore")


def load(tag):
    fd = paths.features_dir()
    Z = np.load(fd / f"feats_{tag}.npz"); meta = pd.read_csv(fd / f"meta_{tag}.csv")
    q = np.load(fd / f"clipq_{tag}.npz")
    sp = pd.read_csv(paths.processed_dir() / f"splits_{tag}.csv")
    return Z["x_w24"].astype(np.float32), meta, q["cough_like"].astype(bool), sp


def clip_vec(c: np.ndarray, sub: bool) -> np.ndarray:
    """c (50,40) -> vektor statistik klip."""
    if sub:
        c = c - c[:4].mean(axis=0, keepdims=True)
    return np.concatenate([c.mean(axis=0), c.std(axis=0), c.mean(axis=1)[::5]])


def participant_features(x, meta, mask, parts, sub, use_mask):
    F = np.zeros((len(parts), 90)); n_used = np.zeros(len(parts), int)
    by = meta.groupby("barcode").indices
    for i, p in enumerate(parts):
        idx = by[p]
        if use_mask:
            sel = idx[mask[idx]]
            if len(sel) == 0:
                sel = idx                                  # cadangan: semua klip peserta
        else:
            sel = idx
        F[i] = np.mean([clip_vec(x[j], sub) for j in sel], axis=0)
        n_used[i] = len(sel)
    return F, n_used


def make_model():
    return make_pipeline(SimpleImputer(), StandardScaler(),
                         LogisticRegressionCV(Cs=np.logspace(-3, 0, 7), cv=3, class_weight="balanced",
                                              scoring="roc_auc", max_iter=1000))


def cv_auc(F, y, strat, reps=5):
    aucs, oof = [], []
    for r in range(reps):
        s = np.zeros(len(y))
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=r).split(F, strat):
            m = make_model(); m.fit(F[tr], y[tr]); s[te] = m.decision_function(F[te])
        aucs.append(roc_auc_score(y, s)); oof.append(s)
    return float(np.mean(aucs)), float(np.std(aucs)), np.mean(oof, axis=0)


def boot_ci(y, s, n=2000, seed=0):
    rng = np.random.default_rng(seed); b = []
    for _ in range(n):
        j = rng.choice(len(y), len(y))
        if y[j].min() != y[j].max():
            b.append(roc_auc_score(y[j], s[j]))
    return np.percentile(b, 2.5), np.percentile(b, 97.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="v2")
    ap.add_argument("--perm", type=int, default=30)
    ap.add_argument("--out", default=str(paths.runs_dir() / "reports" / "remodel-lr.md"))
    a = ap.parse_args()
    x, meta, mask, sp = load(a.tag)
    parts = sp.barcode.values; y = sp.y.values.astype(int)
    fac = meta.drop_duplicates("barcode").set_index("barcode").facility.reindex(parts).values
    strat = np.array([f"{a_}_{b}" for a_, b in zip(y, fac)])
    Fc = fusion.clin_matrix(dataset.load_meta(), parts)
    variants = [("semua klip, mentah", False, False), ("semua klip, kurangi derau", True, False),
                ("klip mirip-batuk, mentah", False, True), ("klip mirip-batuk, kurangi derau", True, True)]
    o = ["# Remodel butir 1: statistik log-mel peserta + regresi logistik", "",
         f"{len(y)} peserta, {int(y.sum())} positif. 5-lipat x 5 ulangan, C via CV dalam. CI 95% bootstrap atas peserta pada skor "
         "OOF rata-rata ulangan. Nol = permutasi label dalam klinik.", "",
         "| fitur | klip dipakai (median/peserta) | AUROC (rata-rata ± sd ulangan) | CI 95% | nol permutasi mean / p95 / maks | p-empiris |",
         "|---|---|---|---|---|---|"]
    res = {}
    rng = np.random.default_rng(0)
    for nm, sub, um in variants:
        F, nu = participant_features(x, meta, mask, parts, sub, um)
        m, sd, oof = cv_auc(F, y, strat)
        lo, hi = boot_ci(y, oof)
        null = []
        for k in range(a.perm):
            yp = y.copy()
            for c in np.unique(fac):
                i = np.where(fac == c)[0]; yp[i] = rng.permutation(y[i])
            sp_ = np.array([f"{a_}_{b}" for a_, b in zip(yp, fac)])
            null.append(cv_auc(F, yp, sp_, reps=1)[0])
        null = np.array(null)
        p = (np.sum(null >= m) + 1) / (len(null) + 1)
        res[nm] = (F, oof)
        o.append(f"| {nm} | {np.median(nu):.0f} | {m:.3f} ± {sd:.3f} | [{lo:.3f}, {hi:.3f}] | "
                 f"{null.mean():.3f} / {np.percentile(null, 95):.3f} / {null.max():.3f} | {p:.3f} |")
        print(o[-1], flush=True)
    # fusi dengan gejala klinis (varian terbaik menurut AUROC)
    best = max(res, key=lambda k: roc_auc_score(y, res[k][1]))
    Fb = res[best][0]
    m_c, sd_c, oof_c = cv_auc(Fc, y, strat)
    m_f, sd_f, oof_f = cv_auc(np.hstack([Fc, Fb]), y, strat)
    lo_c, hi_c = boot_ci(y, oof_c); lo_f, hi_f = boot_ci(y, oof_f)
    d = []
    rng = np.random.default_rng(1)
    for _ in range(2000):
        j = rng.choice(len(y), len(y))
        if y[j].min() != y[j].max():
            d.append(roc_auc_score(y[j], oof_f[j]) - roc_auc_score(y[j], oof_c[j]))
    o += ["", f"## Fusi dengan gejala klinis (audio: {best})",
          "| model | AUROC | CI 95% |", "|---|---|---|",
          f"| gejala klinis saja | {m_c:.3f} ± {sd_c:.3f} | [{lo_c:.3f}, {hi_c:.3f}] |",
          f"| gejala + audio (LR gabungan) | {m_f:.3f} ± {sd_f:.3f} | [{lo_f:.3f}, {hi_f:.3f}] |",
          f"| Δ (fusi - gejala) | {m_f - m_c:+.3f} | [{np.percentile(d, 2.5):+.3f}, {np.percentile(d, 97.5):+.3f}] |", "",
          "Catatan: fitur audio statistik kasar dapat menangkap kondisi rekaman/perangkat dan faktor demografis; angka ini BUKAN bukti "
          "penanda akustik TB. Hasil dalam-dataset, satu sumber."]
    pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True); pathlib.Path(a.out).write_text("\n".join(o) + "\n")
    np.savez(paths.runs_dir() / "remodel_lr_oof.npz", barcode=parts, y=y, **{k.replace(" ", "_"): v[1] for k, v in res.items()})
    print("\n".join(o[-8:]))


if __name__ == "__main__":
    main()
