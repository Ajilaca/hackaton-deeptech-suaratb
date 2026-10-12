"""Fusi skor batuk (CNN, out-of-fold) + gejala klinis (regresi logistik) dan laporan akhir.

    python -m model.fusion --tag v2 --runs mil_v2.npz [--perm mil_perm.npz] [--out $SUARATB_DATA/runs/reports/pelatihan-serius.md]

Fitur klinis yang DIPAKAI: age, sex, coughdur, cough_productive, haemoptysis, chestpain, shortbreath, fever,
ngtsweats, weightloss, hiv_status, prev_tb, tobacco_use, bmi. TIDAK dipakai: xpert/culture/smear/lam, abn_*/tb_*
(prediksi AI X-ray), type_tb, ground_truth_tb (kebocoran label / bukan masukan perangkat).
"""
from __future__ import annotations
import argparse
import pathlib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from golden import paths
from model.data import dataset

CLIN = ["age", "sex", "coughdur", "cough_productive", "haemoptysis", "chestpain", "shortbreath", "fever",
        "ngtsweats", "weightloss", "hiv_status", "prev_tb", "tobacco_use", "bmi"]
DUR = {"no cough": 0, "<1wk": 1, "1-2wks": 2, "3-4wks": 3, ">4weeks": 4}


def clin_matrix(meta: pd.DataFrame, barcodes) -> np.ndarray:
    d = meta.set_index("barcode").loc[list(barcodes)]
    cols = []
    for c in CLIN:
        s = d[c]
        if c == "coughdur":
            v = s.map(DUR).astype(float)
        elif c == "sex":
            v = (s.astype(str).str.lower() == "m").astype(float)
        elif c == "tobacco_use":
            for lv in ("current", "stopped"):
                cols.append((s.astype(str).str.lower() == lv).astype(float).values)
            continue
        elif pd.api.types.is_numeric_dtype(s):
            v = s.astype(float)
        else:
            v = s.astype(str).str.lower().map({"yes": 1.0, "pos": 1.0, "no": 0.0, "neg": 0.0})
        cols.append(v.values)
    return np.column_stack(cols)


def cv_auc(F, y, folds, C=0.3):
    s = np.zeros(len(y))
    for k in np.unique(folds):
        te = folds == k
        m = make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler(),
                          LogisticRegression(C=C, class_weight="balanced", max_iter=1000))
        m.fit(F[~te], y[~te]); s[te] = m.decision_function(F[te])
    return s


def boot(y, s, ref=None, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y)); b = []
    for _ in range(n):
        j = rng.choice(idx, len(idx))
        if y[j].min() == y[j].max():
            continue
        b.append(roc_auc_score(y[j], s[j]) - (roc_auc_score(y[j], ref[j]) if ref is not None else 0))
    return np.percentile(b, 2.5), np.percentile(b, 97.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="v2")
    ap.add_argument("--runs", default="mil_v2.npz")
    ap.add_argument("--perm", default=None)
    ap.add_argument("--out", default=str(paths.runs_dir() / "reports" / "pelatihan-serius.md"))
    a = ap.parse_args()
    R = np.load(paths.runs_dir() / a.runs, allow_pickle=True)
    parts = R["barcode"]; devs = [str(d) for d in R["devs"]]
    sp = pd.read_csv(paths.processed_dir() / f"splits_{a.tag}.csv").set_index("barcode").loc[parts]
    y = sp.y.values.astype(int); folds = sp["fold"].values
    meta = dataset.load_meta()
    fac = meta.set_index("barcode").loc[parts].facility_code.values
    reps = sorted({k.split("_")[1] for k in R.files if k.startswith("run0_rep")})
    cough = {}
    for d in devs:
        arr = np.stack([R[f"run0_{r}_{d}"] for r in reps])
        cough[d] = np.nanmean(arr, axis=0)
    F = clin_matrix(meta, parts)
    sym = cv_auc(F, y, folds)
    o = ["# Pelatihan lebih serius: MIL tingkat peserta + fusi gejala klinis", "",
         f"`python -m model.scratch.mil_train` (CV {len(reps)}x5-lipat per peserta, ensemble seed, early stopping pada validasi dalam, "
         f"kantong {4} batuk dengan rata-rata logit seperti perangkat; augmentasi gain/geser/SpecAugment) lalu `python -m model.fusion`.",
         f"{len(y)} peserta, {int(y.sum())} positif TB (`ground_truth_tb`). CI 95% bootstrap atas peserta (2.000x).", ""]
    rows = []
    a_sym = roc_auc_score(y, sym); lo, hi = boot(y, sym)
    rows.append(("Gejala klinis saja (LR)", "-", a_sym, lo, hi, ""))
    for d in devs:
        s = cough[d]; m = ~np.isnan(s)
        if len(set(y[m])) < 2:
            continue
        au = roc_auc_score(y[m], s[m]); lo, hi = boot(y[m], s[m])
        rows.append(("Batuk saja (CNN, skor out-of-fold)", d, au, lo, hi, f"n={m.sum()}"))
        Ff = np.column_stack([F[m], s[m]])
        fu = cv_auc(Ff, y[m], folds[m])
        ref = cv_auc(F[m], y[m], folds[m])
        lo, hi = boot(y[m], fu, ref=ref)
        rows.append(("Fusi gejala + batuk (LR)", d, roc_auc_score(y[m], fu), *boot(y[m], fu),
                     f"Δ vs gejala saja {roc_auc_score(y[m], fu) - roc_auc_score(y[m], ref):+.3f} [{lo:+.3f}, {hi:+.3f}]"))
    o += ["| model | perangkat batuk | AUROC | CI 95% | catatan |", "|---|---|---|---|---|"]
    for nm, d, au, lo, hi, nt in rows:
        o.append(f"| {nm} | {d} | {au:.3f} | [{lo:.3f}, {hi:.3f}] | {nt} |")
    s = cough["all"]
    o += ["", "## Per klinik (batuk saja, semua perangkat)", "| klinik | peserta | positif | AUROC |", "|---|---|---|---|"]
    for f_ in sorted(set(fac)):
        m = fac == f_
        au = roc_auc_score(y[m], s[m]) if len(set(y[m])) == 2 else float("nan")
        o.append(f"| {f_} | {int(m.sum())} | {int(y[m].sum())} | {au:.3f} |")
    if a.perm:
        P = np.load(paths.runs_dir() / a.perm, allow_pickle=True)
        null = []
        for k in P.files:
            if k.endswith("_all") and k.startswith("run"):
                run = k.split("_")[0]
                yp = P[f"{run}_y"]; sc = P[k]; m = ~np.isnan(sc)
                null.append(roc_auc_score(yp[m], sc[m]))
        obs = roc_auc_score(y, cough["all"])
        null = np.array(null)
        o += ["", "## Uji permutasi (label diacak dalam klinik, seluruh pipeline pelatihan diulang)",
              f"- {len(null)} permutasi: AUROC nol rata-rata {null.mean():.3f}, p95 {np.percentile(null, 95):.3f}, maks {null.max():.3f}",
              f"- AUROC teramati (batuk, semua perangkat): {obs:.3f}; p-empiris ≥ teramati: {(np.sum(null >= obs) + 1) / (len(null) + 1):.3f}"]
    o += ["", "## Target CLAUDE.md", f"- Batuk saja ≥ 0,70: {'TERCAPAI' if roc_auc_score(y, cough['all']) >= 0.70 else 'belum'}",
          f"- Fusi klinis ≥ 0,80: {'TERCAPAI' if max(r[2] for r in rows if r[0].startswith('Fusi')) >= 0.80 else 'belum'}",
          "", "Catatan: 'gejala saja' bukan hasil perangkat; ia ada untuk melihat nilai tambah audio. Hasil dalam-dataset; tidak ada uji lintas-dataset."]
    pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True); pathlib.Path(a.out).write_text("\n".join(o) + "\n")
    print("\n".join(o))


if __name__ == "__main__":
    main()
