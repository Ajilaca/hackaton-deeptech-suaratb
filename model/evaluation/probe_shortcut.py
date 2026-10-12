"""E4-probe: apakah skor embedding hanya jalan pintas (klinik, jenis kelamin, perangkat)?

    python -m model.evaluation.probe_shortcut --model hear [--reps 5]

(1) AUROC skor OOF dalam strata: jenis kelamin, klinik, klinik x jenis kelamin, per perangkat sumber klip.
(2) Seberapa baik embedding memprediksi jenis kelamin, klinik (Kan vs lain), perangkat, usia (AUROC / korelasi, CV per peserta).
Memakai lipatan dan skor protokol E0 (model.evaluation.protocol); tidak ada pemilihan varian.
"""
from __future__ import annotations
import argparse
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from golden import paths
from model.evaluation import protocol as P
from model.clinical_models.clinical import lr
from model.embed_pretrained import participant_matrix

warnings.filterwarnings("ignore")


def cv_auc(F, t, reps=3):
    out = []
    for r in range(reps):
        s = np.zeros(len(t))
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=r).split(F, t):
            m = lr(); m.fit(F[tr], t[tr]); s[te] = m.decision_function(F[te])
        out.append(roc_auc_score(t, s))
    return float(np.mean(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="hear")
    ap.add_argument("--reps", type=int, default=5)
    a = ap.parse_args()
    u = P.universe(); f = P.make_folds(u, a.reps)
    F = participant_matrix(a.model, u); ok = ~np.isnan(F).any(1)
    u = u[ok].reset_index(drop=True); F = F[ok]; f = P.make_folds(u, a.reps)
    y = u.y.values.astype(int); sx = u.sex_m.values
    age = u.age.astype(float).fillna(u.age.astype(float).median()).values
    cov = np.column_stack([sx, age])
    s, _ = P.oof_scores(F, y, u.strat.values, f, cov, None)
    sm = s.mean(0)
    fac = u.fac.values
    z = np.load(paths.features_dir() / f"emb_{a.model}.npz", allow_pickle=True)
    dev = pd.Series(z["device"]).groupby(z["barcode"]).agg(lambda v: v.iloc[0]).reindex(u.barcode.values).values
    o = [f"# Pemeriksaan jalan pintas: {a.model}", "", f"{len(u)} peserta, {int(y.sum())} positif; {a.reps} ulangan CV protokol E0.", "",
         "## 1. AUROC skor audio dalam strata", "| strata | n | positif | AUROC |", "|---|---|---|---|",
         f"| semua (mentah) | {len(y)} | {int(y.sum())} | {roc_auc_score(y, sm):.3f} |"]
    for nm, g in (("jenis kelamin (berbobot)", sx), ("klinik (berbobot)", fac), ("klinik x jenis kelamin (berbobot)", np.char.add(fac.astype(str), sx.astype(str)))):
        o.append(f"| {nm} | {len(y)} | {int(y.sum())} | {P.auc_strat(y, sm, g):.3f} |")
    for g in sorted(set(fac)):
        m = fac == g
        if y[m].sum() >= 3:
            o.append(f"| klinik {g} saja | {int(m.sum())} | {int(y[m].sum())} | {roc_auc_score(y[m], sm[m]):.3f} |")
    for g in (0, 1):
        m = sx == g
        o.append(f"| {'laki-laki' if g else 'perempuan'} saja | {int(m.sum())} | {int(y[m].sum())} | {roc_auc_score(y[m], sm[m]):.3f} |")
    for d in sorted(set(dev)):
        m = dev == d
        if y[m].sum() >= 3 and (1 - y[m]).sum() >= 3:
            o.append(f"| perangkat sumber klip {d} | {int(m.sum())} | {int(y[m].sum())} | {roc_auc_score(y[m], sm[m]):.3f} |")
    o += ["", "## 2. Embedding memprediksi apa? (LR, CV per peserta, 3 ulangan)", "| target | AUROC |", "|---|---|",
          f"| jenis kelamin | {cv_auc(F, sx):.3f} |", f"| klinik Kan vs lain | {cv_auc(F, (fac == 'Kan').astype(int)):.3f} |",
          f"| usia >= median | {cv_auc(F, (age >= np.median(age)).astype(int)):.3f} |"]
    pr = pd.Series(dev).value_counts().to_dict()
    o += ["", f"Perangkat sumber klip per peserta: {pr}. Keluaran skor audio memakai kovariat (jenis kelamin, usia) hanya untuk lipatan; "
          "tidak ada koreksi klinik."]
    out = paths.runs_dir() / "reports" / f"jalan-pintas-{a.model}.md"
    out.write_text("\n".join(o) + "\n"); print("\n".join(o))


if __name__ == "__main__":
    main()
