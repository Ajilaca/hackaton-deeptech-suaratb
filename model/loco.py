"""E4: leave-one-clinic-out terbatas (Kan <-> Cha+Chai) untuk kandidat embedding.

    python -m model.loco [--perm 2000]

Latih LR (C = 0,03 tetap, praregistrasi permtest.py) di satu klinik, uji di klinik lain TANPA melihat klinik uji. Metrik: AUROC audio
terstratifikasi jenis kelamin di klinik uji. Nol: label uji diacak dalam jenis kelamin (model beku, jadi permutasi tepat), p95/p99 dari
nol dan p-empiris. CI bootstrap atas peserta uji. Klinik Cha+Chai hanya 11 positif: CI sangat lebar, interpretasi hati-hati.
"""
from __future__ import annotations
import argparse
import warnings
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from golden import paths
from model import protocol as P
from model.embed_pretrained import participant_matrix

warnings.filterwarnings("ignore")
CANDS = ["hear", "operaCT", "operaCE"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--perm", type=int, default=2000)
    ap.add_argument("--c", type=float, default=0.03)
    a = ap.parse_args()
    u = P.universe()
    Fs = {c: participant_matrix(c, u) for c in CANDS}
    ok = np.all([~np.isnan(Fs[c]).any(1) for c in CANDS], axis=0)
    u = u[ok].reset_index(drop=True)
    y = u.y.values.astype(int); sx = u.sex_m.values; fac = u.fac.values
    rng = np.random.default_rng(0)
    o = ["# Leave-one-clinic-out (terbatas)", "",
         f"C = {a.c}. Cha = Cha + Chai (Chai hanya 1 positif). AUROC strat. = dalam jenis kelamin. Nol: label uji diacak dalam jenis kelamin "
         f"({a.perm} permutasi, model beku). CI 95% bootstrap atas peserta uji.", "",
         "| latih | uji | n latih / positif | n uji / positif | kandidat | AUROC strat. | nol p95 / p99 | p-empiris | CI 95% | AUROC mentah |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    from sklearn.metrics import roc_auc_score
    for tr_c, te_c in (("Kan", "Cha"), ("Cha", "Kan")):
        tr, te = fac == tr_c, fac == te_c
        for c in CANDS:
            m = make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler(),
                              LogisticRegression(C=a.c, class_weight="balanced", max_iter=500))
            m.fit(Fs[c][ok][tr], y[tr]); s = m.decision_function(Fs[c][ok][te])
            yt, gt = y[te], sx[te]
            obs = P.auc_strat(yt, s, gt)
            null = []
            for _ in range(a.perm):
                yp = yt.copy()
                for g in (0, 1):
                    i = np.where(gt == g)[0]; yp[i] = rng.permutation(yt[i])
                try:
                    null.append(P.auc_strat(yp, s, gt))
                except ZeroDivisionError:
                    pass
            null = np.array(null)
            bs = []
            for _ in range(1000):
                j = rng.integers(0, len(yt), len(yt))
                if yt[j].sum() in (0, len(j)):
                    continue
                try:
                    bs.append(P.auc_strat(yt[j], s[j], gt[j]))
                except ZeroDivisionError:
                    pass
            o.append(f"| {tr_c} | {te_c} | {int(tr.sum())} / {int(y[tr].sum())} | {int(te.sum())} / {int(yt.sum())} | {c} | **{obs:.3f}** | "
                     f"{np.percentile(null, 95):.3f} / {np.percentile(null, 99):.3f} | {(1 + np.sum(null >= obs)) / (len(null) + 1):.4f} | "
                     f"[{np.percentile(bs, 2.5):.3f}, {np.percentile(bs, 97.5):.3f}] | {roc_auc_score(yt, s):.3f} |")
    out = paths.runs_dir() / "reports" / "loco.md"
    out.write_text("\n".join(o) + "\n"); print("\n".join(o))


if __name__ == "__main__":
    main()
