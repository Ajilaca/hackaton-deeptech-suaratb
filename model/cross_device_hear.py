"""E4: HeAR lintas-perangkat pada peserta yang sama (Phone A, B, C): latih perangkat X, uji perangkat Y.

    python -m model.cross_device_hear [--reps 5]

Embedding per peserta = rata-rata klip (maks 6 klip batuk teratas per peserta per perangkat). Lipatan = protokol E0 (per peserta;
peserta uji tidak pernah ada di latih, di perangkat mana pun). Metrik: AUROC terstratifikasi jenis kelamin per ulangan, dirata-rata,
dan AUROC mentah; CI 95% bootstrap atas peserta pada skor rata-rata ulangan. Tanpa nol permutasi (lihat protocol.py untuk itu).
"""
from __future__ import annotations
import argparse
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from golden import paths
from model import protocol as P
from model.clinical_models.clinical import lr

warnings.filterwarnings("ignore")
FILES = {"phonec": "emb_hear.npz", "phonea": "emb_hear_phonea.npz", "phoneb": "emb_hear_phoneb.npz"}


def per_participant(fn: str, dev: str, u: pd.DataFrame) -> np.ndarray:
    z = np.load(paths.features_dir() / fn, allow_pickle=True)
    d = pd.DataFrame(z["emb"]); d["b"] = z["barcode"]; d["dev"] = z["device"]
    d = d[d.dev == dev]
    return d.drop(columns="dev").groupby("b").mean().reindex(u.barcode.values).values


def boot(y, s, g, n=1000, seed=0):
    rng = np.random.default_rng(seed); v = []
    for _ in range(n):
        j = rng.integers(0, len(y), len(y))
        if y[j].sum() in (0, len(j)):
            continue
        try:
            v.append(P.auc_strat(y[j], s[j], g[j]))
        except ZeroDivisionError:
            continue
    return np.percentile(v, 2.5), np.percentile(v, 97.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--out", default=str(paths.runs_dir() / "reports" / "hear-lintas-perangkat.md"))
    a = ap.parse_args()
    u = P.universe(); f = P.make_folds(u, a.reps)
    y = u.y.values.astype(int); sx = u.sex_m.values
    E = {d: per_participant(fn, d, u) for d, fn in FILES.items()}
    devs = list(E)
    o = ["# HeAR lintas-perangkat (peserta yang sama)", "",
         f"Semesta {len(u)} peserta; ada embedding: " + ", ".join(f"{d} {int((~np.isnan(E[d]).any(1)).sum())}" for d in devs) +
         f". Lipatan protokol E0 ({a.reps} ulangan). AUROC strat. = dalam jenis kelamin (berbobot).", "",
         "| latih | uji | n uji / positif | AUROC mentah | AUROC strat. | CI 95% strat. |", "|---|---|---|---|---|---|"]
    for tr_d in devs:
        for te_d in devs:
            ok = ~np.isnan(E[tr_d]).any(1) & ~np.isnan(E[te_d]).any(1)
            idx = np.where(ok)[0]
            sc = np.zeros((a.reps, len(idx)))
            for r in range(a.reps):
                for k in range(5):
                    te = f[r][idx] == k
                    m = lr(); m.fit(E[tr_d][idx][~te], y[idx][~te])
                    sc[r, te] = m.decision_function(E[te_d][idx][te])
            yy, gg = y[idx], sx[idx]
            st = np.mean([P.auc_strat(yy, sc[r], gg) for r in range(a.reps)])
            raw = np.mean([roc_auc_score(yy, sc[r]) for r in range(a.reps)])
            lo, hi = boot(yy, sc.mean(0), gg)
            o.append(f"| {tr_d} | {te_d}{' (dalam-perangkat)' if tr_d == te_d else ''} | {len(idx)} / {int(yy.sum())} | {raw:.3f} | "
                     f"**{st:.3f}** | [{lo:.3f}, {hi:.3f}] |")
            print(o[-1], flush=True)
    o += ["", "Rujukan nol: AUROC strat. di bawah nol permutasi (p95 ≈ 0,57 untuk HeAR, protocol.py) tidak dapat dibedakan dari acak."]
    pth = paths.Path(a.out) if hasattr(paths, "Path") else __import__("pathlib").Path(a.out)
    pth.write_text("\n".join(o) + "\n")
    print("\n".join(o))


if __name__ == "__main__":
    main()
