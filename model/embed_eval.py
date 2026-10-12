"""Evaluasi embedding VGGish vs statistik log-mel (lihat model/embed_test.py)."""
from __future__ import annotations
import pathlib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from golden import paths
from model.data import dataset
from model.tbs_eval import mk, ci, row, participant_table


def agg(E, ids):
    u, inv = np.unique(ids, return_inverse=True)
    return u, np.stack([E[inv == k].mean(0) for k in range(len(u))])


def cv(X, y, strat, cov=None, reps=5):
    a, oof = [], []
    for r in range(reps):
        s = np.zeros(len(y))
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=r).split(X, strat):
            Xtr, Xte = X[tr], X[te]
            if cov is not None:
                g = LinearRegression().fit(cov[tr], Xtr); Xtr, Xte = Xtr - g.predict(cov[tr]), Xte - g.predict(cov[te])
            m = mk(); m.fit(Xtr, y[tr]); s[te] = m.decision_function(Xte)
        a.append(roc_auc_score(y, s)); oof.append(s)
    return float(np.mean(a)), float(np.std(a)), np.mean(oof, axis=0)


def resid(X, cov):
    return X - LinearRegression().fit(cov, X).predict(cov)


def z(X):
    return (X - X.mean(0)) / (X.std(0) + 1e-6)


def main():
    fd = paths.features_dir()
    # --- Zambia
    mz = pd.read_csv(fd / "meta_v2.csv"); Ez = np.load(fd / "emb_zambia.npy")
    uz, Pz = agg(Ez, mz.barcode.values)
    g = dataset.load_meta().set_index("barcode").loc[uz]
    yz = g.y.values; sexz = (g.sex.astype(str).str.lower() == "m").astype(int).values; agez = g.age.values.astype(float)
    covz = np.column_stack([sexz, agez]); facz = g.facility_code.values
    sz = np.array([f"{a}_{b}" for a, b in zip(yz, facz)])
    xz = np.load(fd / "feats_v2.npz")["x_w24"]; actz = np.load(fd / "active_v2.npz")["active"]
    _, Lz = participant_table(xz, actz, mz.barcode.values)
    # --- TBscreen (pasif)
    mt = pd.read_csv(fd / "meta_tbs.csv"); Et = np.load(fd / "emb_tbs.npy"); zt = np.load(fd / "feats_tbs.npz")
    sel = (mt.kind == "passive").to_numpy()
    ut, Pt = agg(Et[sel], mt.barcode.values[sel])
    _, Lt = participant_table(zt["x_w24"][sel], zt["active"][sel], mt.barcode.values[sel])
    gt = mt[sel].drop_duplicates("barcode").set_index("barcode").loc[ut]
    yt = gt.y.values; sext = gt.sex_m.values; aget = gt.age.astype(float).fillna(gt.age.median()).values
    covt = np.column_stack([sext, aget]); st = np.array([f"{a}_{b}" for a, b in zip(yt, gt.cohort)])
    o = ["# Langkah 3 (uji murah): embedding VGGish pralatih vs statistik log-mel", "",
         "Guru: VGGish (128-d, bobot TBscreen_code.zip). Peserta = rata-rata embedding antar klip. LR (C via CV dalam). "
         "'residu' = regresi sex+usia dikeluarkan (dipasang pada data latih tiap lipatan). Zambia: CV 5x5 per peserta; "
         "n/TB = peserta / positif.", "",
         "## Dalam-dataset", "| fitur | dataset | n / TB | AUROC | CI 95% | catatan |", "|---|---|---|---|---|---|"]

    def add(nm, ds, y, s, note=""):
        r = row(nm, y, s, note); o.append(r.replace(f"| {nm} |", f"| {nm} | {ds} |", 1))
    for nm, X in (("log-mel mean/std (80)", Lz), ("VGGish (128)", Pz)):
        add(nm, "Zambia", yz, cv(X, yz, sz)[2])
        a, sd, s = cv(X, yz, sz, cov=covz)
        add(nm + ", residu sex+usia", "Zambia", yz, s, f"±{sd:.3f}")
    add("sex+usia saja (LR)", "Zambia", yz, cv(covz.astype(float), yz, sz)[2])
    add("VGGish → jenis kelamin", "Zambia", sexz, cv(Pz, sexz, np.array([f"{a}_{b}" for a, b in zip(sexz, facz)]))[2], "kontrol perancu")
    for nm, X in (("log-mel mean/std (80)", Lt), ("VGGish (128)", Pt)):
        add(nm, "TBscreen", yt, cv(X, yt, st)[2], "cohort ≈ label!")
        add(nm + ", residu sex+usia", "TBscreen", yt, cv(X, yt, st, cov=covt)[2])
    # --- lintas-dataset
    o += ["", "## Lintas-dataset (z-norm per dataset; residu sex+usia dipasang per dataset tanpa label)",
          "| arah | fitur | n uji / TB | AUROC | CI 95% | catatan |", "|---|---|---|---|---|---|"]

    def xfer(Xa, ya, Xb, yb, name, feat, res):
        if res:
            Xa, Xb = resid(Xa, covA[0]), resid(Xb, covA[1])
        m = mk(); m.fit(z(Xa), ya); s = m.decision_function(z(Xb))
        r = row(name, yb, s, "residu sex+usia" if res else "")
        o.append(r.replace(f"| {name} |", f"| {name} | {feat} |", 1))
    for feat, (XZ, XT) in (("log-mel", (Lz, Lt)), ("VGGish", (Pz, Pt))):
        for res in (False, True):
            covA = (covz, covt); xfer(XZ, yz, XT, yt, "Zambia → TBscreen", feat, res)
            covA = (covt, covz); xfer(XT, yt, XZ, yz, "TBscreen → Zambia", feat, res)
    o += ["", "## Keputusan distilasi", "- Diisi setelah angka dibaca (lihat status-fase.md)."]
    out = paths.runs_dir() / "reports" / "embedding-vggish.md"
    out.parent.mkdir(parents=True, exist_ok=True); out.write_text("\n".join(o) + "\n")
    print("\n".join(o))


if __name__ == "__main__":
    main()
