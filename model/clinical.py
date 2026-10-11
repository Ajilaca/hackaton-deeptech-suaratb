"""Langkah 2: perbaiki fitur klinis dan fusi dengan audio, evaluasi bersarang tanpa kebocoran.

    python -m model.clinical --tag v2 [--reps 5] [--out $SUARATB_DATA/runs/reports/klinis-fusi.md]

Fitur klinis memakai HANYA gejala/anamnesis/antropometri. TIDAK dipakai: xpert/culture/smear/lam/type_tb, abn_*/tb_*
(prediksi AI X-ray), ground_truth_tb (kebocoran label), facility_code (kode klinik: jalan pintas rekrutmen, dilaporkan
terpisah). Audio masuk fusi lewat skor LR audio yang dihitung DI DALAM tiap lipatan luar (skor latih = OOF lipatan
dalam), jadi tidak ada kebocoran. Varian 'resid' meregresikan jenis kelamin+usia dari fitur audio (dipasang pada data latih).
"""
from __future__ import annotations
import argparse
import pathlib
import warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression, LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from golden import paths
from model import dataset, fusion
from model.remodel import load, participant_features

warnings.filterwarnings("ignore")


def clin_extended(meta: pd.DataFrame, parts) -> tuple[np.ndarray, list[str]]:
    d = meta.set_index("barcode").loc[list(parts)]
    base = fusion.clin_matrix(meta, parts)
    yes = lambda c: (d[c].astype(str).str.lower() == "yes").astype(float).values  # noqa: E731
    sex = (d.sex.astype(str).str.lower() == "m").astype(float).values
    age = d.age.astype(float).values
    n_sym = yes("fever") + yes("ngtsweats") + yes("weightloss") + (d.coughdur.map(fusion.DUR).astype(float).values >= 3)
    ext = np.column_stack([
        d.body_wt.astype(float).values, d.height.astype(float).values,
        n_sym, n_sym >= 2, ((age - 35) / 10) ** 2, sex * (age - 35) / 10,
        sex * yes("weightloss"), sex * (d.hiv_status.astype(str).str.lower() == "pos").astype(float).values,
    ])
    names = ["body_wt", "height", "n_sym", "n_sym>=2", "age2", "sex*age", "sex*weightloss", "sex*hiv"]
    return np.column_stack([base, ext]), names


def lr(C=None):
    if C is None:
        return make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler(),
                             LogisticRegressionCV(Cs=np.logspace(-3, 0, 7), cv=3, class_weight="balanced",
                                                  scoring="roc_auc", max_iter=2000))
    return make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler(),
                         LogisticRegression(C=C, class_weight="balanced", max_iter=2000))


def gbm():
    return make_pipeline(SimpleImputer(strategy="median"),
                         HistGradientBoostingClassifier(max_depth=2, learning_rate=0.05, max_iter=150,
                                                        min_samples_leaf=10, l2_regularization=1.0,
                                                        class_weight="balanced", random_state=0))


def audio_score(Atr, ytr, Ate, cov_tr, cov_te, resid: bool, strat_tr):
    """Skor audio: (opsional residualisasi) -> LR. Skor latih = OOF lipatan dalam; skor uji = model penuh."""
    if resid:
        reg = LinearRegression().fit(cov_tr, Atr)
        Atr, Ate = Atr - reg.predict(cov_tr), Ate - reg.predict(cov_te)
    s_tr = np.zeros(len(ytr))
    for a, b in StratifiedKFold(5, shuffle=True, random_state=1).split(Atr, strat_tr):
        m = lr(); m.fit(Atr[a], ytr[a]); s_tr[b] = m.decision_function(Atr[b])
    m = lr(); m.fit(Atr, ytr)
    return s_tr, m.decision_function(Ate)


def run(F_by, A, cov, y, strat, reps):
    """F_by: dict nama -> matriks klinis. Kembalikan {varian: (aucs, oof_mean)}."""
    names = ["klinis14", "klinis_ext", "klinis_ext_gbm", "audio", "audio_resid",
             "fusi14+audio", "fusi_ext+audio", "fusi_ext+audio_resid"]
    res = {n: [] for n in names}; oof = {n: [] for n in names}
    for r in range(reps):
        s = {n: np.zeros(len(y)) for n in names}
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=r).split(A, strat):
            F14, Fx = F_by["base"], F_by["ext"]
            for n, F, mk in (("klinis14", F14, lr), ("klinis_ext", Fx, lr), ("klinis_ext_gbm", Fx, gbm)):
                m = mk(); m.fit(F[tr], y[tr])
                s[n][te] = m.decision_function(F[te]) if n != "klinis_ext_gbm" else m.predict_proba(F[te])[:, 1]
            for tag, resid in (("audio", False), ("audio_resid", True)):
                a_tr, a_te = audio_score(A[tr], y[tr], A[te], cov[tr], cov[te], resid, strat[tr])
                s[tag][te] = a_te
                for fn, F in (("fusi14+", F14), ("fusi_ext+", Fx)):
                    k = fn + tag
                    if k not in s:
                        continue
                    m = lr(C=0.3); m.fit(np.column_stack([F[tr], a_tr]), y[tr])
                    s[k][te] = m.decision_function(np.column_stack([F[te], a_te]))
        for n in names:
            res[n].append(roc_auc_score(y, s[n])); oof[n].append(s[n])
    return res, {n: np.mean(v, axis=0) for n, v in oof.items()}


def boot(y, a, b=None, n=2000, seed=0):
    rng = np.random.default_rng(seed); v = []
    for _ in range(n):
        j = rng.choice(len(y), len(y))
        if y[j].min() != y[j].max():
            v.append(roc_auc_score(y[j], a[j]) - (roc_auc_score(y[j], b[j]) if b is not None else 0))
    return np.percentile(v, 2.5), np.percentile(v, 97.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="v2")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--out", default=str(paths.runs_dir() / "reports" / "klinis-fusi.md"))
    a = ap.parse_args()
    x, meta, mask, sp = load(a.tag)
    parts = sp.barcode.values; y = sp.y.values.astype(int)
    fac = meta.drop_duplicates("barcode").set_index("barcode").facility.reindex(parts).values
    strat = np.array([f"{a_}_{b}" for a_, b in zip(y, fac)])
    gm = dataset.load_meta()
    Fb = fusion.clin_matrix(gm, parts); Fx, ext_names = clin_extended(gm, parts)
    d = gm.set_index("barcode").loc[list(parts)]
    cov = np.column_stack([(d.sex.astype(str).str.lower() == "m").astype(float), d.age.astype(float)])
    A, _ = participant_features(x, meta, mask, parts, sub=False, use_mask=False)
    res, oof = run({"base": Fb, "ext": Fx}, A, cov, y, strat, a.reps)
    lab = {"klinis14": "Gejala klinis (14 fitur, LR) [pembanding]", "klinis_ext": f"Klinis diperluas (+{', '.join(ext_names)}) LR",
           "klinis_ext_gbm": "Klinis diperluas, GBM dangkal (depth 2)", "audio": "Audio saja: statistik log-mel (LR)",
           "audio_resid": "Audio saja, residu jenis kelamin+usia (LR)", "fusi14+audio": "Fusi: klinis14 + audio",
           "fusi_ext+audio": "Fusi: klinis diperluas + audio", "fusi_ext+audio_resid": "Fusi: klinis diperluas + audio residu"}
    o = ["# Langkah 2: fitur klinis diperluas dan fusi (evaluasi bersarang)", "",
         f"{len(y)} peserta, {int(y.sum())} positif. {a.reps} ulangan x 5-lipat, stratifikasi label x klinik. Audio = statistik log-mel "
         "peserta (mean+std per band, semua klip). Skor audio dihitung di dalam lipatan (tanpa kebocoran). "
         "Tidak dipakai: xpert/culture/smear/lam, prediksi AI X-ray, ground truth, kode klinik.", "",
         "| model | AUROC (rata-rata ± sd ulangan) | CI 95% (OOF rata-rata) | Δ vs klinis14 [CI 95%] |", "|---|---|---|---|"]
    for n in lab:
        m_, s_ = np.mean(res[n]), np.std(res[n])
        lo, hi = boot(y, oof[n])
        dl = "" if n == "klinis14" else "{:+.3f} [{:+.3f}, {:+.3f}]".format(
            np.mean(res[n]) - np.mean(res["klinis14"]), *boot(y, oof[n], oof["klinis14"]))
        o.append(f"| {lab[n]} | {m_:.3f} ± {s_:.3f} | [{lo:.3f}, {hi:.3f}] | {dl} |")
    # klinis dengan/ tanpa kode klinik: seberapa besar jalan pintas rekrutmen
    fac_oh = pd.get_dummies(pd.Series(fac)).values.astype(float)
    r2, _ = run({"base": np.column_stack([Fb, fac_oh]), "ext": np.column_stack([Fx, fac_oh])}, A[:, :2], cov, y, strat, 1)
    o += ["", f"Pemeriksaan jalan pintas: klinis14 + kode klinik = {np.mean(r2['klinis14']):.3f} (tanpa: {np.mean(res['klinis14']):.3f}); "
          "kode klinik TIDAK dipakai di model utama karena bukan masukan perangkat dan tidak umum ke lokasi lain.",
          "", "## Target CLAUDE.md",
          f"- Fusi ≥ 0,80: {'TERCAPAI' if max(np.mean(res[k]) for k in res if k.startswith('fusi')) >= 0.80 else 'belum'} "
          f"(terbaik {max(np.mean(res[k]) for k in res if k.startswith('fusi')):.3f})",
          "- Hasil dalam-dataset, satu sumber; CI lebar (58 positif). Lihat remodel.md untuk perancu jenis kelamin."]
    pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True); pathlib.Path(a.out).write_text("\n".join(o) + "\n")
    np.savez(paths.runs_dir() / "klinis_fusi_oof.npz", barcode=parts, y=y, **{k.replace("+", "_"): v for k, v in oof.items()})
    print("\n".join(o))


if __name__ == "__main__":
    main()
