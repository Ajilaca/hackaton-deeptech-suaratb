"""Klinis pada SEMUA peserta berlabel (n=1.828) dan manfaat tambahan audio pada 360 peserta berlaudio.

    python -m model.clinical_full --tag v2 [--out $SUARATB_DATA/runs/reports/klinis-penuh.md]

Pertanyaan: (1) seberapa baik gejala klinis memprediksi TB bila memakai semua peserta (tanpa audio)? (2) apakah 360 peserta
berlaudio mewakili 1.828? (3) apakah audio menambah di atas skor klinis? Tanpa kebocoran: skor klinis untuk 360 peserta
berlaudio dihitung dari model yang dilatih HANYA pada peserta tanpa audio (pasangan latih-uji terpisah), skor audio dihitung
di dalam lipatan. Fitur klinis = fusion.clin_matrix / clinical.clin_extended (tanpa xpert/culture/smear/AI X-ray).
"""
from __future__ import annotations
import argparse
import pathlib
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from golden import paths
from model import dataset, fusion
from model.clinical import clin_extended, lr, audio_score, boot
from model.remodel import load, participant_features

warnings.filterwarnings("ignore")


def cv_scores(F, y, strat, reps=5):
    s = np.zeros((reps, len(y)))
    for r in range(reps):
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=r).split(F, strat):
            m = lr(); m.fit(F[tr], y[tr]); s[r, te] = m.decision_function(F[te])
    return s


def auc_ci(y, s):
    lo, hi = boot(y, s)
    return roc_auc_score(y, s), lo, hi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="v2")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--out", default=str(paths.runs_dir() / "reports" / "klinis-penuh.md"))
    a = ap.parse_args()
    gm = dataset.load_meta().drop_duplicates("barcode").reset_index(drop=True)
    gm = gm[gm.ground_truth_tb.isin(["pos", "neg"])].reset_index(drop=True)
    allp = gm.barcode.values; y = gm.y.values.astype(int)
    F14 = fusion.clin_matrix(gm, allp); Fx, ext_names = clin_extended(gm, allp)
    strat = np.array([f"{a_}_{b}" for a_, b in zip(y, gm.facility_code.astype(str))])
    x, meta, mask, sp = load(a.tag)
    aud = set(sp.barcode.values)
    has_a = np.array([b in aud for b in allp])
    o = ["# Klinis pada semua peserta berlabel dan manfaat tambahan audio", "",
         f"Peserta berlabel: {len(y)} ({int(y.sum())} positif). Berlaudio (bisa dipakai): {int(has_a.sum())} "
         f"({int(y[has_a].sum())} positif). Tanpa audio: {int((~has_a).sum())} ({int(y[~has_a].sum())} positif).", ""]
    # ---- 1. perbandingan peserta berlaudio vs tanpa audio ----------------------------------------
    sex = (gm.sex.astype(str).str.lower() == "m").astype(float).values
    age = gm.age.astype(float).values
    o += ["## 1. Apakah peserta berlaudio mewakili semuanya?", "",
          "| kelompok | n | prevalensi TB | % laki-laki | usia median |", "|---|---|---|---|---|"]
    for nm, sel in (("berlaudio", has_a), ("tanpa audio", ~has_a), ("semua", np.ones(len(y), bool))):
        o.append(f"| {nm} | {int(sel.sum())} | {100 * y[sel].mean():.1f}% | {100 * np.nanmean(sex[sel]):.0f}% | "
                 f"{np.nanmedian(age[sel]):.0f} |")
    fac = gm.facility_code.astype(str).values
    o += ["", "| klinik | n berlaudio | n tanpa audio | prevalensi berlaudio | prevalensi tanpa audio |", "|---|---|---|---|---|"]
    for f in sorted(set(fac)):
        s1, s0 = (fac == f) & has_a, (fac == f) & ~has_a
        pv = lambda s: f"{100 * y[s].mean():.1f}%" if s.sum() else "-"  # noqa: E731
        o.append(f"| {f} | {int(s1.sum())} | {int(s0.sum())} | {pv(s1)} | {pv(s0)} |")
    # ---- 2. klinis pada semua peserta ---------------------------------------------------------
    o += ["", "## 2. Gejala klinis saja pada semua peserta (CV 5x%d, stratifikasi label x klinik)" % a.reps, "",
          "| model | n / positif | AUROC | CI 95% |", "|---|---|---|---|"]
    S = {}
    for nm, F in (("klinis 14 fitur (LR)", F14), (f"klinis diperluas (+{len(ext_names)} fitur) (LR)", Fx)):
        s = cv_scores(F, y, strat, a.reps)
        m = s.mean(0)
        S[nm] = m
        aucs = [roc_auc_score(y, s[r]) for r in range(a.reps)]
        lo, hi = boot(y, m)
        o.append(f"| {nm} | {len(y)} / {int(y.sum())} | {np.mean(aucs):.3f} ± {np.std(aucs):.3f} | [{lo:.3f}, {hi:.3f}] |")
    # ---- 3. skor klinis untuk 360 berlaudio dari model yang dilatih pada yang tanpa audio ----
    ia, ina = np.where(has_a)[0], np.where(~has_a)[0]
    m14 = lr(); m14.fit(F14[ina], y[ina]); c14 = m14.decision_function(F14[ia])
    mx = lr(); mx.fit(Fx[ina], y[ina]); cx = mx.decision_function(Fx[ia])
    ya = y[ia]
    # pembanding: klinis dilatih pada 360 itu sendiri (CV)
    s360 = cv_scores(F14[ia], ya, strat[ia], a.reps).mean(0)
    o += ["", f"## 3. Klinis diuji pada {len(ia)} peserta berlaudio", "",
          "| model | latih pada | AUROC | CI 95% |", "|---|---|---|---|"]
    for nm, tr, sc in (("klinis 14 (CV dalam 360)", f"{len(ia)} (CV)", s360),
                       ("klinis 14", f"{len(ina)} tanpa audio", c14),
                       ("klinis diperluas", f"{len(ina)} tanpa audio", cx)):
        au, lo, hi = auc_ci(ya, sc)
        o.append(f"| {nm} | {tr} | {au:.3f} | [{lo:.3f}, {hi:.3f}] |")
    # ---- 4. audio menambah di atas klinis? ---------------------------------------------------------
    parts = sp.barcode.values
    ordr = {b: i for i, b in enumerate(allp[ia])}
    idx = np.array([ordr[b] for b in parts])
    A, _ = participant_features(x, meta, mask, parts, sub=False, use_mask=False)
    cov = np.column_stack([sex[ia][idx], age[ia][idx]])
    cc = cx[idx]; yy = ya[idx]; st = strat[ia][idx]
    reps = a.reps
    aud_s = np.zeros((reps, len(yy))); fus = np.zeros((reps, len(yy)))
    for r in range(reps):
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=r).split(A, st):
            a_tr, a_te = audio_score(A[tr], yy[tr], A[te], cov[tr], cov[te], False, st[tr])
            aud_s[r, te] = a_te
            m = lr(C=0.3); m.fit(np.column_stack([cc[tr], a_tr]), yy[tr])
            fus[r, te] = m.decision_function(np.column_stack([cc[te], a_te]))
    am, fm = aud_s.mean(0), fus.mean(0)
    o += ["", "## 4. Apakah audio menambah di atas skor klinis? (peserta berlaudio, skor klinis dari model tanpa audio)", "",
          "| model | AUROC | CI 95% | Δ vs klinis |", "|---|---|---|---|"]
    base = roc_auc_score(yy, cc); blo, bhi = boot(yy, cc)
    o.append(f"| klinis diperluas (dilatih pada tanpa-audio) | {base:.3f} | [{blo:.3f}, {bhi:.3f}] | - |")
    au, lo, hi = auc_ci(yy, am)
    o.append(f"| audio saja (statistik log-mel, LR) | {au:.3f} | [{lo:.3f}, {hi:.3f}] | - |")
    au, lo, hi = auc_ci(yy, fm)
    dlo, dhi = boot(yy, fm, cc)
    o.append(f"| klinis + audio (fusi) | {au:.3f} | [{lo:.3f}, {hi:.3f}] | {au - base:+.3f} [{dlo:+.3f}, {dhi:+.3f}] |")
    sig = "mencakup 0 (tidak berbeda)" if dlo <= 0 <= dhi else ("di atas 0" if dlo > 0 else "di bawah 0")
    o += ["", f"Selisih fusi vs klinis: CI 95% {sig}. Audio saja {roc_auc_score(yy, am):.3f}. "
          "CI bootstrap atas peserta; skor ulangan dirata-rata (CI sedikit optimistis)."]
    out = pathlib.Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(o) + "\n")
    print("\n".join(o))


if __name__ == "__main__":
    main()
