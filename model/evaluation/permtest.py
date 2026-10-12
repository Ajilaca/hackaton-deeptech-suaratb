"""Uji permutasi final untuk kandidat embedding (hear, operaCT, operaCE) dengan statistik maksimum.

    python -m model.evaluation.permtest --kind label --perm 20 --pilot      # pilot: cek kode dan ukur waktu
    python -m model.evaluation.permtest --kind label --perm 1000            # H0 (a): audio tidak berhubungan dengan label
    python -m model.evaluation.permtest --kind delta --perm 1000            # H0 (b): audio tidak menambah di atas klinis

PRAREGISTRASI (ditulis 2026-10-11 sebelum angka final; tidak boleh diubah setelah melihat hasil):
  Kandidat: hear, operaCT, operaCE. Semesta = protokol E0 (364 peserta; 60 positif). Pelatih: LR, C = 0,03 tetap (tanpa tuning,
  class_weight seimbang), fusi LR C = 0,3 pada [skor klinis, skor audio OOF dalam]. Skor klinis tetap (dilatih pada peserta tanpa audio).
  Statistik: (a) AUROC audio terstratifikasi jenis kelamin; (b) Δ = AUROC strat. fusi - AUROC strat. klinis.
  Teramati = rata-rata per ulangan (R ulangan CV); nol = SATU ulangan per permutasi (ulangan bergilir).
  Nol (a): label diacak dalam klinik x jenis kelamin, SELURUH pipeline diulang. Nol (b): baris fitur audio diacak dalam klinik x jenis
  kelamin, label dan skor klinis tetap. Permutasi yang sama dipakai untuk ketiga kandidat; p disesuaikan lewat maksimum atas kandidat:
  p = (1 + #{perm: max_c nol_c >= teramati_c}) / (N + 1).
  KRITERIA LOLOS distilasi (SEMUA): (1) p_a < 0,01; (2) p_b < 0,05 DAN batas bawah CI bootstrap Δ > 0; (3) leave-one-clinic-out
  (latih Kan -> uji Cha dan sebaliknya) tidak di bawah p95 nol. Bila ada yang gagal: audio tidak masuk model akhir.
"""
from __future__ import annotations
import argparse
import os
import time
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import multiprocessing as mp
import warnings
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from golden import paths
from model.evaluation import protocol as P
from model.pretrained.embed_pretrained import participant_matrix

warnings.filterwarnings("ignore")
CANDS = ["hear", "operaCT", "operaCE"]
C_AUDIO, C_FUSE = 0.03, 0.3
REPS = 5
G: dict = {}


def mk(C):
    return make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler(),
                         LogisticRegression(C=C, class_weight="balanced", max_iter=500))


def one_rep(F, y, strat, fold, clin, sx, scores=False):
    """Satu ulangan CV: kembalikan (AUROC strat. audio, AUROC strat. fusi)."""
    a = np.zeros(len(y)); fu = np.zeros(len(y))
    for k in range(5):
        te = fold == k; tr = ~te
        s_tr = np.zeros(tr.sum()); Ftr, ytr, st = F[tr], y[tr], strat[tr]
        for i, (p, q) in enumerate(StratifiedKFold(5, shuffle=True, random_state=1).split(Ftr, st)):
            m = mk(C_AUDIO); m.fit(Ftr[p], ytr[p]); s_tr[q] = m.decision_function(Ftr[q])
        m = mk(C_AUDIO); m.fit(Ftr, ytr); a[te] = m.decision_function(F[te])
        f = mk(C_FUSE); f.fit(np.column_stack([clin[tr], s_tr]), ytr)
        fu[te] = f.decision_function(np.column_stack([clin[te], a[te]]))
    if scores:
        return fu
    return P.auc_strat(y, a, sx), P.auc_strat(y, fu, sx)


def groups(u):
    return (u.fac + u.sex_m.astype(str)).values


def perm_job(args):
    kind, seed, rep = args
    rng = np.random.default_rng(seed)
    u, y, sx, clin, folds = G["u"], G["y"], G["sx"], G["clin"], G["folds"]
    g = groups(u)
    idx = np.arange(len(y))
    yp, order = y.copy(), idx.copy()
    for v in np.unique(g):
        i = np.where(g == v)[0]
        if kind == "label":
            yp[i] = rng.permutation(y[i])
        else:
            order[i] = rng.permutation(i)
    strat = np.array([f"{a}_{b}_{c}" for a, b, c in zip(yp, u.fac, u.sex_m)])
    out = []
    for c in CANDS:
        F = G["F"][c][order] if kind == "delta" else G["F"][c]
        a_, f_ = one_rep(F, yp, strat, folds[rep % folds.shape[0]], clin, sx)
        out.append(a_ if kind == "label" else f_ - P.auc_strat(y, clin, sx))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["label", "delta"], required=True)
    ap.add_argument("--perm", type=int, default=20)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--c", type=float, default=0.03, help="C LR audio (praregistrasi: 0,03); selain itu = analisis sensitivitas")
    ap.add_argument("--noise", action="store_true", help="kontrol negatif: ganti fitur dengan Gaussian acak")
    ap.add_argument("--tag", default="")
    ap.add_argument("--noise-seed", type=int, default=0)
    a = ap.parse_args()
    globals()["C_AUDIO"] = a.c
    u = P.universe()
    Fs = {c: participant_matrix(c, u) for c in CANDS}
    ok = np.all([~np.isnan(Fs[c]).any(1) for c in CANDS], axis=0)
    u = u[ok].reset_index(drop=True)
    if a.noise:
        rg = np.random.default_rng(a.noise_seed)
        Fs = {c: rg.normal(size=Fs[c].shape) for c in CANDS}
    G.update(u=u, F={c: Fs[c][ok] for c in CANDS}, y=u.y.values.astype(int), sx=u.sex_m.values, clin=u.clin.values,
             folds=P.make_folds(u, REPS))
    y, sx, clin, folds = G["y"], G["sx"], G["clin"], G["folds"]
    strat = u.strat.values
    t0 = time.time()
    obs = []
    for c in CANDS:
        r = np.array([one_rep(G["F"][c], y, strat, folds[r_], clin, sx) for r_ in range(REPS)])
        obs.append(r[:, 0].mean() if a.kind == "label" else (r[:, 1] - P.auc_strat(y, clin, sx)).mean())
    t_obs = time.time() - t0
    ci = {}
    if a.kind == "delta":
        for c in CANDS:
            fu = [one_rep(G["F"][c], y, strat, folds[r_], clin, sx, scores=True) for r_ in range(REPS)]
            ci[c] = P.boot_delta_reps(y, fu, clin, sx, n=1000)
    t0 = time.time()
    with mp.Pool(a.workers) as pool:
        null = np.array(pool.map(perm_job, [(a.kind, 1000 + i, i) for i in range(a.perm)], chunksize=1))
    t_perm = time.time() - t0
    mx = null.max(1)
    stat = "AUROC audio strat." if a.kind == "label" else "Δ fusi - klinis (strat.)"
    o = [f"# Uji permutasi ({a.kind}): {stat}", "", f"n={len(y)}, positif={int(y.sum())}, permutasi={a.perm}, ulangan teramati={REPS}, "
         f"C audio {a.c}{' (kontrol negatif: fitur acak)' if a.noise else ''}. Waktu: teramati {t_obs:.0f} s; permutasi {t_perm:.0f} s ({t_perm / a.perm:.1f} s/permutasi dengan {a.workers} proses).", "",
         "| kandidat | teramati | nol (mean / p95 / p99 / maks), per kandidat | p tidak disesuaikan | p disesuaikan (maks 3 kandidat) | CI 95% bootstrap Δ |",
         "|---|---|---|---|---|---|"]
    for j, c in enumerate(CANDS):
        n = null[:, j]
        pu = (1 + np.sum(n >= obs[j])) / (a.perm + 1)
        pa = (1 + np.sum(mx >= obs[j])) / (a.perm + 1)
        o.append(f"| {c} | {obs[j]:.3f} | {n.mean():.3f} / {np.percentile(n, 95):.3f} / {np.percentile(n, 99):.3f} / {n.max():.3f} | {pu:.4f} | {pa:.4f} | {('[%+.3f, %+.3f]' % ci[c]) if c in ci else '-'} |")
    o += ["", f"Nol maksimum atas kandidat: mean {mx.mean():.3f}, p95 {np.percentile(mx, 95):.3f}, p99 {np.percentile(mx, 99):.3f}.",
          "Resolusi p minimum = 1/(N+1)."]
    sfx = ("-pilot" if a.pilot else "") + (f"-{a.tag}" if a.tag else "")
    name = f"permtest-{a.kind}{sfx}.md"
    (paths.runs_dir() / "reports" / name).write_text("\n".join(o) + "\n")
    np.savez(paths.runs_dir() / f"permtest-{a.kind}{sfx}.npz", null=null, obs=obs)
    print("\n".join(o))


if __name__ == "__main__":
    main()
