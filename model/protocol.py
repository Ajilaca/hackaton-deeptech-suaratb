"""E0: protokol evaluasi tunggal untuk eksperimen audio (embedding pralatih, statistik log-mel, dst.).

    python -m model.protocol --baseline [--perm 40] [--out $SUARATB_DATA/runs/reports/protokol-baseline.md]

Ditetapkan SEBELUM ada angka baru:
  - Unit = peserta. Semesta = peserta berlabel yang punya audio (barcode asli, hasil audit E1). Lipatan 5 x 10 ulangan,
    stratifikasi label x klinik (Chai digabung Cha: positifnya 1) x jenis kelamin; disimpan di processed/protocol_folds.csv.
  - Skor klinis = LR klinis diperluas dilatih HANYA pada peserta tanpa audio (tidak pernah melihat peserta uji).
  - Metrik utama (urut): (1) AUROC terstratifikasi jenis kelamin (rata-rata AUROC dalam laki-laki dan perempuan, berbobot
    n_pos x n_neg); (2) selisih AUROC fusi(klinis + audio) - klinis, CI bootstrap berpasangan atas peserta.
    Pembanding: AUROC mentah, jenis kelamin+usia saja, klinis saja.
  - Nol: label diacak dalam klinik x jenis kelamin, SELURUH pipeline CV diulang (bukan hanya skor); dilaporkan mean/p95/maks
    dan p-empiris. Audio dianggap punya sinyal hanya bila (1) > p95 nol DAN CI selisih (2) tidak mencakup <= 0.
  - Uji jalan pintas: audio -> jenis kelamin, audio -> klinik (AUROC/akurasi), dilaporkan terpisah.
  - Semua varian dilaporkan; tidak ada pemilihan varian terbaik di luar CV bersarang.
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
from model import dataset
from model.clinical import clin_extended, lr, audio_score

warnings.filterwarnings("ignore")
REPS, K = 10, 5


def universe() -> pd.DataFrame:
    """Peserta berlabel dengan audio (barcode asli atau pemulihan yakin dari audit E1) + kovariat + skor klinis eksternal."""
    gm = dataset.load_meta().drop_duplicates("barcode").reset_index(drop=True)
    gm = gm[gm.ground_truth_tb.isin(["pos", "neg"])].reset_index(drop=True)
    fx = pd.read_csv(paths.processed_dir() / "barcode_fix.csv")
    have = set(fx[fx.tier.isin(["asli", "pola", "tengah"])].barcode_fix.dropna())
    gm["has_audio"] = gm.barcode.isin(have)
    allp = gm.barcode.values
    Fx, _ = clin_extended(gm, allp)
    ina, ia = np.where(~gm.has_audio.values)[0], np.where(gm.has_audio.values)[0]
    m = lr(); m.fit(Fx[ina], gm.y.values[ina])
    gm["clin"] = np.nan
    gm.loc[ia, "clin"] = m.decision_function(Fx[ia])
    gm["sex_m"] = (gm.sex.astype(str).str.lower() == "m").astype(int)
    gm["fac"] = gm.facility_code.astype(str).replace({"Chai": "Cha"})
    u = gm[gm.has_audio].reset_index(drop=True)
    u["strat"] = u.y.astype(str) + "_" + u.fac + "_" + u.sex_m.astype(str)
    return u


def make_folds(u: pd.DataFrame, reps: int = REPS) -> np.ndarray:
    f = np.zeros((reps, len(u)), int)
    for r in range(reps):
        for k, (_, te) in enumerate(StratifiedKFold(K, shuffle=True, random_state=r).split(u, u.strat)):
            f[r, te] = k
    return f


def save_folds(u, f):
    d = pd.DataFrame(f.T, columns=[f"rep{r}" for r in range(f.shape[0])]); d.insert(0, "barcode", u.barcode.values)
    d.to_csv(paths.processed_dir() / "protocol_folds.csv", index=False)


def auc_strat(y, s, g) -> float:
    """AUROC dalam strata g, dirata-rata berbobot n_pos x n_neg."""
    num = den = 0.0
    for v in np.unique(g):
        m = g == v
        n1, n0 = int(y[m].sum()), int((1 - y[m]).sum())
        if n1 and n0:
            num += n1 * n0 * roc_auc_score(y[m], s[m]); den += n1 * n0
    return num / den


def oof_scores(F, y, strat, folds, cov=None, clin=None):
    """Skor audio OOF (reps, n) dan skor fusi klinis+audio OOF; audio dihitung di dalam lipatan (skor latih = OOF dalam)."""
    reps, n = folds.shape
    a = np.zeros((reps, n)); fu = np.zeros((reps, n))
    cov = np.zeros((n, 1)) if cov is None else cov
    for r in range(reps):
        for k in range(K):
            te = folds[r] == k; tr = ~te
            a_tr, a_te = audio_score(F[tr], y[tr], F[te], cov[tr], cov[te], False, strat[tr])
            a[r, te] = a_te
            if clin is not None:
                m = lr(C=0.3); m.fit(np.column_stack([clin[tr], a_tr]), y[tr])
                fu[r, te] = m.decision_function(np.column_stack([clin[te], a_te]))
    return a, fu


def boot_delta(y, s1, s0, g, n=2000, seed=0):
    rng = np.random.default_rng(seed); v = []
    for _ in range(n):
        j = rng.integers(0, len(y), len(y))
        if y[j].sum() in (0, len(j)):
            continue
        try:
            v.append(auc_strat(y[j], s1[j], g[j]) - auc_strat(y[j], s0[j], g[j]))
        except ZeroDivisionError:
            continue
    return np.percentile(v, 2.5), np.percentile(v, 97.5)


def boot_delta_reps(y, fu, clin, g, n=1000, seed=0):
    """Δ per ulangan (fusi OOF satu ulangan vs skor klinis), dirata-rata antar-ulangan; bootstrap atas peserta."""
    rng = np.random.default_rng(seed); v = []
    for _ in range(n):
        j = rng.integers(0, len(y), len(y))
        if y[j].sum() in (0, len(j)):
            continue
        try:
            c = auc_strat(y[j], clin[j], g[j])
            v.append(np.mean([auc_strat(y[j], fu[r][j], g[j]) - c for r in range(len(fu))]))
        except ZeroDivisionError:
            continue
    return np.percentile(v, 2.5), np.percentile(v, 97.5)


def perm_null(F, y, u, folds, n_perm, seed=7):
    rng = np.random.default_rng(seed); out = []
    for _ in range(n_perm):
        yp = y.copy()
        for g in np.unique(u.fac + u.sex_m.astype(str)):
            i = np.where((u.fac + u.sex_m.astype(str)).values == g)[0]; yp[i] = rng.permutation(y[i])
        st = np.array([f"{a}_{b}_{c}" for a, b, c in zip(yp, u.fac, u.sex_m)])
        a, _ = oof_scores(F, yp, st, folds[:1])
        out.append(auc_strat(yp, a[0], u.sex_m.values))
    return np.array(out)


def evaluate(F: np.ndarray, u: pd.DataFrame, folds: np.ndarray, name: str, n_perm: int = 40, reps: int | None = None) -> list[str]:
    """F: (len(u), d) fitur peserta, urutan = u. Kembalikan baris tabel markdown (metrik protokol)."""
    reps = reps or folds.shape[0]; folds = folds[:reps]
    y = u.y.values.astype(int); sx = u.sex_m.values; clin = u.clin.values
    cov = np.column_stack([sx, u.age.astype(float).fillna(u.age.astype(float).median()).values])
    a, fu = oof_scores(F, y, u.strat.values, folds, cov, clin)
    # Semua statistik per ulangan (satu skor OOF per ulangan), lalu dirata-rata: nol memakai satu ulangan, jadi adil.
    raw = float(np.mean([roc_auc_score(y, a[r]) for r in range(reps)]))
    st = float(np.mean([auc_strat(y, a[r], sx) for r in range(reps)]))
    st_f = float(np.mean([auc_strat(y, fu[r], sx) for r in range(reps)]))
    st_c = auc_strat(y, clin, sx)
    d = st_f - st_c
    d_lo, d_hi = boot_delta_reps(y, fu, clin, sx)
    nul = perm_null(F, y, u, folds, n_perm) if n_perm else np.array([np.nan])
    p = (np.sum(nul >= st) + 1) / (len(nul) + 1) if n_perm else float("nan")
    ok = (st > np.percentile(nul, 95)) and d_lo > 0 if n_perm else False
    return [f"| {name} | {raw:.3f} | **{st:.3f}** | {nul.mean():.3f} / {np.nanpercentile(nul, 95):.3f} / "
            f"{np.nanmax(nul):.3f} | {p:.3f} | {st_f:.3f} | {d:+.3f} [{d_lo:+.3f}, {d_hi:+.3f}] | {'ADA' if ok else 'tidak ada'} |"]


HEAD = ["| fitur | AUROC mentah | AUROC strat. jenis kelamin | nol mean / p95 / maks | p-empiris | fusi klinis+audio (strat.) | Δ vs klinis [CI 95%] | sinyal (kriteria) |",
        "|---|---|---|---|---|---|---|---|"]


def reference_rows(u):
    y = u.y.values.astype(int); sx = u.sex_m.values
    age = u.age.astype(float).fillna(u.age.astype(float).median()).values
    return [f"| usia saja (AUROC dalam jenis kelamin) | {roc_auc_score(y, age):.3f} | {auc_strat(y, age, sx):.3f} | | | | | |",
            f"| klinis diperluas (dilatih pada peserta tanpa audio) | {roc_auc_score(y, u.clin):.3f} | {auc_strat(y, u.clin.values, sx):.3f} | | | | | |"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", action="store_true", help="evaluasi statistik log-mel (v2) dengan protokol ini")
    ap.add_argument("--perm", type=int, default=40)
    ap.add_argument("--reps", type=int, default=REPS)
    ap.add_argument("--out", default=str(paths.runs_dir() / "reports" / "protokol-baseline.md"))
    a = ap.parse_args()
    u = universe(); f = make_folds(u, a.reps); save_folds(u, f)
    o = ["# Protokol evaluasi audio (E0) dan baseline", "",
         f"Semesta: {len(u)} peserta berlabel dengan audio, {int(u.y.sum())} positif; laki-laki {int(u.sex_m.sum())} "
         f"({int(u[u.sex_m == 1].y.sum())} positif), perempuan {int((1 - u.sex_m).sum())} ({int(u[u.sex_m == 0].y.sum())} positif). "
         f"Lipatan {K} x {a.reps}, stratifikasi label x klinik x jenis kelamin. Aturan dan metrik: docstring `model/protocol.py`.", ""]
    o += HEAD
    o += reference_rows(u)
    if a.baseline:
        from model.remodel import load, participant_features
        x, meta, mask, sp = load("v2")
        have = u.barcode.isin(set(sp.barcode))
        us = u[have].reset_index(drop=True)
        fs = make_folds(us, a.reps)
        Fp, _ = participant_features(x, meta, mask, us.barcode.values, sub=False, use_mask=False)
        o += evaluate(Fp, us, fs, f"statistik log-mel v2 (80 fitur, n={len(us)}, positif {int(us.y.sum())})", a.perm)
    out = pathlib.Path(a.out); out.parent.mkdir(parents=True, exist_ok=True); out.write_text("\n".join(o) + "\n")
    print("\n".join(o))


if __name__ == "__main__":
    main()
