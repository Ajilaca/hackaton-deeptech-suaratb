"""EKSPLORATORI (K-23 sudah memutuskan distilasi tidak dikejar; ini hanya untuk mengukur): distilasi fitur HeAR -> CNN golden.

    python -m model.distillation.distill_hear build            # tabel klip: fitur golden (50x40 int8) + embedding HeAR per klip
    python -m model.distillation.distill_hear run [--reps 2]   # distilasi per lipatan + evaluasi protokol E0 + nol permutasi + banding guru

Desain tanpa kebocoran label:
  - Distilasi TIDAK memakai label TB: murid (arsitektur golden: Conv16/32/32 -> GAP32 -> FC32->16 -> kepala 16->K) meregresi K=32
    komponen PCA dari embedding HeAR (512-d) per klip. PCA dan murid dilatih hanya pada klip peserta lipatan latih, per lipatan.
  - Murid melihat 0,5 s log-mel (fitur hardware); guru melihat 2 s. Ketidakcocokan konteks ini batas atas bagi murid.
  - Peserta = rata-rata embedding 16-d murid (jaringan yang sama dengan hardware; probe 16->1 setara FC terakhir). LR probe
    memakai label, di dalam lipatan yang sama dengan protokol E0. Pembanding: probe pada embedding guru dari klip yang sama.
  - Nol permutasi: label diacak dalam klinik x jenis kelamin, murid beku (tak bergantung label), hanya probe yang dilatih ulang.
"""
from __future__ import annotations
import argparse
import multiprocessing as mp
import os
import pathlib
import time
import warnings
import numpy as np
import pandas as pd
from golden import params, paths
from golden.blocks import frontend as F
from golden.blocks.cough_det import hop_energy, cough_det
from golden.blocks.preemph import preemph
from model.data.audio_io import load_16k

warnings.filterwarnings("ignore")
HOP = params.HOP
FILES = [("emb_hear.npz", ("phonec", "phoneb", "phonea", "recorder")), ("emb_hear_phonea.npz", ("phonea",)),
         ("emb_hear_phoneb.npz", ("phoneb",))]


def _job(a):
    """Satu peserta/perangkat: urutan klip identik dengan extract_hear (energi puncak menurun, maks 6)."""
    barcode, ps = a
    cands = []
    for pth in ps:
        x = load_16k(str(paths.raw_dir() / pth)).astype(np.int64)
        E = hop_energy(preemph(x))
        for e in cough_det(E):
            cands.append((float(E[e.onset:e.onset + 12].max()), pth, x, e.start * HOP))
    cands.sort(key=lambda t: -t[0])
    out = []
    for _, pth, x, s0 in cands[:6]:
        X = F.features_q(F.pad_clip(x[s0:s0 + params.CLIP_SAMPLES_PADDED])).astype(np.int8)
        out.append((pth, X))
    return barcode, out


def build():
    fx = pd.read_csv(paths.processed_dir() / "barcode_fix.csv")
    fx = fx[fx.tier.isin(["asli", "pola", "tengah"]) & fx.path.str.lower().str.endswith(".wav")]
    Xs, Es, B, D, Pth = [], [], [], [], []
    for fn, pref in FILES:
        z = np.load(paths.features_dir() / fn, allow_pickle=True)
        jobs = []
        for b, g in fx.groupby("barcode_fix"):
            for d in pref:
                gd = g[g.device == d]
                if len(gd):
                    jobs.append((b, gd.path.tolist())); break
        with mp.Pool(8) as p:
            res = p.map(_job, jobs, chunksize=4)
        rows = [(b, pth, X) for b, out in res for pth, X in out]
        assert len(rows) == len(z["emb"]), (fn, len(rows), len(z["emb"]))
        assert all(r[1] == s for r, s in zip(rows, z["path"])) and all(r[0] == s for r, s in zip(rows, z["barcode"])), fn
        Xs += [r[2] for r in rows]; Es.append(z["emb"]); B += [r[0] for r in rows]; Pth += [r[1] for r in rows]
        D += list(z["device"])
        print(fn, len(rows), flush=True)
    np.savez_compressed(paths.features_dir() / "distill_clips.npz", X=np.stack(Xs), emb=np.concatenate(Es),
                        barcode=np.array(B), device=np.array(D), path=np.array(Pth))


# ---------------------------------------------------------------------------- distilasi + evaluasi
def run(reps: int, epochs: int, n_perm: int, control: str = ""):
    import torch
    import torch.nn as nn
    from sklearn.decomposition import PCA
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from model.evaluation import protocol as P
    torch.set_num_threads(8)
    z = np.load(paths.features_dir() / "distill_clips.npz", allow_pickle=True)
    X, E, B = z["X"], z["emb"], z["barcode"]
    u = P.universe()
    pid = {b: i for i, b in enumerate(u.barcode.values)}
    keep = np.array([b in pid for b in B])
    X, E, B = X[keep], E[keep], B[keep]
    cp = np.array([pid[b] for b in B])                           # indeks peserta per klip
    folds = P.make_folds(u, reps)
    y = u.y.values.astype(int); sx = u.sex_m.values; clin = u.clin.values; strat = u.strat.values
    Xt = torch.from_numpy(X.astype(np.float32) / 64.0).permute(0, 2, 1).unsqueeze(1).contiguous()   # (N,1,40,50)

    class Student(nn.Module):
        def __init__(s, K):
            super().__init__()
            L, cin = [], 1
            for co in (16, 32, 32):
                L += [nn.Conv2d(cin, co, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)]; cin = co
            s.f = nn.Sequential(*L); s.h = nn.Sequential(nn.Linear(32, 16), nn.ReLU()); s.o = nn.Linear(16, K)

        def emb(s, x):
            return s.h(s.f(x).mean(dim=(2, 3)))

        def forward(s, x):
            return s.o(s.emb(x))

    def train_student(tr_idx, T, seed):
        torch.manual_seed(seed); rng = np.random.default_rng(seed)
        net = Student(T.shape[1]); opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-2)
        steps = epochs * int(np.ceil(len(tr_idx) / 64))
        sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-3, total_steps=steps)
        Tt = torch.from_numpy(T.astype(np.float32))
        for ep in range(epochs):
            net.train(); order = rng.permutation(len(tr_idx))
            for i in range(0, len(order), 64):
                b = order[i:i + 64]; xb = Xt[tr_idx[b]]
                xb = xb + torch.randint(-12, 13, (len(b), 1, 1, 1)).float() / 64.0         # gain
                xb = torch.roll(xb, int(torch.randint(-2, 3, (1,))), dims=3)               # geser waktu kecil
                opt.zero_grad(); loss = ((net(xb) - Tt[b]) ** 2).mean(); loss.backward(); opt.step(); sch.step()
        return net.eval()

    def probe(C):
        return make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler(),
                             LogisticRegression(C=C, class_weight="balanced", max_iter=500))

    def part_mean(M, idx_clips, n):
        S = np.zeros((n, M.shape[1])); c = np.zeros(n)
        for m, p in zip(M, cp[idx_clips]):
            S[p] += m; c[p] += 1
        return S / np.maximum(c, 1)[:, None], c > 0

    def stat(Fk, yy, st, fold):
        """Fk: daftar 5 matriks peserta (per lipatan). Kembalikan (AUROC strat. audio, AUROC strat. fusi)."""
        a = np.zeros(len(yy)); fu = np.zeros(len(yy))
        for k in range(5):
            te = fold == k; tr = ~te; Fm = Fk[k]
            s_tr = np.zeros(tr.sum()); Ftr, ytr = Fm[tr], yy[tr]
            for p_, q_ in StratifiedKFold(5, shuffle=True, random_state=1).split(Ftr, st[tr]):
                m = probe(0.03); m.fit(Ftr[p_], ytr[p_]); s_tr[q_] = m.decision_function(Ftr[q_])
            m = probe(0.03); m.fit(Ftr, ytr); a[te] = m.decision_function(Fm[te])
            f = probe(0.3); f.fit(np.column_stack([clin[tr], s_tr]), ytr)
            fu[te] = f.decision_function(np.column_stack([clin[te], a[te]]))
        stat.last = a
        return P.auc_strat(yy, a, sx), P.auc_strat(yy, fu, sx)

    res = {"murid": [], "guru": []}; fid = []; Fk_student = []; t0 = time.time()
    for r in range(reps):
        fs = []
        for k in range(5):
            te_part = folds[r] == k
            te_clip = te_part[cp]; tr_clip = ~te_clip
            tr_idx = np.where(tr_clip)[0]
            mu, sd = E[tr_idx].mean(0), E[tr_idx].std(0) + 1e-6
            pca = PCA(32, random_state=0).fit((E[tr_idx] - mu) / sd)
            Tall = pca.transform((E - mu) / sd); Tsd = Tall[tr_idx].std(0)
            T_tr = Tall[tr_idx] / Tsd
            if control == "shuffled":                      # kontrol: target tak berhubungan dengan klip
                T_tr = T_tr[np.random.default_rng(7 + k).permutation(len(T_tr))]
            if control == "random_init":                   # kontrol: jaringan acak tanpa pelatihan
                torch.manual_seed(100 * r + k); net = Student(32).eval()
            else:
                net = train_student(tr_idx, T_tr, seed=100 * r + k)
            with torch.no_grad():
                pred = np.concatenate([net(Xt[i:i + 512]).numpy() for i in range(0, len(Xt), 512)])
                M = np.concatenate([net.emb(Xt[i:i + 512]).numpy() for i in range(0, len(Xt), 512)])
            te_idx = np.where(te_clip)[0]
            tgt = Tall[te_idx] / Tsd
            r2 = 1 - ((pred[te_idx] - tgt) ** 2).sum() / ((tgt - tgt.mean(0)) ** 2).sum()
            fid.append(float(r2))
            Fm, ok = part_mean(M, np.arange(len(M)), len(u))
            fs.append(Fm)
            print(f"rep{r} lipatan{k}: R2 uji klip = {r2:.3f} ({time.time() - t0:.0f}s)", flush=True)
        Fk_student.append(fs)
    Eg, _ = part_mean(E, np.arange(len(E)), len(u))
    has = np.bincount(cp, minlength=len(u)) > 0
    assert has.all(), "ada peserta tanpa klip"
    guru = [Eg] * 5
    sc = {"murid": [stat(Fk_student[r], y, strat, folds[r]) for r in range(reps)],
          "guru": [stat(guru, y, strat, folds[r]) for r in range(reps)]}
    c_st = P.auc_strat(y, clin, sx)
    # pemeriksaan jalan pintas (rep 0): embedding memprediksi apa, dan AUROC audio dalam strata
    def sc_oof(Fk, tgt, fold):
        s_ = np.zeros(len(tgt))
        for k in range(5):
            te = fold == k
            m = probe(0.03); m.fit(Fk[k][~te], tgt[~te]); s_[te] = m.decision_function(Fk[k][te])
        return roc_auc_score(tgt, s_)
    short = {}
    for nm, Fk in (("guru", guru), ("murid", Fk_student[0])):
        stat(Fk, y, strat, folds[0]); a_oof = stat.last.copy()
        fac = u.fac.values
        short[nm] = dict(sex=sc_oof(Fk, sx, folds[0]), kan=sc_oof(Fk, (fac == "Kan").astype(int), folds[0]),
                         in_kan=roc_auc_score(y[fac == "Kan"], a_oof[fac == "Kan"]),
                         in_cha=roc_auc_score(y[fac == "Cha"], a_oof[fac == "Cha"]),
                         in_m=roc_auc_score(y[sx == 1], a_oof[sx == 1]), in_f=roc_auc_score(y[sx == 0], a_oof[sx == 0]))
    # nol permutasi (murid beku; hanya probe diulang), rep 0
    rng = np.random.default_rng(5); g = (u.fac + u.sex_m.astype(str)).values
    nulls = {"murid": [], "guru": []}
    for _ in range(n_perm):
        yp = y.copy()
        for v in np.unique(g):
            i = np.where(g == v)[0]; yp[i] = rng.permutation(y[i])
        st = np.array([f"{a}_{b}_{c}" for a, b, c in zip(yp, u.fac, u.sex_m)])
        nulls["murid"].append(stat(Fk_student[0], yp, st, folds[0])[0])
        nulls["guru"].append(stat(guru, yp, st, folds[0])[0])
    o = ["# Distilasi HeAR -> CNN golden (eksploratori)" + (f" [kontrol: {control}]" if control else ""), "",
         f"{len(u)} peserta, {int(y.sum())} positif; {len(X)} klip; {reps} ulangan x 5 lipatan; {epochs} epoch; K=32 komponen PCA. "
         f"Fidelitas murid pada klip uji (R² komponen PCA): rata-rata {np.mean(fid):.3f} (min {np.min(fid):.3f}, maks {np.max(fid):.3f}). "
         f"Waktu {time.time() - t0:.0f} s.", "",
         "| model | AUROC strat. jenis kelamin (rata ulangan) | nol (mean / p95 / maks) | p-empiris | fusi strat. | Δ vs klinis |",
         "|---|---|---|---|---|---|"]
    for nm in ("guru", "murid"):
        a_ = np.mean([s[0] for s in sc[nm]]); f_ = np.mean([s[1] for s in sc[nm]]); nl = np.array(nulls[nm])
        p = (1 + np.sum(nl >= a_)) / (len(nl) + 1)
        o.append(f"| {'HeAR guru (512-d, klip yang sama)' if nm == 'guru' else 'Murid CNN (16-d, 0,5 s log-mel)'} | **{a_:.3f}** | "
                 f"{nl.mean():.3f} / {np.percentile(nl, 95):.3f} / {nl.max():.3f} | {p:.4f} | {f_:.3f} | {f_ - c_st:+.3f} |")
    o += ["", "## Jalan pintas (ulangan 0)", "| model | memprediksi jenis kelamin | memprediksi klinik Kan | AUROC dalam Kan | dalam Cha | dalam laki-laki | dalam perempuan |",
          "|---|---|---|---|---|---|---|"]
    for nm in ("guru", "murid"):
        d = short[nm]
        o.append(f"| {nm} | {d['sex']:.3f} | {d['kan']:.3f} | {d['in_kan']:.3f} | {d['in_cha']:.3f} | {d['in_m']:.3f} | {d['in_f']:.3f} |")
    o += ["", f"Klinis saja (strat.): {c_st:.3f}. Pembanding model dilatih-dari-nol (model_v2, supervised): AUROC mentah ≈ 0,56.",
          "Nol: label diacak dalam klinik x jenis kelamin, murid beku; hanya probe diulang (nol sah karena distilasi tak memakai label)."]
    out = paths.runs_dir() / "reports" / f"distilasi-hear{'-' + control if control else ''}.md"
    out.write_text("\n".join(o) + "\n"); print("\n".join(o))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "run"])
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--perm", type=int, default=200)
    ap.add_argument("--control", default="", choices=["", "random_init", "shuffled"])
    a = ap.parse_args()
    build() if a.cmd == "build" else run(a.reps, a.epochs, a.perm, a.control)
