"""EKSPLORATORI: murid distilasi HeAR -> kuantisasi int8 golden -> ekspor bobot (tidak menimpa model_v2).

    python -m model.distillation.distill_export cv       # per lipatan (ulangan 0 protokol E0): murid float + probe -> PTQ -> golden int8; AUROC float vs int8
    python -m model.distillation.distill_export final    # murid pada SEMUA klip (tanpa label) + probe pada semua peserta -> qmodel + .mem di runs/model_distill

Probe LR (standardisasi + LR) dilipat menjadi FC terakhir 16->1 (w' = w/sigma, b' = b - sum(w*mu/sigma)), sehingga arsitektur hardware
tidak berubah: logit klip = FC(embedding 16-d), skor peserta = rata-rata logit klip. PTQ saja (tanpa QAT; QAT memakai label).
Kriteria kuantisasi (K-21): penurunan AUROC <= 0,02. Bobot TIDAK di-commit.
"""
from __future__ import annotations
import argparse
import time
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from golden import paths
from golden.blocks import cnn as C
from golden.gen.io_mem import write_mem
from golden import params
from model.evaluation import protocol as P
from model.scratch import export as EX
from model.scratch import mil_train as M

warnings.filterwarnings("ignore")
torch.set_num_threads(8)
K_PCA, EPOCHS, C_PROBE = 32, 25, 0.03


class Student(nn.Module):
    def __init__(s, K=K_PCA):
        super().__init__()
        L, cin = [], 1
        for co in (16, 32, 32):
            L += [nn.Conv2d(cin, co, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)]; cin = co
        s.f = nn.Sequential(*L); s.h = nn.Sequential(nn.Linear(32, 16), nn.ReLU()); s.o = nn.Linear(16, K)

    def emb(s, x):
        return s.h(s.f(x).mean(dim=(2, 3)))

    def forward(s, x):
        return s.o(s.emb(x))


def train_student(Xt, tr_idx, T, seed, epochs=EPOCHS):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    net = Student(T.shape[1]); opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-2)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-3, total_steps=epochs * int(np.ceil(len(tr_idx) / 64)))
    Tt = torch.from_numpy(T.astype(np.float32))
    for ep in range(epochs):
        net.train(); order = rng.permutation(len(tr_idx))
        for i in range(0, len(order), 64):
            b = order[i:i + 64]; xb = Xt[tr_idx[b]]
            xb = xb + torch.randint(-12, 13, (len(b), 1, 1, 1)).float() / 64.0
            xb = torch.roll(xb, int(torch.randint(-2, 3, (1,))), dims=3)
            opt.zero_grad(); ((net(xb) - Tt[b]) ** 2).mean().backward(); opt.step(); sch.step()
    return net.eval()


def embed(net, Xt):
    with torch.no_grad():
        return np.concatenate([net.emb(Xt[i:i + 512]).numpy() for i in range(0, len(Xt), 512)])


def to_hw_net(st: Student, w: np.ndarray, b: float) -> M.Net:
    """Murid + probe terlipat -> M.Net (struktur yang dipakai export.ptq)."""
    net = M.Net(c=(16, 32, 32), h=16)
    net.f.load_state_dict(st.f.state_dict())
    net.h[1].load_state_dict(st.h[0].state_dict())
    with torch.no_grad():
        net.h[3].weight.copy_(torch.tensor(w[None], dtype=torch.float32)); net.h[3].bias.copy_(torch.tensor([b], dtype=torch.float32))
    return net.eval()


def fit_probe(E_part, y):
    pipe = make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler(),
                         LogisticRegression(C=C_PROBE, class_weight="balanced", max_iter=1000)).fit(E_part, y)
    sc, lr = pipe[1], pipe[2]
    assert sc.mean_.shape[0] == 16, "indikator imputer menambah kolom"
    w = lr.coef_[0] / sc.scale_
    b = float(lr.intercept_[0] - np.sum(lr.coef_[0] * sc.mean_ / sc.scale_))
    return w, b


def load():
    z = np.load(paths.features_dir() / "distill_clips.npz", allow_pickle=True)
    u = P.universe()
    pid = {b: i for i, b in enumerate(u.barcode.values)}
    keep = np.array([b in pid for b in z["barcode"]])
    X, E, B = z["X"][keep], z["emb"][keep], z["barcode"][keep]
    cp = np.array([pid[b] for b in B])
    Xt = torch.from_numpy(X.astype(np.float32) / 64.0).permute(0, 2, 1).unsqueeze(1).contiguous()
    return u, X, E, cp, Xt


def part_mean(v, cp, n):
    s = np.zeros((n,) + v.shape[1:]); c = np.zeros(n)
    np.add.at(s, cp, v); np.add.at(c, cp, 1)
    return s / np.maximum(c, 1).reshape((-1,) + (1,) * (v.ndim - 1))


def targets(E, tr_idx):
    mu, sd = E[tr_idx].mean(0), E[tr_idx].std(0) + 1e-6
    pca = PCA(K_PCA, random_state=0).fit((E[tr_idx] - mu) / sd)
    T = pca.transform((E - mu) / sd); return T[tr_idx] / T[tr_idx].std(0)


def cv():
    u, X, E, cp, Xt = load()
    n = len(u); y = u.y.values.astype(int); sx = u.sex_m.values
    fold = P.make_folds(u, 1)[0]
    sf, si = np.zeros(n), np.zeros(n); rows = []; t0 = time.time()
    for k in range(5):
        te_clip = (fold == k)[cp]; tr_idx = np.where(~te_clip)[0]; te_idx = np.where(te_clip)[0]
        st = train_student(Xt, tr_idx, targets(E, tr_idx), seed=k)
        Em = embed(st, Xt)
        trp = np.where(fold != k)[0]; tep = np.where(fold == k)[0]
        Ep = part_mean(Em[tr_idx], cp[tr_idx], n)[trp]
        w, b = fit_probe(Ep, y[trp])
        net = to_hw_net(st, w, b)
        qm = EX.ptq(net, Xt[tr_idx[::3]])
        with torch.no_grad():
            zf = net(Xt[te_idx]).numpy()
        zi = EX.int_logits(qm, X[te_idx])
        corr = float(np.corrcoef(zf, zi)[0, 1])
        sf[tep] = part_mean(zf[:, None], cp[te_idx], n)[tep, 0]; si[tep] = part_mean(zi[:, None], cp[te_idx], n)[tep, 0]
        rows.append((k, corr, float(np.sqrt(np.mean((zf - zi) ** 2)))))
        print(f"lipatan {k}: corr(float,int8)={corr:.4f} rmse={rows[-1][2]:.3f} ({time.time() - t0:.0f}s)", flush=True)
    af, ai = P.auc_strat(y, sf, sx), P.auc_strat(y, si, sx)
    rf, ri = roc_auc_score(y, sf), roc_auc_score(y, si)
    o = ["# Kuantisasi murid distilasi HeAR (eksploratori)", "",
         "Murid + probe terlipat -> PTQ -> golden int8; out-of-fold (ulangan 0 protokol E0), skor peserta = rata-rata logit klip.", "",
         "| lipatan | korelasi logit float vs int8 | RMSE logit |", "|---|---|---|"]
    o += [f"| {k} | {c:.4f} | {r:.3f} |" for k, c, r in rows]
    o += ["", "| | AUROC strat. jenis kelamin | AUROC mentah |", "|---|---|---|",
          f"| float | {af:.3f} | {rf:.3f} |", f"| int8 golden | {ai:.3f} | {ri:.3f} |",
          f"| selisih (float - int8) | {af - ai:+.3f} | {rf - ri:+.3f} |", "",
          f"Kriteria penurunan <= 0,02: {'LULUS' if af - ai <= 0.02 else 'GAGAL'}. Catatan: fusi klinis dilakukan di host, tidak dihitung di sini."]
    (paths.runs_dir() / "reports" / "distilasi-kuantisasi.md").write_text("\n".join(o) + "\n"); print("\n".join(o))


def final():
    u, X, E, cp, Xt = load()
    n = len(u); y = u.y.values.astype(int)
    tr_idx = np.arange(len(X))
    st = train_student(Xt, tr_idx, targets(E, tr_idx), seed=999)
    Ep = part_mean(embed(st, Xt), cp, n)
    w, b = fit_probe(Ep, y)
    net = to_hw_net(st, w, b)
    qm = EX.ptq(net, Xt[::3])
    stats = {}
    sub = np.arange(0, len(X), 5)
    zi = EX.int_logits(qm, X[sub], stats)
    with torch.no_grad():
        zf = net(Xt[sub]).numpy()
    out = paths.runs_dir() / "model_distill"; out.mkdir(parents=True, exist_ok=True)
    torch.save(net.state_dict(), out / "float.pt"); C.save_qmodel(qm, out / "qmodel.npz")
    for L in qm.layers:
        write_mem(out / f"{L.name}_w.mem", L.w.ravel(), 8, f"{L.name} bobot int8 (cout,cin[,3,3]) row-major")
        write_mem(out / f"{L.name}_b.mem", L.b, params.BIAS_BITS, f"{L.name} bias int24")
    write_mem(out / "requant.mem", [v for L in qm.layers for v in (L.m0, L.shift)], 16, "per layer: M0(Q15), shift")
    print("korelasi float/int8:", float(np.corrcoef(zf, zi)[0, 1]), "| puncak acc:", {k: v for k, v in stats.items()})
    for L in qm.layers:
        print(L.name, L.w.size, "M0", L.m0, "shift", L.shift, "maks|bias|", int(np.abs(L.b).max()))
    print("tersimpan", out)


def vectors(n_clips: int = 8):
    """Vektor uji RTL: n_clips klip nyata + logit golden dari qmodel terekspor (dipakai tb/cnn/test_cnn_trained.py)."""
    out = paths.runs_dir() / "model_distill"
    qm = C.load_qmodel(out / "qmodel.npz")
    z = np.load(paths.features_dir() / "distill_clips.npz", allow_pickle=True)
    sel = np.linspace(0, len(z["X"]) - 1, n_clips).astype(int)
    d = out / "vectors"; d.mkdir(exist_ok=True)
    lg = []
    for k, i in enumerate(sel):
        x = z["X"][i].astype(np.int64).T[None, :, :]          # (1,40,50)
        write_mem(d / f"clip{k:02d}_in.mem", x.ravel(), 8)
        lg.append(int(C.cnn_forward(x, qm)[0]))
    write_mem(d / "logits.mem", lg, 16)
    print("vektor:", d, "logit golden:", lg)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["cv", "final", "vectors"]); a = ap.parse_args()
    {"cv": cv, "final": final, "vectors": vectors}[a.cmd]()
