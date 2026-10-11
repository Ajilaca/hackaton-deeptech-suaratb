"""Baseline CNN FLOAT (arsitektur yang sama dengan golden) + uji sensitivitas lebar FFT, CV 5-lipat per peserta.

    python -m model.baseline [--epochs 20] [--threads 4]

Rezim A: latih pada fitur float-ref, uji pada fitur golden W=24/22/20/18 (sensitivitas lebar FFT).
Rezim B: latih dan uji pada fitur golden W=24 (baseline yang dapat dipakai).
Label klip = label peserta (lemah). Skor peserta = rata-rata logit klip. Tanpa pemilihan epoch pada lipatan uji.
Hasil ditulis ke $SUARATB_DATA/runs/reports/baseline-float.md; logit out-of-fold ke $SUARATB_DATA/runs/baseline_v1.npz.
"""
from __future__ import annotations
import argparse
import pathlib
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from golden import paths

VARS = ["x_ref", "x_w24", "x_w22", "x_w20", "x_w18"]


class Net(nn.Module):
    def __init__(self, c=(16, 32, 32), h=16):
        super().__init__()
        L, cin = [], 1
        for co in c:
            L += [nn.Conv2d(cin, co, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)]
            cin = co
        self.f = nn.Sequential(*L)
        self.h = nn.Sequential(nn.Linear(cin, h), nn.ReLU(), nn.Linear(h, 1))

    def forward(self, x):
        z = self.f(x).mean(dim=(2, 3))
        return self.h(z).squeeze(1)


def to_t(x):                       # (N,50,40) int8 -> (N,1,40,50) float
    return torch.from_numpy(x.astype(np.float32) / 64.0).permute(0, 2, 1).unsqueeze(1).contiguous()


def train(xtr, ytr, epochs, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    net = Net()
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-3, total_steps=epochs * ((len(xtr) + 63) // 64))
    pw = torch.tensor((len(ytr) - ytr.sum()) / max(ytr.sum(), 1), dtype=torch.float32)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pw)
    X, Y = to_t(xtr), torch.from_numpy(ytr.astype(np.float32))
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(len(X))
        for i in range(0, len(X), 64):
            b = perm[i:i + 64]
            xb = X[b]
            xb = xb + (torch.randint(-4, 5, (len(b), 1, 1, 1)).float() / 64.0)      # variasi gain
            sh = int(torch.randint(-2, 3, (1,)))
            xb = torch.roll(xb, sh, dims=3)                                           # geser waktu
            opt.zero_grad()
            lossf(net(xb), Y[b]).backward()
            opt.step(); sched.step()
    return net


@torch.no_grad()
def predict(net, x):
    net.eval()
    return torch.cat([net(to_t(x[i:i + 512])) for i in range(0, len(x), 512)]).numpy()


def part_scores(logit, barcodes):
    d = pd.DataFrame(dict(b=barcodes, s=logit)).groupby("b").s.mean()
    return d


def auc_ci(y, s, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    a = roc_auc_score(y, s)
    idx = np.arange(len(y)); bs = []
    for _ in range(n):
        j = rng.choice(idx, len(idx))
        if y[j].min() == y[j].max():
            continue
        bs.append(roc_auc_score(y[j], s[j]))
    return a, np.percentile(bs, 2.5), np.percentile(bs, 97.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", default=str(paths.runs_dir() / "reports" / "baseline-float.md"))
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    fd = paths.features_dir()
    Z = np.load(fd / "feats_v1.npz"); meta = pd.read_csv(fd / "meta_v1.csv")
    sp = pd.read_csv(paths.processed_dir() / "splits_v1.csv")
    meta = meta.merge(sp[["barcode", "fold"]], on="barcode")
    y = meta.y.values; fold = meta.fold.values; bc = meta.barcode.values
    oof = {("A", v): np.zeros(len(y)) for v in VARS}
    oof[("B", "x_w24")] = np.zeros(len(y))
    t0 = time.time()
    for f in range(5):
        tr, te = fold != f, fold == f
        for regime, trv in (("A", "x_ref"), ("B", "x_w24")):
            net = train(Z[trv][tr], y[tr], a.epochs, seed=f)
            tests = VARS if regime == "A" else ["x_w24"]
            for v in tests:
                oof[(regime, v)][te] = predict(net, Z[v][te])
        print(f"lipatan {f} selesai ({time.time() - t0:.0f}s)", flush=True)
    pdf = meta.drop_duplicates("barcode")[["barcode", "y", "facility"]].set_index("barcode")
    res = {}
    for k, s in oof.items():
        ps = part_scores(s, bc).reindex(pdf.index)
        a_c = roc_auc_score(y, s)
        a_p, lo, hi = auc_ci(pdf.y.values, ps.values)
        res[k] = (a_c, a_p, lo, hi, ps.values)
    ref = res[("A", "x_ref")][4]
    o = ["# Baseline float: CV 5-lipat per peserta (audio perekam, Zambia CIDRZ)", "",
         f"Dihasilkan `python -m model.baseline --epochs {a.epochs}`. {len(y)} klip, {len(pdf)} peserta, "
         f"{int(pdf.y.sum())} positif TB (ground_truth_tb). Label klip = label peserta (lemah). "
         "Skor peserta = rata-rata logit klip. CI 95% bootstrap atas peserta (1000x).", "",
         "| rezim | fitur uji | AUROC klip | AUROC peserta | CI 95% peserta | Δ vs float-ref (peserta, CI 95%) |", "|---|---|---|---|---|---|"]
    rng = np.random.default_rng(1)
    yy = pdf.y.values
    for (regime, v), (ac, ap_, lo, hi, ps) in res.items():
        d = ""
        if not (regime == "A" and v == "x_ref"):
            ds = []
            for _ in range(1000):
                j = rng.choice(len(yy), len(yy))
                if yy[j].min() == yy[j].max():
                    continue
                ds.append(roc_auc_score(yy[j], ps[j]) - roc_auc_score(yy[j], ref[j]))
            d = f"{ap_ - res[('A', 'x_ref')][1]:+.3f} [{np.percentile(ds, 2.5):+.3f}, {np.percentile(ds, 97.5):+.3f}]"
        o.append(f"| {regime} (latih {'float-ref' if regime == 'A' else 'W=24'}) | {v} | {ac:.3f} | {ap_:.3f} | "
                 f"[{lo:.3f}, {hi:.3f}] | {d} |")
    # per klinik (rezim B)
    o += ["", "## Per klinik (rezim B, W=24, AUROC peserta)", "| klinik | peserta | positif | AUROC |", "|---|---|---|---|"]
    psB = res[("B", "x_w24")][4]
    for fac, g in pdf.assign(s=psB).groupby("facility"):
        ok = g.y.nunique() == 2
        o.append(f"| {fac} | {len(g)} | {int(g.y.sum())} | {roc_auc_score(g.y, g.s):.3f} |" if ok else f"| {fac} | {len(g)} | {int(g.y.sum())} | n/a |")
    o += ["", "Catatan: 34 positif membuat CI lebar; angka ini bukan klaim kinerja. Fusi gejala klinis belum dikerjakan."]
    pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True); pathlib.Path(a.out).write_text("\n".join(o) + "\n")
    rd = paths.runs_dir(); rd.mkdir(parents=True, exist_ok=True)
    np.savez(rd / "baseline_v1.npz", y=y, fold=fold, barcode=bc, **{f"{r}_{v}": s for (r, v), s in oof.items()})
    print("\n".join(o))


if __name__ == "__main__":
    main()
