"""Pelatihan "lebih serius": MIL tingkat peserta (rata-rata logit k batuk, seperti perangkat), augmentasi,
early stopping pada validasi dalam, ensemble seed, CV 5-lipat berulang per peserta, uji permutasi.

    python -m model.mil_train --tag v2 --repeats 2 --seeds 2 --epochs 30 --threads 4
    python -m model.mil_train --tag v2 --perm 8 --repeats 1 --seeds 1 --epochs 20 --threads 4 --out runs/mil_perm.npz

Label peserta = ground_truth_tb. Gabungan batuk dari semua perangkat peserta membentuk "kantong".
Keluaran: $SUARATB_DATA/runs/<out>.npz berisi skor peserta per ulangan/perangkat.
Tidak memakai kolom hasil lab/AI X-ray (xpert, culture, smear, tb_predictions) di sini maupun di fusion.py.
"""
from __future__ import annotations
import argparse
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
from golden import paths

BAG = 4
SCALE = 64.0


class Net(nn.Module):
    def __init__(self, c=(16, 32, 32), h=16, drop=0.25):
        super().__init__()
        L, cin = [], 1
        for co in c:
            L += [nn.Conv2d(cin, co, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)]
            cin = co
        self.f = nn.Sequential(*L)
        self.h = nn.Sequential(nn.Dropout(drop), nn.Linear(cin, h), nn.ReLU(), nn.Linear(h, 1))

    def forward(self, x):
        return self.h(self.f(x).mean(dim=(2, 3))).squeeze(1)


def augment(x):
    """x (B,1,40,50) float (nilai/64). Gain, geser waktu, SpecAugment (mask frekuensi dan waktu)."""
    B = x.shape[0]
    x = x + torch.randint(-12, 13, (B, 1, 1, 1)).float() / SCALE
    x = torch.roll(x, int(torch.randint(-3, 4, (1,))), dims=3)
    mean = x.mean(dim=(1, 2, 3), keepdim=True)
    f0 = torch.randint(0, 40, (B, 1, 1, 1)); fw = torch.randint(0, 6, (B, 1, 1, 1))
    ar = torch.arange(40).view(1, 1, 40, 1)
    x = torch.where((ar >= f0) & (ar < f0 + fw), mean.expand_as(x), x)
    t0 = torch.randint(0, 50, (B, 1, 1, 1)); tw = torch.randint(0, 7, (B, 1, 1, 1))
    at = torch.arange(50).view(1, 1, 1, 50)
    x = torch.where((at >= t0) & (at < t0 + tw), mean.expand_as(x), x)
    return x


def make_bags(pools, parts, rng):
    idx = []
    for p in parts:
        pl = pools[p]
        idx.append(rng.choice(pl, BAG, replace=len(pl) < BAG))
    return np.array(idx)


def bag_logits(net, X, idx):
    B = idx.shape[0]
    z = net(X[idx.reshape(-1)])
    return z.view(B, BAG).mean(dim=1)


@torch.no_grad()
def part_logit(net, X, pools, parts):
    net.eval()
    out = {}
    for p in parts:
        pl = pools[p]
        out[p] = float(net(X[pl]).mean()) if len(pl) else np.nan
    return out


def fit(X, pools, ytab, tr_parts, va_parts, epochs, seed, patience=8, bs=32):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net = Net()
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-2)
    steps = epochs * int(np.ceil(2 * len(tr_parts) / bs))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-3, total_steps=steps)
    ytr = np.array([ytab[p] for p in tr_parts])
    pw = torch.tensor((len(ytr) - ytr.sum()) / max(ytr.sum(), 1), dtype=torch.float32)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pw)
    yva = np.array([ytab[p] for p in va_parts], dtype=np.float32)
    best, best_state, bad = 1e9, None, 0
    for ep in range(epochs):
        net.train()
        order = np.concatenate([rng.permutation(len(tr_parts)) for _ in range(2)])   # 2 kantong/peserta/epoch
        for i in range(0, len(order), bs):
            sel = [tr_parts[j] for j in order[i:i + bs]]
            idx = make_bags(pools, sel, rng)
            xb = augment(X[idx.reshape(-1)])
            yb = torch.tensor([ytab[p] for p in sel], dtype=torch.float32)
            opt.zero_grad()
            z = net(xb).view(len(sel), BAG).mean(dim=1)
            lossf(z, yb).backward()
            opt.step(); sched.step()
        pl = part_logit(net, X, pools, va_parts)
        zv = torch.tensor([pl[p] for p in va_parts])
        vl = float(nn.BCEWithLogitsLoss(pos_weight=pw)(zv, torch.from_numpy(yva)))
        if vl < best - 1e-4:
            best, bad = vl, 0
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    net.load_state_dict(best_state)
    return net, ep + 1, best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="v2")
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--perm", type=int, default=0, help="jumlah permutasi label (dalam klinik); 0 = label asli")
    ap.add_argument("--out", default="mil_v2.npz")
    ap.add_argument("--feat", default="x_w24")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    fd = paths.features_dir()
    Z = np.load(fd / f"feats_{a.tag}.npz")
    meta = pd.read_csv(fd / f"meta_{a.tag}.csv")
    sp = pd.read_csv(paths.processed_dir() / f"splits_{a.tag}.csv")
    X = torch.from_numpy(Z[a.feat].astype(np.float32) / SCALE).permute(0, 2, 1).unsqueeze(1).contiguous()
    parts = sp.barcode.tolist()
    pools_all = {p: np.where(meta.barcode.values == p)[0] for p in parts}
    dev = meta.device.values
    devs = ["all"] + sorted(set(dev))
    pools = {d: {p: (pools_all[p] if d == "all" else pools_all[p][dev[pools_all[p]] == d]) for p in parts} for d in devs}
    y0 = dict(zip(sp.barcode, sp.y.astype(int)))
    fac = dict(zip(meta.barcode, meta.facility))
    fcols = [c for c in sp.columns if c.startswith("fold")]
    if "fold0" in fcols:
        fcols = [c for c in fcols if c != "fold"]
    n_runs = max(a.perm, 1)
    res = {}
    t0 = time.time()
    for run in range(n_runs):
        if a.perm:
            rng = np.random.default_rng(1000 + run)
            ytab = dict(y0)
            fa = pd.Series(fac)
            for f_, grp in fa.groupby(fa):
                ks = list(grp.index)
                vals = rng.permutation([y0[k] for k in ks])
                for k, v in zip(ks, vals):
                    ytab[k] = int(v)
        else:
            ytab = y0
        yarr = np.array([ytab[p] for p in parts])
        for r, fc in enumerate(fcols):
            fold = sp[fc].values
            score = {d: np.full(len(parts), np.nan) for d in devs}
            for k in range(5):
                te = np.where(fold == k)[0]; trp = np.where(fold != k)[0]
                sss = StratifiedShuffleSplit(1, test_size=0.15, random_state=10 * r + k)
                a_, b_ = next(sss.split(trp, yarr[trp]))
                tr_parts = [parts[i] for i in trp[a_]]; va_parts = [parts[i] for i in trp[b_]]
                te_parts = [parts[i] for i in te]
                zs = {d: np.zeros(len(te)) for d in devs}
                for s in range(a.seeds):
                    net, ne, bl = fit(X, pools["all"], ytab, tr_parts, va_parts, a.epochs, seed=100 * r + 10 * k + s)
                    for d in devs:
                        pl = part_logit(net, X, pools[d], te_parts)
                        zs[d] += np.array([pl[p] for p in te_parts]) / a.seeds
                for d in devs:
                    score[d][te] = zs[d]
                print(f"run{run} rep{r} lipatan{k} ep={ne} val={bl:.3f} ({time.time() - t0:.0f}s)", flush=True)
            for d in devs:
                m = ~np.isnan(score[d])
                au = roc_auc_score(yarr[m], score[d][m]) if len(set(yarr[m])) == 2 else np.nan
                print(f"run{run} rep{r} {d}: AUROC peserta {au:.3f} (n={m.sum()})", flush=True)
                res[f"run{run}_rep{r}_{d}"] = score[d]
        res[f"run{run}_y"] = yarr
    rd = paths.runs_dir(); rd.mkdir(parents=True, exist_ok=True)
    np.savez(rd / a.out, barcode=np.array(parts), devs=np.array(devs), **res)
    print("tersimpan", rd / a.out)


if __name__ == "__main__":
    main()
