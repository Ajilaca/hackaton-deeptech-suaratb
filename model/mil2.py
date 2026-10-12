"""Butir 2: CNN MIL diperbaiki: kurangi derau pra-onset, hanya klip mirip-batuk, epoch tetap (tanpa early stopping
berisik), tugas bantu (usia/jenis kelamin/HIV/TB sebelumnya/lama batuk/demam/keringat malam/berat badan turun),
ensemble seed. Arsitektur inferensi identik dengan golden (kepala TB = rata-rata logit klip).

    python -m model.mil2 --tag v2 --aux 1 --reps 2 --seeds 2 --epochs 14 --perm 0 --out mil2_aux1.npz
"""
from __future__ import annotations
import argparse
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from golden import paths
from model.data import dataset
from model.mil_train import augment, BAG

SCALE = 64.0
AUXC = ["sex_m", "hiv_pos", "prev_tb", "fever", "ngtsweats", "weightloss", "cough_productive", "haemoptysis"]
AUXR = ["age_z", "coughdur_z"]


class Net2(nn.Module):
    def __init__(self, n_aux, c=(16, 32, 32), h=16, drop=0.25):
        super().__init__()
        L, cin = [], 1
        for co in c:
            L += [nn.Conv2d(cin, co, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)]
            cin = co
        self.f = nn.Sequential(*L)
        self.d = nn.Dropout(drop)
        self.fc1 = nn.Linear(cin, h)
        self.tb = nn.Linear(h, 1)
        self.aux = nn.Linear(h, max(n_aux, 1))

    def embed(self, x):
        return torch.relu(self.fc1(self.d(self.f(x).mean(dim=(2, 3)))))

    def forward(self, x):
        return self.tb(self.embed(x)).squeeze(1)


def prep(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    return x - x[:, :4, :].mean(axis=1, keepdims=True)           # kurangi derau pra-onset per band


def aux_table(parts):
    m = dataset.load_meta().set_index("barcode").loc[list(parts)]
    dur = {"no cough": 0, "<1wk": 1, "1-2wks": 2, "3-4wks": 3, ">4weeks": 4}
    d = pd.DataFrame(index=m.index)
    s = lambda c: m[c].astype(str).str.lower()
    d["sex_m"] = (s("sex") == "m").astype(float)
    d["hiv_pos"] = np.where(s("hiv_status").isin(["pos", "neg"]), (s("hiv_status") == "pos").astype(float), np.nan)
    for c in ("prev_tb", "fever", "ngtsweats", "weightloss", "cough_productive", "haemoptysis"):
        d[c] = np.where(s(c).isin(["yes", "no"]), (s(c) == "yes").astype(float), np.nan)
    d["age_z"] = (m.age - m.age.mean()) / m.age.std()
    d["coughdur_z"] = m.coughdur.map(dur).astype(float)
    d["coughdur_z"] = (d.coughdur_z - d.coughdur_z.mean()) / d.coughdur_z.std()
    return d[AUXC].values.astype(np.float32), d[AUXR].values.astype(np.float32)


def fit(X, pools, ytab, tr_parts, epochs, seed, aux_c, aux_r, pidx, aux_on, bs=32):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    net = Net2(len(AUXC) + len(AUXR))
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-2)
    steps = epochs * int(np.ceil(2 * len(tr_parts) / bs))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-3, total_steps=steps)
    ytr = np.array([ytab[p] for p in tr_parts])
    lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor((len(ytr) - ytr.sum()) / max(ytr.sum(), 1), dtype=torch.float32))
    for ep in range(epochs):
        net.train()
        order = np.concatenate([rng.permutation(len(tr_parts)) for _ in range(2)])
        for i in range(0, len(order), bs):
            sel = [tr_parts[j] for j in order[i:i + bs]]
            idx = np.array([rng.choice(pools[p], BAG, replace=len(pools[p]) < BAG) for p in sel])
            xb = augment(X[idx.reshape(-1)])
            yb = torch.tensor([ytab[p] for p in sel], dtype=torch.float32)
            opt.zero_grad()
            e = net.embed(xb).view(len(sel), BAG, -1).mean(dim=1)
            loss = lossf(net.tb(e).squeeze(1), yb)
            if aux_on:
                pi = [pidx[p] for p in sel]
                ac = torch.from_numpy(aux_c[pi]); ar = torch.from_numpy(aux_r[pi])
                o = net.aux(e)
                mc = ~torch.isnan(ac)
                if mc.any():
                    loss = loss + 0.5 * nn.functional.binary_cross_entropy_with_logits(o[:, :len(AUXC)][mc], ac[mc])
                mr = ~torch.isnan(ar)
                if mr.any():
                    loss = loss + 0.5 * ((o[:, len(AUXC):][mr] - ar[mr]) ** 2).mean()
            loss.backward(); opt.step(); sched.step()
    return net


@torch.no_grad()
def part_score(net, X, pools, parts):
    net.eval()
    return np.array([float(net(X[pools[p]]).mean()) for p in parts])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="v2")
    ap.add_argument("--aux", type=int, default=1)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=14)
    ap.add_argument("--perm", type=int, default=0)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", default="mil2.npz")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    fd = paths.features_dir()
    Z = np.load(fd / f"feats_{a.tag}.npz"); meta = pd.read_csv(fd / f"meta_{a.tag}.csv")
    mask = np.load(fd / f"clipq_{a.tag}.npz")["cough_like"].astype(bool)
    sp = pd.read_csv(paths.processed_dir() / f"splits_{a.tag}.csv")
    parts = sp.barcode.tolist(); y0 = dict(zip(sp.barcode, sp.y.astype(int)))
    X = torch.from_numpy(prep(Z["x_w24"]) / SCALE).permute(0, 2, 1).unsqueeze(1).contiguous()
    by = meta.groupby("barcode").indices
    pools = {}
    for p in parts:
        idx = by[p]; sel = idx[mask[idx]]
        pools[p] = sel if len(sel) else idx
    aux_c, aux_r = aux_table(parts)
    pidx = {p: i for i, p in enumerate(parts)}
    fac = meta.drop_duplicates("barcode").set_index("barcode").facility.reindex(parts).values
    fcols = [c for c in sp.columns if c.startswith("fold") and c != "fold"][: a.reps] or ["fold"]
    res, t0 = {}, time.time()
    for run in range(max(a.perm, 1)):
        ytab = dict(y0)
        if a.perm:
            rng = np.random.default_rng(2000 + run)
            for c in np.unique(fac):
                ks = [p for p, f in zip(parts, fac) if f == c]
                for k, v in zip(ks, rng.permutation([y0[k] for k in ks])):
                    ytab[k] = int(v)
        yarr = np.array([ytab[p] for p in parts])
        for r, fc in enumerate(fcols):
            fold = sp[fc].values; score = np.zeros(len(parts))
            for k in range(5):
                te = np.where(fold == k)[0]; tr = [parts[i] for i in np.where(fold != k)[0]]
                z = np.zeros(len(te))
                for s in range(a.seeds):
                    net = fit(X, pools, ytab, tr, a.epochs, 100 * r + 10 * k + s, aux_c, aux_r, pidx, bool(a.aux))
                    z += part_score(net, X, pools, [parts[i] for i in te]) / a.seeds
                score[te] = z
            res[f"run{run}_rep{r}"] = score
            print(f"run{run} rep{r}: AUROC {roc_auc_score(yarr, score):.3f} ({time.time() - t0:.0f}s)", flush=True)
        res[f"run{run}_y"] = yarr
    np.savez(paths.runs_dir() / a.out, barcode=np.array(parts), **res)


if __name__ == "__main__":
    main()
