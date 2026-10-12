"""Fase M: CNN float terlatih -> int8 (PTQ, lalu QAT bila perlu) -> QModel golden + tabel requant + laporan.

    python -m model.scratch.export --tag v2 --cv        # evaluasi kuantisasi per lipatan (float vs golden int8)
    python -m model.scratch.export --tag v2 --final     # latih model akhir pada semua peserta dan ekspor bobot

Aritmatika target = golden/blocks/cnn.py (akumulator 24 bit, requant (acc*M0 + 2^(s-1)) >> s, ReLU+sat [0,127]).
Bobot hasil ekspor ditulis ke model/out/model_<tag>/ (symlink -> $SUARATB_DATA/runs, di-.gitignore): JANGAN di-commit.
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
from sklearn.model_selection import StratifiedShuffleSplit
from golden import params, paths
from golden.blocks import cnn as C
from golden.io_mem import write_mem
from model.scratch import mil_train as M

S_IN = 1.0 / M.SCALE            # real = q * S_IN
LOGIT_SCALE = 1.0 / 256.0       # logit int16 = real_logit * 256


def float_layers(net: M.Net):
    f, h = net.f, net.h
    return [(f[0], f[3], f[6]), (h[1], h[3])]


@torch.no_grad()
def act_scales(net: M.Net, X: torch.Tensor, pct: float = 99.9):
    """Skala aktivasi pasca-ReLU tiap layer tersembunyi: persentil/127."""
    net.eval()
    a, sc = [], []
    h = X
    f = net.f
    for i in range(0, 9, 3):
        h = f[i + 1](f[i](h))                      # conv + ReLU
        sc.append(float(np.percentile(h.numpy(), pct)) / 127.0)
        h = f[i + 2](h)                            # pool
    g = h.mean(dim=(2, 3))
    z = net.h[2](net.h[1](g))                      # fc1 + ReLU
    sc.append(float(np.percentile(z.numpy(), pct)) / 127.0)
    return [max(s, 1e-6) for s in sc]


def ptq(net: M.Net, Xcal: torch.Tensor) -> C.QModel:
    sc = act_scales(net, Xcal)
    f, h = net.f, net.h
    convs = [f[0], f[3], f[6]]
    fcs = [h[1], h[3]]
    qm = C.QModel()
    s_prev = S_IN
    descs = params.layer_descs()
    for i, conv in enumerate(convs):
        W = conv.weight.detach().numpy(); b = conv.bias.detach().numpy()
        sw = max(float(np.abs(W).max()), 1e-8) / 127.0
        wq = np.clip(np.rint(W / sw), -128, 127).astype(np.int8)
        bq = np.rint(b / (s_prev * sw)).astype(np.int64)
        m0, sh = C.m0_shift_from_scale(s_prev * sw / sc[i])
        qm.layers.append(C.QLayer(descs[i]["name"], "conv", wq, bq.astype(np.int32), m0, sh, pool=True))
        s_prev = sc[i]
    for j, fc in enumerate(fcs):
        W = fc.weight.detach().numpy(); b = fc.bias.detach().numpy()
        sw = max(float(np.abs(W).max()), 1e-8) / 127.0
        wq = np.clip(np.rint(W / sw), -128, 127).astype(np.int8)
        bq = np.rint(b / (s_prev * sw)).astype(np.int64)
        last = j == 1
        s_out = LOGIT_SCALE if last else sc[3]
        m0, sh = C.m0_shift_from_scale(s_prev * sw / s_out)
        qm.layers.append(C.QLayer(descs[3 + j]["name"], "fc", wq, bq.astype(np.int32), m0, sh,
                                  final=last, gap_hw=(descs[3]["gap_hw"] if j == 0 else 0)))
        s_prev = s_out
    for L in qm.layers:
        assert np.abs(L.b).max() < (1 << (params.ACC_BITS - 1)), "bias di luar 24 bit"
    return qm


def int_logits(qm: C.QModel, x_int8: np.ndarray, stats: dict | None = None) -> np.ndarray:
    """x (N,50,40) int8 -> logit nyata (N,) dari golden cnn_forward."""
    out = np.zeros(len(x_int8))
    for i in range(len(x_int8)):
        xi = x_int8[i].astype(np.int64).T[None, :, :]
        out[i] = float(C.cnn_forward(xi, qm, stats)[0]) * LOGIT_SCALE
    return out


def part_mean(logits: np.ndarray, barcodes: np.ndarray, parts) -> np.ndarray:
    s = pd.Series(logits).groupby(barcodes).mean()
    return s.reindex(parts).values


# ---- QAT (STE) -------------------------------------------------------------------

def _fq(x, s, lo, hi):
    q = torch.clamp(torch.round(x / s), lo, hi) * s
    return x + (q - x).detach()


def qat_forward(net: M.Net, x: torch.Tensor, sc: list[float]) -> torch.Tensor:
    f, h = net.f, net.h
    a = x
    convs = [f[0], f[3], f[6]]
    for i, conv in enumerate(convs):
        sw = conv.weight.detach().abs().max().clamp_min(1e-8) / 127.0
        w = _fq(conv.weight, sw, -128, 127)
        a = torch.nn.functional.conv2d(a, w, conv.bias, padding=1)
        a = torch.relu(a)
        a = _fq(a, sc[i], 0, 127)
        a = torch.nn.functional.max_pool2d(a, 2)
    g = a.mean(dim=(2, 3))
    fc1, fc2 = h[1], h[3]
    sw = fc1.weight.detach().abs().max().clamp_min(1e-8) / 127.0
    z = torch.relu(torch.nn.functional.linear(g, _fq(fc1.weight, sw, -128, 127), fc1.bias))
    z = _fq(z, sc[3], 0, 127)
    sw = fc2.weight.detach().abs().max().clamp_min(1e-8) / 127.0
    return torch.nn.functional.linear(z, _fq(fc2.weight, sw, -128, 127), fc2.bias).squeeze(1)


def qat_finetune(net, X, pools, ytab, tr_parts, sc, epochs=6, seed=0):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    opt = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=1e-2)
    ytr = np.array([ytab[p] for p in tr_parts])
    lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor((len(ytr) - ytr.sum()) / max(ytr.sum(), 1), dtype=torch.float32))
    for ep in range(epochs):
        net.train()
        order = rng.permutation(len(tr_parts))
        for i in range(0, len(order), 32):
            sel = [tr_parts[j] for j in order[i:i + 32]]
            idx = M.make_bags(pools, sel, rng)
            xb = M.augment(X[idx.reshape(-1)])
            yb = torch.tensor([ytab[p] for p in sel], dtype=torch.float32)
            opt.zero_grad()
            z = qat_forward(net, xb, sc).view(len(sel), M.BAG).mean(dim=1)
            lossf(z, yb).backward()
            opt.step()
    return net


# ---- driver ------------------------------------------------------------------------

def load(tag):
    fd = paths.features_dir()
    Z = np.load(fd / f"feats_{tag}.npz")
    meta = pd.read_csv(fd / f"meta_{tag}.csv")
    sp = pd.read_csv(paths.processed_dir() / f"splits_{tag}.csv")
    x = Z["x_w24"]
    X = torch.from_numpy(x.astype(np.float32) / M.SCALE).permute(0, 2, 1).unsqueeze(1).contiguous()
    parts = sp.barcode.tolist()
    pools = {p: np.where(meta.barcode.values == p)[0] for p in parts}
    ytab = dict(zip(sp.barcode, sp.y.astype(int)))
    return x, X, meta, sp, parts, pools, ytab


def cv_eval(a):
    x, X, meta, sp, parts, pools, ytab = load(a.tag)
    y = np.array([ytab[p] for p in parts])
    fold = sp["fold0" if "fold0" in sp else "fold"].values
    bc = meta.barcode.values
    sf, si, sq = np.full(len(parts), np.nan), np.full(len(parts), np.nan), np.full(len(parts), np.nan)
    rows = []
    t0 = time.time()
    for k in range(5):
        te = np.where(fold == k)[0]; trp = np.where(fold != k)[0]
        a_, b_ = next(StratifiedShuffleSplit(1, test_size=0.15, random_state=k).split(trp, y[trp]))
        trn = [parts[i] for i in trp[a_]]; van = [parts[i] for i in trp[b_]]
        net, ne, bl = M.fit(X, pools, ytab, trn, van, a.epochs, seed=k)
        calib = X[np.concatenate([pools[p] for p in trn])]
        qm = ptq(net, calib)
        te_idx = np.concatenate([pools[parts[i]] for i in te])
        with torch.no_grad():
            net.eval(); zf = net(X[te_idx]).numpy()
        zi = int_logits(qm, x[te_idx])
        corr = float(np.corrcoef(zf, zi)[0, 1])
        qat_used = False
        if corr < a.corr_min:
            sc = act_scales(net, calib)
            net = qat_finetune(net, X, pools, ytab, trn, sc, epochs=a.qat_epochs, seed=k)
            qm = ptq(net, calib)
            zi = int_logits(qm, x[te_idx])
            with torch.no_grad():
                net.eval(); zf = net(X[te_idx]).numpy()
            corr = float(np.corrcoef(zf, zi)[0, 1]); qat_used = True
        tp = [parts[i] for i in te]
        sf[te] = part_mean(zf, bc[te_idx], tp); si[te] = part_mean(zi, bc[te_idx], tp)
        rows.append((k, ne, corr, float(np.sqrt(np.mean((zf - zi) ** 2))), qat_used))
        print(f"lipatan {k}: ep={ne} corr(float,int8)={corr:.4f} rmse={rows[-1][3]:.3f} qat={qat_used} ({time.time() - t0:.0f}s)", flush=True)
    af, ai = roc_auc_score(y, sf), roc_auc_score(y, si)
    return rows, af, ai, y, sf, si


def final_train(a):
    x, X, meta, sp, parts, pools, ytab = load(a.tag)
    y = np.array([ytab[p] for p in parts])
    a_, b_ = next(StratifiedShuffleSplit(1, test_size=0.15, random_state=123).split(np.arange(len(parts)), y))
    trn = [parts[i] for i in a_]; van = [parts[i] for i in b_]
    best = None
    for s in range(a.seeds):
        net, ne, bl = M.fit(X, pools, ytab, trn, van, a.epochs, seed=1000 + s)
        print(f"seed {s}: ep={ne} val_loss={bl:.4f}", flush=True)
        if best is None or bl < best[1]:
            best = (net, bl, s)
    net = best[0]
    calib = X[np.concatenate([pools[p] for p in parts])]
    qm = ptq(net, calib)
    stats = {}
    idx = np.concatenate([pools[p] for p in parts])
    zi = int_logits(qm, x[idx[::5]], stats)
    with torch.no_grad():
        net.eval(); zf = net(X[idx[::5]]).numpy()
    corr = float(np.corrcoef(zf, zi)[0, 1])
    if corr < a.corr_min:
        sc = act_scales(net, calib)
        net = qat_finetune(net, X, pools, ytab, trn, sc, epochs=a.qat_epochs)
        qm = ptq(net, calib)
        zi = int_logits(qm, x[idx[::5]], stats)
        with torch.no_grad():
            net.eval(); zf = net(X[idx[::5]]).numpy()
        corr = float(np.corrcoef(zf, zi)[0, 1])
    out = paths.runs_dir() / f"model_{a.tag}"
    out.mkdir(parents=True, exist_ok=True)
    torch.save(net.state_dict(), out / "float.pt")
    C.save_qmodel(qm, out / "qmodel.npz")
    for L in qm.layers:
        write_mem(out / f"{L.name}_w.mem", L.w.ravel(), 8, f"{L.name} bobot int8 (cout,cin[,3,3]) row-major")
        write_mem(out / f"{L.name}_b.mem", L.b, params.BIAS_BITS, f"{L.name} bias int24")
    write_mem(out / "requant.mem", [v for L in qm.layers for v in (L.m0, L.shift)], 16, "per layer: M0(Q15), shift")
    return qm, stats, corr, best[2], out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="v2")
    ap.add_argument("--cv", action="store_true")
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--corr-min", type=float, default=0.985)
    ap.add_argument("--qat-epochs", type=int, default=6)
    ap.add_argument("--out", default=str(paths.runs_dir() / "reports" / "model-fase-m.md"))
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    o = ["# Fase M: model int8 (golden) dan kuantisasi", "",
         f"`python -m model.scratch.export --tag {a.tag}`. Arsitektur = golden (Conv 16/32/32, GAP, FC 32->16->1, 14.496 bobot). "
         "Bobot terlatih TIDAK di-commit (disimpan di `model/out/model_<tag>/`).", ""]
    if a.cv:
        rows, af, ai, y, sf, si = cv_eval(a)
        o += ["## Kuantisasi per lipatan (golden bit-exact vs float, uji out-of-fold)",
              "| lipatan | epoch | korelasi logit | RMSE logit | QAT dipakai |", "|---|---|---|---|---|"]
        for k, ne, c, r, q in rows:
            o.append(f"| {k} | {ne} | {c:.4f} | {r:.3f} | {'ya' if q else 'tidak'} |")
        o += ["", f"AUROC peserta (satu seed, satu ulangan): float {af:.3f}, int8 golden {ai:.3f}, selisih {ai - af:+.3f} "
              f"(target penurunan ≤ 0,02: {'LULUS' if af - ai <= 0.02 else 'GAGAL'}).", ""]
    if a.final:
        qm, st, corr, seed, out = final_train(a)
        o += ["## Model akhir", f"- seed terpilih {seed} (val loss terendah); korelasi logit float vs int8 {corr:.4f}",
              f"- bobot/tabel: `{out}`", "",
              "| layer | bobot | M0 (Q15) | shift | puncak |acc| teramati | maks |bias| |", "|---|---|---|---|---|---|"]
        for L in qm.layers:
            o.append(f"| {L.name} | {L.w.size} | {L.m0} | {L.shift} | {st.get(L.name, 0)} | {int(np.abs(L.b).max())} |")
        o += ["", f"Akumulator 24 bit: batas {1 << 23}; semua puncak di bawahnya (assert di golden)."]
    pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True); pathlib.Path(a.out).write_text("\n".join(o) + "\n")
    print("\n".join(o))


if __name__ == "__main__":
    main()
