"""Paket konfigurasi CNN (SystemVerilog) dan ROM bobot/bias/requant yang dikemas untuk P lane.

    python -m golden.gen_cnn_rom --cfg golden/tables/cnn_cfg_pkg.sv          # konfigurasi dari params.layer_descs()
    python -m golden.gen_cnn_rom --model /data/suaratb/runs/model_v2/qmodel.npz --out <dir>   # ROM dari model terlatih

ROM bobot  : kata P*8 bit; urutan (layer, grup g, ci, tap); lane l = kanal keluaran g*P+l (0 bila di luar cout).
ROM bias   : kata P*BIAS_BITS bit; satu kata per (layer, grup).
ROM requant: 2 kata 16 bit per layer: M0 (Q15), shift.
Layer mesin = conv1, conv2, conv3, fc1, fc2 (GAP berdiri sendiri antara conv3 dan fc1).
Padanan RTL: rtl/cnn/*.sv; sumber tunggal bentuk layer = params.layer_descs().
"""
from __future__ import annotations
import argparse
import pathlib
import numpy as np
from . import params
from .blocks import cnn as C
from .io_mem import write_mem


def engine_layers(c1=None, c2=None, c3=None, hidden=None):
    d = params.layer_descs(c1=c1 or params.C1, c2=c2 or params.C2, c3=c3 or params.C3, hidden=hidden or params.FC_HIDDEN)
    return d


def shapes(P: int | None = None, **kw):
    """Bentuk per layer mesin + penempatan buffer + alamat ROM. Kembalikan (list[dict], info dict)."""
    P = params.P if P is None else P
    L = []
    wbase = bbase = 0
    for i, d in enumerate(engine_layers(**kw)):
        k = d["k"]
        T = k * k
        h, w = d["h"], d["w"]
        pool = bool(d["pool"])
        ho, wo = (h // 2, w // 2) if pool else (h, w)
        G = -(-d["cout"] // P)
        src = "AB"[i % 2] if i < 3 else ("A" if d["name"] == "fc1" else "B")
        dst = "BA"[i % 2] if i < 3 else ("B" if d["name"] == "fc1" else "-")
        L.append(dict(i=i, name=d["name"], kind=d["kind"], cin=d["cin"], cout=d["cout"], h=h, w=w, k=k, T=T, pool=pool,
                      ho=ho, wo=wo, G=G, src=src, dst=dst, final=(d["name"] == "fc2"), wbase=wbase, bbase=bbase))
        wbase += G * d["cin"] * T
        bbase += G
    gap_ch = L[2]["cout"]
    gap_hw = L[2]["ho"] * L[2]["wo"]
    # ukuran buffer A (masukan, keluaran layer ber-dst A, keluaran GAP) dan B
    size = {"A": L[0]["cin"] * L[0]["h"] * L[0]["w"], "B": 1}
    for l in L:
        if l["dst"] in "AB":
            size[l["dst"]] = max(size[l["dst"]], l["cout"] * l["ho"] * l["wo"])
    size["A"] = max(size["A"], gap_ch)
    pow2 = lambda n: 1 << (n - 1).bit_length()  # noqa: E731
    info = dict(NL=len(L), WROM_DEPTH=wbase, NBG=bbase, GAP_CH=gap_ch, GAP_HW=gap_hw,
                GAP_RECIP=int(np.floor((1 << params.GAP_RECIP_FRAC) / gap_hw + 0.5)),
                A_DEPTH=pow2(size["A"]), B_DEPTH=pow2(size["B"]), P=P)
    return L, info


def cfg_package(P: int | None = None, name: str = "cnn_cfg_pkg", **kw) -> str:
    L, info = shapes(P, **kw)
    o = [f"// Dihasilkan oleh golden/gen_cnn_rom.py dari params.layer_descs() (P={info['P']}). JANGAN diedit manual.",
         f"package {name};"]
    for k, v in info.items():
        o.append(f"  localparam int {k} = {v};")
    fields = ["cin", "cout", "h", "w", "k", "T", "ho", "wo", "G", "wbase", "bbase"]
    for f in fields:
        o.append(f"  function automatic int cfg_{f}(input int l);")
        o.append("    case (l)")
        for l in L:
            o.append(f"      {l['i']}: return {l[f]};")
        o.append("      default: return 0;")
        o.append("    endcase")
        o.append("  endfunction")
    for f, fn in (("pool", lambda l: int(l["pool"])), ("final", lambda l: int(l["final"])),
                  ("src_b", lambda l: int(l["src"] == "B")), ("dst_b", lambda l: int(l["dst"] == "B"))):
        o.append(f"  function automatic int cfg_{f}(input int l);")
        o.append("    case (l)")
        for l in L:
            o.append(f"      {l['i']}: return {fn(l)};")
        o.append("      default: return 0;")
        o.append("    endcase")
        o.append("  endfunction")
    o.append("endpackage")
    return "\n".join(o) + "\n"


def rom_words(model: C.QModel, P: int | None = None):
    """-> (wrom words list[int], bias words list[int], rq words list[int])."""
    P = params.P if P is None else P
    L, info = shapes(P)
    wr, br, rq = [], [], []
    bmask = (1 << params.BIAS_BITS) - 1
    for l, q in zip(L, model.layers):
        assert q.name == l["name"], (q.name, l["name"])
        w = q.w.astype(np.int64)
        if l["kind"] == "conv":
            w = w.reshape(l["cout"], l["cin"], 9)
        else:
            w = w.reshape(l["cout"], l["cin"], 1)
        for g in range(l["G"]):
            for ci in range(l["cin"]):
                for t in range(l["T"]):
                    word = 0
                    for ln in range(P):
                        co = g * P + ln
                        if co < l["cout"]:
                            word |= (int(w[co, ci, t]) & 0xFF) << (8 * ln)
                    wr.append(word)
            bw = 0
            for ln in range(P):
                co = g * P + ln
                if co < l["cout"]:
                    bw |= (int(q.b[co]) & bmask) << (params.BIAS_BITS * ln)
            br.append(bw)
        rq += [int(q.m0), int(q.shift)]
    assert len(wr) == info["WROM_DEPTH"] and len(br) == info["NBG"]
    return wr, br, rq


def write_roms(model: C.QModel, out: pathlib.Path, P: int | None = None) -> dict:
    P = params.P if P is None else P
    wr, br, rq = rom_words(model, P)
    out = pathlib.Path(out)

    def wmem(path, words, bits, hdr):
        digits = (bits + 3) // 4
        lines = [f"// {hdr}"] + [format(int(x) & ((1 << bits) - 1), f"0{digits}x") for x in words]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(lines) + "\n")
    wmem(out / "cnn_wrom.mem", wr, P * 8, f"bobot int8 dikemas {P} lane; urut (layer, g, ci, tap)")
    wmem(out / "cnn_bias.mem", br, P * params.BIAS_BITS, f"bias int{params.BIAS_BITS} dikemas {P} lane; satu kata per (layer, g)")
    write_mem(out / "cnn_rq.mem", rq, 16, "per layer: M0 (Q15), shift")
    return dict(wrom=len(wr), bias=len(br), rq=len(rq))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default=None, help="tulis paket konfigurasi ke berkas ini")
    ap.add_argument("--model", default=None, help="qmodel.npz")
    ap.add_argument("--out", default=None)
    ap.add_argument("--p", type=int, default=params.P)
    a = ap.parse_args()
    if a.cfg:
        pathlib.Path(a.cfg).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(a.cfg).write_text(cfg_package(a.p))
    if a.model:
        print(write_roms(C.load_qmodel(a.model), pathlib.Path(a.out), a.p))


if __name__ == "__main__":
    main()
