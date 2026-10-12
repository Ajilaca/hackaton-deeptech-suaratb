"""Anggaran memori / pengali / siklus untuk konfigurasi (P, C1..C3, FFT_W) dan penilaian muat terhadap perangkat.

    python -m golden.analysis.budget                # tabel konfigurasi default + sapuan P
    python -m golden.analysis.budget --json         # keluaran mesin

SEMUA angka di sini adalah hitungan ANALITIK dari parameter (bit logis / jumlah pengali /
siklus ideal). Bukan hasil sintesis. Angka sumber daya nyata hanya dari laporan alat (Fase 6).
Kapasitas perangkat: DARI INGATAN, VERIFIKASI DI DATASHEET sebelum dipakai di proposal.
"""
from __future__ import annotations
import argparse
import json
import math
from dataclasses import dataclass
from golden import params

DEVICES_NOTE = "dari ingatan, verifikasi di datasheet"
# kapasitas: lut (LUT/LE), mem_bits (total bit blok RAM), blk_bits (ukuran satu blok), n_blk,
# dsp_blocks (jumlah blok), mult18 (pengali 18x18 yang tersedia), dsp_a/dsp_b (lebar port bertanda)
DEVICES = {
    "XC7A35T":  dict(lut=20800, mem_bits=50 * 36864, blk_bits=36864, n_blk=50, mult18=90, dsp_a=25, dsp_b=18,
                     fam="Artix-7 (DSP48E1 25x18; BRAM36)"),
    "EP4CE22":  dict(lut=22320, mem_bits=608256, blk_bits=9216, n_blk=66, mult18=66, dsp_a=18, dsp_b=18,
                     fam="Cyclone IV E (M9K; mult 18x18)"),
    "EP4CE6":   dict(lut=6272, mem_bits=276480, blk_bits=9216, n_blk=30, mult18=15, dsp_a=18, dsp_b=18,
                     fam="Cyclone IV E (M9K; mult 18x18)"),
    "ECP5-25F": dict(lut=24000, mem_bits=1032192, blk_bits=18432, n_blk=56, mult18=56, dsp_a=18, dsp_b=18,
                     fam="ECP5 LFE5U-25 (EBR 18Kb; sysDSP 18x18)"),
}
LUT_EST = (4000, 8000)  # estimasi awal proyek: estimasi kasar, keyakinan RENDAH (faktor 2). Bukan dari alat.


def n_tiles(bits_signed: int, port: int) -> int:
    """Jumlah irisan agar operan bertanda `bits_signed` bit muat port `port` (irisan bawah tak bertanda: port-1 bit)."""
    if bits_signed <= port:
        return 1
    return 1 + math.ceil((bits_signed - port) / (port - 1))


def mult_cost(a_bits: int, b_bits: int, a_signed: bool, b_signed: bool, pa: int, pb: int) -> int:
    ea = a_bits + (0 if a_signed else 1)
    eb = b_bits + (0 if b_signed else 1)
    if ea < eb:
        ea, eb = eb, ea
    return n_tiles(ea, max(pa, pb)) * n_tiles(eb, min(pa, pb))


@dataclass
class Mem:
    name: str
    bits: int
    width: int
    depth: int
    kind: str  # 'ROM' | 'RAM'


def memories(P: int = params.P, W: int = params.FFT_W, c=(params.C1, params.C2, params.C3),
             hidden: int = params.FC_HIDDEN, fuse_pool: bool = True, feat_double_buffer: bool = False):
    L = params.layer_descs(*c, hidden=hidden)
    m: list[Mem] = []
    # --- bobot: lebar kata P*8 (satu bobot per PE per siklus), kedalaman = sum ceil(cout/P)*cin*k*k
    depth = sum(math.ceil(l["cout"] / P) * l["cin"] * l["k"] * l["k"] for l in L)
    m.append(Mem("weight_rom", depth * P * params.WGT_BITS, P * params.WGT_BITS, depth, "ROM"))
    nb = params.n_biases(c1=c[0], c2=c[1], c3=c[2], hidden=hidden)
    m.append(Mem("bias_rom", nb * params.BIAS_BITS, params.BIAS_BITS, nb, "ROM"))
    m.append(Mem("requant_rom", len(L) * (params.REQ_M0_BITS + 5), params.REQ_M0_BITS + 5, len(L), "ROM"))
    # --- buffer aktivasi A/B (ping-pong); jika pool tidak digabung simpan keluaran pra-pool
    sizes_in = []   # ukuran aktivasi (elemen) yang disimpan setelah tiap layer
    for l in L:
        if l["kind"] != "conv":
            continue
        post = l["cout"] * (l["h"] // 2) * (l["w"] // 2) if l["pool"] else l["cout"] * l["h"] * l["w"]
        pre = l["cout"] * l["h"] * l["w"]
        sizes_in.append(post if fuse_pool else pre)
    inp = L[0]["cin"] * L[0]["h"] * L[0]["w"]
    chain = [inp] + sizes_in                 # masukan layer1, keluaran layer1, ...
    a_sz = max(chain[0::2]); b_sz = max(chain[1::2])
    m.append(Mem("act_buf_A", a_sz * params.ACT_BITS, params.ACT_BITS, a_sz, "RAM"))
    m.append(Mem("act_buf_B", b_sz * params.ACT_BITS, params.ACT_BITS, b_sz, "RAM"))
    # --- fitur: ring N_FRAMES x N_MEL int8 (gandakan bila feat_buf terpisah dari buffer A)
    fb = params.N_FRAMES * params.N_MEL
    for i in range(2 if feat_double_buffer else 1):
        m.append(Mem(f"feat_buf{i}", fb * params.ACT_BITS, params.ACT_BITS, fb, "RAM"))
    # --- front-end
    m.append(Mem("sample_ring", params.NFFT * 16, 16, params.NFFT, "RAM"))             # 400 sampel (+margin) pra-emphasis
    m.append(Mem("fft_ram", params.NFFT * 2 * W, 2 * W, params.NFFT, "RAM"))            # re+im in-place
    m.append(Mem("twiddle_rom", (params.NFFT // 2) * 2 * params.TW_FRAC + (params.NFFT // 2) * 2, 32, params.NFFT // 2, "ROM"))
    m.append(Mem("window_rom", (params.FRAME_LEN // 2) * 16, 16, params.FRAME_LEN // 2, "ROM"))
    from golden.gen.gen_tables import mel_rom_words
    nmel = len(mel_rom_words())
    m.append(Mem("mel_rom", nmel * 19, 19, nmel, "ROM"))
    m.append(Mem("mel_acc", params.N_MEL * params.MEL_ACC_BITS, params.MEL_ACC_BITS, params.N_MEL, "RAM"))
    m.append(Mem("log_lut", 256 * 8, 8, 256, "ROM"))
    return m


def multipliers(P: int = params.P, W: int = params.FFT_W, dev: dict | None = None):
    """Pengali per fungsi, sesuai tombol port perangkat (dsp_a x dsp_b). Satuan: blok pengali port-penuh."""
    pa, pb = (25, 18) if dev is None else (dev["dsp_a"], dev["dsp_b"])
    items = {
        "preemph (16b x Q15)": (1, mult_cost(16, 16, True, True, pa, pb)),
        "window (16+G b x Q15)": (1, mult_cost(16 + params.PREEMPH_FRAC, 16, True, True, pa, pb)),
        "fft butterfly (W x Q15) x4": (4, mult_cost(W, 16, True, True, pa, pb)),
        "power (W x W) x2": (2, mult_cost(W, W, True, True, pa, pb)),
        "mel MAC (P_bits x w12)": (1, mult_cost(params.POWER_BITS, params.MEL_W_FRAC, False, False, pa, pb)),
        "cough energy (16b x 16b)": (1, mult_cost(16, 16, True, True, pa, pb)),
        f"pe_array (8b x 8b) x{P}": (P, mult_cost(8, 8, True, True, pa, pb)),
        "requant (24b x M0 16b)": (1, mult_cost(params.ACC_BITS, params.REQ_M0_BITS, True, False, pa, pb)),
        "gap (12b x recip)": (1, mult_cost(12, 11, False, False, pa, pb)),
    }
    return {k: n * c for k, (n, c) in items.items()}


def cycles_cnn(P: int = params.P, c=(params.C1, params.C2, params.C3), hidden: int = params.FC_HIDDEN) -> dict:
    """Siklus ideal output-stationary: per layer ceil(cout/P) * H*W * cin*k*k. +~10% overhead belum termasuk."""
    per = {}
    for l in params.layer_descs(*c, hidden=hidden):
        per[l["name"]] = math.ceil(l["cout"] / P) * l["h"] * l["w"] * l["cin"] * l["k"] * l["k"]
    per["total"] = sum(per.values())
    return per


def frontend_cycles_per_frame(W: int = params.FFT_W) -> dict:
    """Perkiraan kasar (1 butterfly, 1 butterfly/siklus): FFT (N/2)*log2N + overhead."""
    n = params.NFFT
    fft = (n // 2) * 9 + 9 * 8
    return dict(fft=fft, window=params.NFFT, power=params.N_BINS, mel=len(__import__("golden.gen.gen_tables", fromlist=["x"]).mel_rom_words()),
                log=params.N_MEL, total_est=fft + params.NFFT + params.N_BINS + 500 + params.N_MEL * 2)


def assess(mems, mults_by_dev, dev_name: str, lut_est=LUT_EST):
    d = DEVICES[dev_name]
    bits = sum(m.bits for m in mems)
    blocks = sum(math.ceil(m.bits / d["blk_bits"]) for m in mems)   # tiap memori minimal 1 blok
    dsp = mults_by_dev[dev_name]
    mem_ok = bits <= d["mem_bits"] and blocks <= d["n_blk"]
    dsp_ok = dsp <= d["mult18"] if d["dsp_a"] == 18 else dsp <= d["mult18"]
    lut_ok = lut_est[1] <= d["lut"]
    lut_maybe = lut_est[0] <= d["lut"]
    return dict(device=dev_name, mem_bits=bits, mem_cap=d["mem_bits"], mem_pct=100.0 * bits / d["mem_bits"],
                blocks=blocks, blocks_cap=d["n_blk"], mult_blocks=dsp, mult_cap=d["mult18"],
                mem_ok=mem_ok, dsp_ok=dsp_ok, lut_ok="ya" if lut_ok else ("mungkin" if lut_maybe else "tidak"),
                fits=mem_ok and dsp_ok and lut_ok, fits_maybe=mem_ok and dsp_ok and lut_maybe)


def report(P: int = params.P, W: int = params.FFT_W, c=(params.C1, params.C2, params.C3),
           hidden: int = params.FC_HIDDEN) -> dict:
    mems = memories(P, W, c, hidden)
    mult = {n: sum(multipliers(P, W, d).values()) for n, d in DEVICES.items()}
    mult_detail = {n: multipliers(P, W, d) for n, d in DEVICES.items()}
    return dict(P=P, W=W, c=list(c), hidden=hidden,
                weights=params.n_weights(c1=c[0], c2=c[1], c3=c[2], hidden=hidden),
                macs=params.n_macs(c1=c[0], c2=c[1], c3=c[2], hidden=hidden),
                mems=[m.__dict__ for m in mems], mem_bits=sum(m.bits for m in mems),
                mult_detail=mult_detail, mult_total=mult, cycles=cycles_cnn(P, c, hidden),
                frontend_cycles=frontend_cycles_per_frame(W),
                devices={n: assess(mems, mult, n) for n in DEVICES}, devices_note=DEVICES_NOTE)


def fmt_markdown(r: dict) -> str:
    o = [f"### Konfigurasi P={r['P']}, FFT_W={r['W']}, kanal={r['c']}, FC={r['hidden']}",
         f"Bobot: {r['weights']}  MAC/inferensi: {r['macs']:,}  (analitik)", "",
         "| Memori | kata x lebar | bit |", "|---|---|---|"]
    for m in r["mems"]:
        o.append(f"| {m['name']} ({m['kind']}) | {m['depth']} x {m['width']} | {m['bits']:,} |")
    o.append(f"| **total (logis)** | | **{r['mem_bits']:,}** ({r['mem_bits']/1024:.0f} Kbit) |")
    o += ["", "Siklus CNN ideal: " + ", ".join(f"{k}={v:,}" for k, v in r["cycles"].items()),
          f"Latensi CNN ideal @100 MHz: {r['cycles']['total']/100e6*1e3:.2f} ms; "
          f"@150 MHz: {r['cycles']['total']/150e6*1e3:.2f} ms (target/estimasi, belum diukur)", "",
          f"Kapasitas perangkat: {r['devices_note']}.", "",
          "| Perangkat | memori logis / kapasitas | blok | pengali / tersedia | LUT (est. 4-8 rb) | memori | pengali | muat? |",
          "|---|---|---|---|---|---|---|---|"]
    for n, a in r["devices"].items():
        verdict = "ya (estimasi)" if a["fits"] else ("mungkin" if a["fits_maybe"] else "TIDAK")
        o.append(f"| {n} | {a['mem_bits']:,} / {a['mem_cap']:,} ({a['mem_pct']:.0f}%) | {a['blocks']}/{a['blocks_cap']} "
                 f"| {a['mult_blocks']}/{a['mult_cap']} | {a['lut_ok']} | {'ok' if a['mem_ok'] else 'LEWAT'} "
                 f"| {'ok' if a['dsp_ok'] else 'LEWAT'} | {verdict} |")
    return "\n".join(o)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--P", type=int, default=params.P)
    ap.add_argument("--W", type=int, default=params.FFT_W)
    a = ap.parse_args()
    reps = [report(P=a.P, W=a.W)]
    if not a.json:
        reps += [report(P=p, W=a.W) for p in (8, 32) if p != a.P]
        reps.append(report(P=8, W=a.W, c=(8, 16, 16)))
        reps.append(report(P=8, W=18, c=(8, 16, 16)))
    if a.json:
        print(json.dumps(reps[0], indent=1))
    else:
        print("\n\n".join(fmt_markdown(r) for r in reps))


if __name__ == "__main__":
    main()
