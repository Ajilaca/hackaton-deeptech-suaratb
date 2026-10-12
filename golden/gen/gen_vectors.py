"""Vektor uji per blok (SINTETIS: sinus, derau, impuls, klip batuk-sintetis; model CNN ACAK).

    python -m golden.gen.gen_vectors [--out tb/vectors]

Tiap blok: <blok>/<kasus>_in.mem, _out.mem, dan meta.json (lebar bit, jumlah kata, parameter).
Format = .mem hex komplemen-dua, satu kata per baris, siap $readmemh.
Vektor CNN memakai bobot ACAK deterministik (BUKAN terlatih); nilai requant dari kalibrasi sintetis.
"""
from __future__ import annotations
import argparse
import json
import pathlib
import numpy as np
from golden import params
from golden.gen import synth
from golden.gen.io_mem import write_mem
from golden.blocks import frontend as F
from golden.blocks.preemph import preemph
from golden.blocks.window import apply_window, hamming_q15
from golden.blocks.fft512 import fft512
from golden.blocks.power import power
from golden.blocks.mel import mel_energy
from golden.blocks.log_q import log2_q
from golden.blocks.quant import quant_feat
from golden.blocks import cnn as C
from golden.blocks.cough_det import hop_energy, cough_det

DEFAULT_OUT = pathlib.Path(__file__).resolve().parents[2] / "tb" / "vectors"


def test_signals():
    n = params.CLIP_SAMPLES_PADDED
    return {
        "sine1k": F.pad_clip(synth.clip(0, 2000.0, 0, "sine")),
        "noise3": F.pad_clip(synth.clip(1, 3.0, 0, "noise")),
        "noise100": F.pad_clip(synth.clip(2, 100.0, 0, "noise")),
        "impulse": F.pad_clip(synth.clip(3, 8000.0, 0, "impulse")),
        "cough": F.pad_clip(synth.clip(4, 10.0, 20.0, "cough")),
    }


def _meta(d: pathlib.Path, name: str, **kw):
    (d / f"{name}.json").write_text(json.dumps(kw, indent=1, default=int))


def main(out: pathlib.Path = DEFAULT_OUT) -> list[str]:
    out = pathlib.Path(out)
    W = params.FFT_W
    made = []
    sig = test_signals()
    for case, x in sig.items():
        x16 = x
        # preemph (aliran)
        d = out / "preemph"
        pf = params.PREEMPH_FRAC
        y = preemph(x16, frac=pf)
        write_mem(d / f"{case}_in.mem", x16, 16); write_mem(d / f"{case}_out.mem", y, 16 + pf)
        write_mem(d / f"{case}_out_s16.mem", preemph(x16), 16)          # jalur detektor (int16)
        # window + fft + power untuk satu frame representatif (frame 20)
        fr = F.frame_signal(y)[20]
        d = out / "window"
        win = apply_window(fr, W, in_frac=pf)
        write_mem(d / f"{case}_in.mem", fr, 16 + pf); write_mem(d / f"{case}_out.mem", win, W)
        d = out / "fft512"
        re, im, ns = fft512(win, W=W)
        write_mem(d / f"{case}_in.mem", win, W)
        write_mem(d / f"{case}_out_re.mem", re, W); write_mem(d / f"{case}_out_im.mem", im, W)
        d = out / "power"
        p = power(re, im)
        write_mem(d / f"{case}_in_re.mem", re[: params.N_BINS], W); write_mem(d / f"{case}_in_im.mem", im[: params.N_BINS], W)
        write_mem(d / f"{case}_out.mem", p, 2 * W)
        d = out / "mel"
        e = mel_energy(p)
        write_mem(d / f"{case}_in.mem", p, 2 * W); write_mem(d / f"{case}_out.mem", e, params.LOG_IN_BITS)
        d = out / "log_q"
        L = log2_q(e)
        write_mem(d / f"{case}_in.mem", e, params.LOG_IN_BITS); write_mem(d / f"{case}_out.mem", L, 14)
        d = out / "quant"
        q = quant_feat(L)
        write_mem(d / f"{case}_in.mem", L, 14); write_mem(d / f"{case}_out.mem", q, 8)
        # front-end penuh (semua frame)
        d = out / "frontend"
        Lall = F.log_mel_q(x16)
        write_mem(d / f"{case}_in.mem", x16, 16)
        write_mem(d / f"{case}_out_feat.mem", F.features_q(x16), 8)
        # frame_buf: 6 frame pertama (1200 sampel) + energi hop
        d = out / "frame_buf"
        nfr = 6
        nin = params.FRAME_LEN + params.HOP * (nfr - 1)
        fr_all = F.frame_signal(y)[:nfr]
        write_mem(d / f"{case}_in_wide.mem", y[:nin], 16 + pf)
        write_mem(d / f"{case}_in_s16.mem", preemph(x16)[:nin], 16)
        write_mem(d / f"{case}_out_frames.mem", fr_all.ravel(), 16 + pf)
        write_mem(d / f"{case}_out_e.mem", hop_energy(preemph(x16))[:nfr], 48)
        # i2s_rx: kata 24 bit (sampel<<8 | byte rendah acak), harapan = 16 bit teratas
        d = out / "i2s_rx"
        rng = np.random.default_rng(sum(map(ord, case)))
        ns = 96
        words = (x16[:ns].astype(np.int64) << 8) | rng.integers(0, 256, ns)
        write_mem(d / f"{case}_in_words.mem", words, 24)
        write_mem(d / f"{case}_out.mem", x16[:ns], 16)
        # detektor pada aliran
        d = out / "cough_det"
        E = hop_energy(preemph(x16))        # detektor: pre-emphasis int16 (pembulatan)
        ev = cough_det(E)
        write_mem(d / f"{case}_in_E.mem", E, 48)
        write_mem(d / f"{case}_out_onsets.mem", [e_.onset for e_ in ev] or [-1], 16)
    for blk in ("preemph", "window", "fft512", "power", "mel", "log_q", "quant", "frontend", "cough_det", "frame_buf", "i2s_rx"):
        _meta(out / blk, "meta", block=blk, fft_w=W, cases=list(sig), calibration=params.CALIBRATION_SOURCE)
        made.append(blk)
    # CNN: model acak + kalibrasi requant pada fitur sintetis
    model = C.random_model(seed=123)
    feats = [F.to_cnn_input(F.features_q(x)) for x in sig.values()]
    C.calibrate_requant(model, feats)
    d = out / "cnn"
    for i, L_ in enumerate(model.layers):
        write_mem(d / f"{L_.name}_w.mem", L_.w.ravel(), 8)
        write_mem(d / f"{L_.name}_b.mem", L_.b, params.BIAS_BITS)
    write_mem(d / "requant.mem", [v for L_ in model.layers for v in (L_.m0, L_.shift)], 16)
    from golden.gen.gen_cnn_rom import write_roms
    write_roms(model, d)                                   # ROM dikemas untuk P lane (cnn_top)
    for case, x in zip(sig, feats):
        write_mem(d / f"{case}_in.mem", x.ravel(), 8)
        write_mem(d / f"{case}_out_logit.mem", C.cnn_forward(x, model), params.LOGIT_BITS)
    _meta(d, "meta", block="cnn", note="bobot ACAK (bukan terlatih)", layers=[L_.name for L_ in model.layers],
          requant=[(L_.m0, L_.shift) for L_ in model.layers], cases=list(sig))
    made.append("cnn")
    return made


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    print(main(ap.parse_args().out))
