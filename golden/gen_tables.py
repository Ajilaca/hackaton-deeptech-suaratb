"""Menghasilkan tabel ROM (.mem) dan paket SV dari params.py. Tabel diturunkan dari parameter,
bukan dari data/bobot, sehingga boleh di-commit.

    python -m golden.gen_tables [--out golden/tables]

Berkas:
  twiddle_cos.mem / twiddle_sin.mem  256 x 16 bit (Q1.15, W^k = cos - j sin, k=0..255)
  window_hamming.mem                 200 x 16 bit (setengah simetris; indeks n dan 399-n sama)
  mel_rom.mem                        N x 19 bit {band[5:0], w[11:0], eob}; tiap bin >= 1 entri
  log_lut.mem                        256 x 8 bit
  suaratb_pkg.sv                     paket parameter
"""
from __future__ import annotations
import argparse
import pathlib
import numpy as np
from . import params
from .blocks.fft512 import twiddle_q15
from .blocks.window import hamming_q15
from .blocks.mel import mel_sparse
from .blocks.log_q import log_lut
from .io_mem import write_mem

DEFAULT_OUT = pathlib.Path(__file__).parent / "tables"


def mel_rom_words():
    """Satu kata per (bin, band) berbobot; bin tanpa band diberi kata nol. eob=1 pada kata terakhir tiap bin."""
    sp = mel_sparse()
    by_bin: dict[int, list] = {}
    for k, b, w in sp:
        by_bin.setdefault(k, []).append((b, w))
    words = []
    for k in range(params.N_BINS):
        ent = by_bin.get(k, [(0, 0)])
        for i, (b, w) in enumerate(ent):
            eob = 1 if i == len(ent) - 1 else 0
            words.append((b << 13) | (w << 1) | eob)
    return np.array(words, dtype=np.int64)


def main(out: pathlib.Path = DEFAULT_OUT) -> dict:
    out = pathlib.Path(out)
    c, s = twiddle_q15()
    write_mem(out / "twiddle_cos.mem", c, 16, "cos(2*pi*k/512) Q1.15, k=0..255")
    write_mem(out / "twiddle_sin.mem", s, 16, "sin(2*pi*k/512) Q1.15, k=0..255")
    w = hamming_q15()
    assert np.array_equal(w, w[::-1])
    write_mem(out / "window_hamming.mem", w[: params.FRAME_LEN // 2], 16, "Hamming Q1.15 n=0..199 (simetris)")
    mw = mel_rom_words()
    write_mem(out / "mel_rom.mem", mw, 19, "{band[5:0], w[11:0], eob}; urut bin 0..256")
    write_mem(out / "log_lut.mem", log_lut(), 8, "round(256*log2(1+(m+0.5)/256)), maks 255")
    (out / "suaratb_pkg.sv").write_text(params.sv_package())
    from .gen_cnn_rom import cfg_package
    (out / "cnn_cfg_pkg.sv").write_text(cfg_package())
    return dict(twiddle=len(c), window=len(w) // 2, mel_rom=len(mw), log_lut=256)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    print(main(ap.parse_args().out))
