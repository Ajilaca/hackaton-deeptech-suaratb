"""Model alamat/urutan modul DSP Fase 3 (Python, bukan simulasi RTL): window (alamat ROM simetris), mel (jalan ROM sparse
sampai eob), feat_buf (alamat cincin dan urutan baca CNN). Mencerminkan persamaan yang sama dengan RTL."""
import numpy as np
from golden import params
from golden.blocks.window import apply_window, hamming_q15
from golden.blocks.mel import mel_energy
from golden.blocks.frontend import segment, to_cnn_input
from golden.gen.gen_tables import mel_rom_words


def test_window_rom_address_symmetry():
    half = hamming_q15()[: params.FRAME_LEN // 2]
    rng = np.random.default_rng(1)
    x = rng.integers(-(1 << 23), 1 << 23, size=params.FRAME_LEN)
    y = np.zeros(params.NFFT, dtype=np.int64)
    for n in range(params.FRAME_LEN):
        a = n if n < 200 else (params.FRAME_LEN - 1) - n
        y[n] = (x[n] * int(half[a]) + (1 << 14)) >> 15          # G = 0, SH = 15
    y = np.clip(y, -(1 << 23), (1 << 23) - 1)
    assert np.array_equal(y, apply_window(x, W=24, in_frac=8))


def test_mel_rom_walk_matches_golden():
    words = mel_rom_words()
    rng = np.random.default_rng(2)
    p = rng.integers(0, 1 << 46, size=params.N_BINS)
    acc = [0] * params.N_MEL
    ptr = 0
    for k in range(params.N_BINS):
        while True:
            w = int(words[ptr]); ptr += 1
            band, wt, eob = (w >> 13) & 0x3F, (w >> 1) & 0xFFF, w & 1
            acc[band] += int(p[k]) * wt
            if eob:
                break
    assert ptr == len(words)
    assert np.array_equal(np.minimum(np.array(acc), (1 << params.LOG_IN_BITS) - 1), mel_energy(p))


def test_feat_buf_ring_addressing():
    nm, nf, ring = params.N_MEL, params.N_FRAMES, 64
    rng = np.random.default_rng(4)
    arr = rng.integers(-128, 128, size=(100, nm)).astype(np.int8)
    for n_written, starts in ((60, (0, 7, 10)), (100, (36, 50))):    # segmen harus masih di dalam cincin
        mem = np.zeros(ring * nm, dtype=np.int8)
        for f in range(n_written):
            for m in range(nm):
                mem[(f % ring) * nm + m] = arr[f, m]
        for start in starts:
            assert start + nf <= n_written and n_written - start <= ring
            out = [mem[((start + f) % ring) * nm + m] for m in range(nm) for f in range(nf)]
            assert out == to_cnn_input(segment(arr[:n_written], start)).ravel().tolist()
