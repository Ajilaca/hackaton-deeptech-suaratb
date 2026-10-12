import numpy as np
from golden import params
from golden.gen import gen_tables, gen_vectors
from golden.gen.io_mem import read_mem, write_mem
from golden.blocks.fft512 import twiddle_q15
from golden.blocks.mel import mel_weights_q
from golden.blocks.log_q import log_lut


def test_tables_roundtrip(tmp_path):
    info = gen_tables.main(tmp_path)
    assert info["mel_rom"] >= params.N_BINS
    c, s = twiddle_q15()
    assert np.array_equal(read_mem(tmp_path / "twiddle_cos.mem", 16), c)
    assert np.array_equal(read_mem(tmp_path / "twiddle_sin.mem", 16), s)
    assert np.array_equal(read_mem(tmp_path / "log_lut.mem", 8, signed=False), log_lut())
    # mel_rom -> rekonstruksi bobot
    words = read_mem(tmp_path / "mel_rom.mem", 19, signed=False)
    d = np.zeros((params.N_MEL, params.N_BINS), dtype=np.int64)
    k = 0
    for w in words:
        band, wt, eob = (int(w) >> 13) & 63, (int(w) >> 1) & 4095, int(w) & 1
        if wt:
            d[band, k] = wt
        if eob:
            k += 1
    assert k == params.N_BINS and np.array_equal(d, mel_weights_q())


def test_write_mem_range_check(tmp_path):
    import pytest
    with pytest.raises(AssertionError):
        write_mem(tmp_path / "x.mem", [300], 8)
    write_mem(tmp_path / "y.mem", [-1, 127], 8)
    assert read_mem(tmp_path / "y.mem", 8).tolist() == [-1, 127]


def test_vectors_deterministic(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    gen_vectors.main(a)
    gen_vectors.main(b)
    fa = sorted(p.relative_to(a) for p in a.rglob("*.mem"))
    assert fa and fa == sorted(p.relative_to(b) for p in b.rglob("*.mem"))
    for f in fa:
        assert (a / f).read_bytes() == (b / f).read_bytes()
