import numpy as np
import pytest
from golden import params
from golden.gen import synth
from golden.blocks.fixed import rshift_round, sat, bit_length
from golden.blocks.preemph import preemph
from golden.blocks.window import hamming_q15, apply_window
from golden.blocks.fft512 import fft512, twiddle_q15
from golden.blocks.power import power
from golden.blocks.mel import mel_filterbank_float, mel_weights_q, mel_sparse, mel_energy
from golden.blocks.log_q import log2_q, log_lut
from golden.blocks.quant import quant_feat

RNG = np.random.default_rng(42)


# ---- fixed ---------------------------------------------------------------
def test_rshift_round_half_up():
    assert rshift_round(3, 1) == 2 and rshift_round(-3, 1) == -1 and rshift_round(5, 2) == 1
    assert rshift_round(-5, 2) == -1 and rshift_round(7, 0) == 7


def test_bit_length_exact_large():
    v = np.array([0, 1, 2, 3, 2**52 - 1, 2**53 - 1, 2**53, 2**57, 2**57 - 1, 2**62 - 1], dtype=np.int64)
    assert list(bit_length(v)) == [0, 1, 2, 2, 52, 53, 54, 58, 57, 62]


# ---- preemph --------------------------------------------------------------
def test_preemph_matches_float_within_1lsb():
    x = RNG.integers(-30000, 30000, 5000)
    y = preemph(x)
    ref = x.astype(float).copy()
    ref[1:] -= 0.97 * x[:-1]
    ref = np.clip(ref, -32768, 32767)
    assert np.max(np.abs(y - ref)) <= 1.0


def test_preemph_streaming_equals_block():
    x = RNG.integers(-32768, 32768, 1000)
    whole = preemph(x)
    parts = np.concatenate([preemph(x[:300]), preemph(x[300:], prev=int(x[299]))])
    assert np.array_equal(whole, parts)


def test_preemph_fractional_bits_reduce_error():
    x = RNG.integers(-30, 30, 4000)
    ref = x.astype(float).copy(); ref[1:] -= 0.97 * x[:-1]
    e0 = np.sqrt(np.mean((preemph(x) - ref) ** 2))
    e8 = np.sqrt(np.mean((preemph(x, frac=8) / 256.0 - ref) ** 2))
    assert e8 < e0 / 20 and e8 < 0.01
    assert np.array_equal(preemph(x, frac=8) >> 8, preemph(x, frac=8) // 256)


def test_preemph_saturates():
    y = preemph(np.array([-32768, 32767, -32768, 32767]))
    assert y.max() <= 32767 and y.min() >= -32768 and y[1] == 32767


# ---- window ---------------------------------------------------------------
def test_hamming_symmetric_and_close():
    w = hamming_q15()
    assert len(w) == 400 and np.array_equal(w, w[::-1])
    f = 0.54 - 0.46 * np.cos(2 * np.pi * np.arange(400) / 399)
    assert np.max(np.abs(w / 32768 - f)) <= 1 / 32768
    assert w.min() >= 0 and w.max() <= 32767


def test_window_output_range_and_pad():
    fr = np.full((3, 400), -32768, dtype=np.int64)
    out = apply_window(fr, W=24)
    assert out.shape == (3, 512) and np.all(out[:, 400:] == 0)
    assert out.min() >= -(1 << 23) and out.max() < (1 << 23)


# ---- fft --------------------------------------------------------------------
def _ref_fft(x, W, ss=None):
    ss = params.FFT_STAGE_SHIFT if ss is None else ss
    return np.fft.fft(x.astype(float), axis=-1) / (1 << sum(ss))


def test_twiddles():
    c, s = twiddle_q15()
    assert len(c) == 256 and c[0] == 32767 and s[0] == 0 and abs(int(s[128]) - 32767) <= 0
    assert np.max(np.abs(c / 32768 - np.cos(2 * np.pi * np.arange(256) / 512))) <= 1.5 / 32768


@pytest.mark.parametrize("W", [18, 24])
def test_fft_matches_float_and_no_overflow(W):
    g = W - 16
    x = RNG.integers(-32768, 32768, (4, 512)) << g
    re, im, ns = fft512(x, W=W)
    assert ns == 0
    ref = _ref_fft(x, W)
    snr = 10 * np.log10(np.sum(np.abs(ref) ** 2) / np.sum(np.abs((re + 1j * im) - ref) ** 2))
    # sinyal skala penuh: dibatasi presisi twiddle Q1.15 (~85 dB), bukan lebar data
    assert snr > (55 if W == 18 else 80), snr


def test_fft_worst_case_inputs_do_not_saturate():
    for W in (18, 24):
        g = W - 16
        lim = (1 << (W - 1)) - 1
        cases = [np.full(512, lim), np.where(np.arange(512) % 2 == 0, lim, -lim - 1),
                 np.where(np.arange(512) % 4 < 2, lim, -lim - 1), np.full(512, -lim - 1)]
        for c in cases:
            re, im, ns = fft512(c.astype(np.int64), W=W)
            assert ns == 0


def test_fft_impulse_and_sine():
    x = np.zeros(512, dtype=np.int64)
    x[0] = 1 << 22
    re, im, _ = fft512(x, W=24)
    assert np.all(np.abs(re - (1 << 22) / 512) <= 2) and np.all(np.abs(im) <= 2)
    n = np.arange(512)
    s = np.rint(20000 * (1 << 8) * np.sin(2 * np.pi * 64 * n / 512)).astype(np.int64)
    re, im, _ = fft512(s, W=24)
    mag = np.hypot(re, im)
    assert np.argmax(mag[:257]) == 64
    assert abs(mag[64] - 20000 * 256 / 2) / (20000 * 256 / 2) < 1e-3     # A*2^G*(N/2)/512


def test_fft_snr_improves_with_width():
    x = np.rint(RNG.normal(0, 3, (8, 512))).astype(np.int64)
    snrs = []
    for W in (18, 20, 24):
        xi = x << (W - 16)
        re, im, _ = fft512(xi, W=W)
        ref = _ref_fft(xi, W)
        e = (re - ref.real) ** 2 + (im - ref.imag) ** 2
        snrs.append(10 * np.log10(np.sum(np.abs(ref) ** 2) / np.sum(e)))
    assert snrs[0] < snrs[1] < snrs[2]
    assert snrs[2] > 30.0            # FFT 24 bit: > 30 dB pada derau 3 LSB


def test_fft_linear_in_two_runs_determinism():
    x = RNG.integers(-1000, 1000, (2, 512)) << 8
    a = fft512(x, W=24)
    b = fft512(x, W=24)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])


def test_fft_real_input_conjugate_symmetry_approx():
    x = RNG.integers(-5000, 5000, 512) << 8
    re, im, _ = fft512(x, W=24)
    k = np.arange(1, 256)
    assert np.max(np.abs(re[k] - re[512 - k])) <= 4 and np.max(np.abs(im[k] + im[512 - k])) <= 4


# ---- power -----------------------------------------------------------------
def test_power_range():
    lim = 1 << (params.FFT_W - 1)
    re = np.array([-lim, lim - 1]); im = np.array([-lim, lim - 1])
    p = power(np.pad(re, (0, 300)), np.pad(im, (0, 300)))
    assert p.max() < (1 << (params.POWER_BITS + 1))   # re^2+im^2 <= 2*lim^2 = 2^(2W-1)
    assert p.shape[-1] == params.N_BINS


# ---- mel --------------------------------------------------------------------
def test_mel_filterbank_properties():
    wq = mel_weights_q()
    assert wq.shape == (params.N_MEL, params.N_BINS)
    assert np.all(wq.sum(axis=1) > 0), "ada band kosong"
    assert wq.max() <= (1 << params.MEL_W_FRAC) - 1
    assert np.all((wq > 0).sum(axis=0) <= 2), "satu bin masuk > 2 band"
    fb = mel_filterbank_float()
    assert np.max(np.abs(wq / (1 << params.MEL_W_FRAC) - fb)) <= 1.0 / 4096 + 1e-9


def test_mel_sparse_equals_dense():
    wq = mel_weights_q()
    d = np.zeros_like(wq)
    for k, b, w in mel_sparse():
        d[b, k] = w
    assert np.array_equal(d, wq)


def test_mel_energy_matches_dense_float():
    p = RNG.integers(0, 1 << 40, (3, params.N_BINS))
    e = mel_energy(p, shift=0, in_bits=60)
    ref = p.astype(float) @ (mel_weights_q().T).astype(float)
    assert np.max(np.abs(e - ref) / np.maximum(ref, 1)) < 1e-12


# ---- log --------------------------------------------------------------------
def test_log_lut_monotone_and_bounds():
    l = log_lut()
    assert len(l) == 256 and np.all(np.diff(l) >= 0) and l[0] == 1 and l[-1] == 255


def test_log2_zero_and_powers():
    assert log2_q(0) == 0
    for p in (0, 1, 7, 8, 9, 31, 40, 57):
        assert log2_q(1 << p) == (p << 8) + 1


def test_log2_accuracy():
    e = np.unique(np.concatenate([RNG.integers(1, 1 << 50, 5000), np.arange(1, 3000)]))
    L = log2_q(e)
    ref = 256 * np.log2(e.astype(float))
    assert np.max(np.abs(L - ref)) <= 2.0          # <= 2 LSB Q.8 (0,008 oktaf)
    assert np.all(np.diff(log2_q(np.sort(e))) >= 0)


# ---- quant ------------------------------------------------------------------
def test_quant_saturation_and_monotone():
    L = np.arange(0, 20000)
    q = quant_feat(L, offset=6668, shift=5)
    assert q.min() == -128 and q.max() == 127 and np.all(np.diff(q) >= 0)
    assert quant_feat(6668, 6668, 5) == 0
