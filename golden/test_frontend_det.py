import numpy as np
from golden import params, synth
from golden.blocks import frontend as F
from golden.blocks.preemph import preemph
from golden.blocks.cough_det import hop_energy, cough_det
from golden.blocks.fft512 import fft512
from golden.blocks.window import apply_window


def _snr_fft(x, W):
    fr = F.frame_signal(preemph(x, frac=W - 16))
    win = apply_window(fr, W, in_frac=W - 16)
    re, im, ns = fft512(win, W=W)
    ref = np.fft.fft(win.astype(float), axis=-1) / 512
    return 10 * np.log10(np.sum(np.abs(ref) ** 2) / np.sum((re - ref.real) ** 2 + (im - ref.imag) ** 2)), ns


def test_fft_snr_by_width_quiet_room():
    """Derau 3 LSB (ruang tenang): 18 bit tak memadai, 24 bit memadai."""
    x = F.pad_clip(synth.clip(11, 3.0, 15.0, "cough"))
    s18, _ = _snr_fft(x, 18)
    s24, n24 = _snr_fft(x, 24)
    assert n24 == 0
    assert s18 < 15.0 < 30.0 < s24


def test_features_close_to_float_reference():
    for sg, tol in ((3.0, 0.08), (10.0, 0.04), (30.0, 0.02)):
        x = F.pad_clip(synth.clip(5, sg, 20.0, "cough"))
        L = F.log_mel_q(x)
        ref = F.log_mel_ref(x, x_pe=preemph(x, frac=8) / 256.0)
        m = ref >= F.input_floor_L()[None, :] + 850.0       # >= lantai derau kuantisasi + 10 dB
        err = np.sqrt(np.mean(((L - ref)[m] / 256.0) ** 2))
        assert err < tol, (sg, err)


def test_end_to_end_vs_float_reasonable():
    """Termasuk semua pembulatan (pre-emphasis, window, FFT, mel, log) tanpa referensi yang diisolasi."""
    for sg, tol in ((3.0, 0.08), (30.0, 0.03)):
        x = F.pad_clip(synth.clip(6, sg, 20.0, "cough"))
        L = F.log_mel_q(x)
        ref = F.log_mel_ref(x)
        m = ref >= F.input_floor_L()[None, :] + 850.0
        assert m.mean() > 0.3
        assert np.sqrt(np.mean(((L - ref)[m] / 256.0) ** 2)) < tol, sg


def test_feature_range_not_saturated_much():
    fr = []
    for sg in (3.0, 10.0, 30.0):
        for kind in ("cough", "noise", "speech"):
            fr.append(F.features_q(F.pad_clip(synth.clip(2, sg, 15.0, kind)), offset=6690))   # offset kalibrasi sintetis
    q = np.concatenate([f.ravel() for f in fr]).astype(int)
    assert (np.mean(q == 127) + np.mean(q == -128)) < 0.05


def test_clip_pad_and_frame_count():
    x = F.pad_clip(np.ones(8000))
    assert len(x) == params.CLIP_SAMPLES_PADDED
    assert F.log_mel_q(x).shape == (params.N_FRAMES, params.N_MEL)


def test_silence_gives_zero_log():
    L = F.log_mel_q(np.zeros(params.CLIP_SAMPLES_PADDED, dtype=np.int64))
    assert np.all(L == 0)


def test_segment_pad():
    a = np.arange(60 * 2).reshape(60, 2)
    s = F.segment(a, -3)
    assert s.shape == (50, 2) and np.all(s[:3] == 0) and np.array_equal(s[3], a[0])
    assert F.to_cnn_input(np.zeros((50, 40), dtype=np.int8)).shape == (1, 40, 50)


# ---- detektor ---------------------------------------------------------------
def test_detector_quiet_noise_no_events():
    x, _ = synth.stream(1, 30.0, 10.0, 0, 0.0)
    assert cough_det(hop_energy(preemph(x))) == []


def test_detector_finds_synthetic_coughs():
    x, pos = synth.stream(2, 20.0, 10.0, 6, 20.0)
    ev = cough_det(hop_energy(preemph(x)))
    hits = [any(p // 160 - 3 <= e.onset <= p // 160 + 14 for e in ev) for p in pos]
    assert all(hits)
    for e in ev:
        assert e.end - e.onset + 1 == params.POST_ONSET
        assert e.onset - e.start <= params.PRE_ROLL


def test_detector_single_hop_spike_ignored_and_two_hops_trigger():
    E = [1000] * 100
    E[50] = 10 ** 9
    assert cough_det(E) == []
    E[51] = 10 ** 9
    ev = cough_det(E)
    assert len(ev) == 1 and ev[0].onset == 50 and ev[0].start == 45


def test_detector_refractory_and_noise_floor():
    E = [1000] * 400
    for f in (50, 51, 52, 150, 151, 152):
        E[f] = 10 ** 8
    ev = cough_det(E)
    assert [e.onset for e in ev] == [50, 150]            # kedua letupan terpisah >= hold
    E2 = [1000] * 400
    for f in (50, 51, 52, 53, 54, 70, 71, 72):
        E2[f] = 10 ** 8
    assert len(cough_det(E2)) == 1                       # letupan kedua jatuh di dalam segmen


def test_detector_floor_blocks_digital_silence_trigger():
    E = [0] * 50 + [params.DET_N_FLOOR * 5] * 5 + [0] * 50
    assert cough_det(E) == []                            # 5x lantai < K=8 x lantai


def test_hop_energy_alignment():
    x = np.zeros(1000, dtype=np.int64)
    x[240:400] = 10
    E = hop_energy(x)
    assert E[0] == 160 * 100 and E[1] == 0
