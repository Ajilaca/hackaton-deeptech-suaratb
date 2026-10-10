import numpy as np
import pytest
from golden import params
from golden.blocks import cnn as C

RNG = np.random.default_rng(7)


def _naive_conv(x, w, b):
    cin, h, wd = x.shape
    cout = w.shape[0]
    out = np.zeros((cout, h, wd), dtype=np.int64)
    xp = np.pad(x, ((0, 0), (1, 1), (1, 1)))
    for o in range(cout):
        for i in range(h):
            for j in range(wd):
                out[o, i, j] = int(np.sum(xp[:, i:i + 3, j:j + 3] * w[o])) + int(b[o])
    return out


def test_conv_matches_naive():
    x = RNG.integers(-128, 128, (3, 6, 7))
    w = RNG.integers(-128, 128, (4, 3, 3, 3))
    b = RNG.integers(-1000, 1000, 4)
    assert np.array_equal(C.conv3x3(x, w, b), _naive_conv(x, w, b))


def test_requant_exact_python_ints():
    acc = np.array([-100000, -1, 0, 1, 12345, 8_000_000])
    m0, s = 20000, 22
    y = C.requant(acc, m0, s, relu=False)
    for a, v in zip(acc, y):
        r = (int(a) * m0 + (1 << (s - 1))) >> s
        assert v == max(-128, min(127, r))
    assert C.requant(np.array([-5000]), m0, s, relu=True)[0] == 0
    assert C.requant(np.array([10 ** 8]), m0, s, relu=True)[0] == 127
    assert C.requant(np.array([10 ** 9]), m0, s, relu=False, out_bits=16)[0] == 32767


def test_maxpool_and_gap():
    x = np.arange(2 * 5 * 6).reshape(2, 5, 6)
    p = C.maxpool2(x)
    assert p.shape == (2, 2, 3) and p[0, 0, 0] == 7
    g = C.gap(np.full((4, 5, 6), 100))
    assert np.all(g == 100)
    g2 = C.gap(np.full((1, 5, 6), 127))
    assert abs(int(g2[0]) - 127) <= 1


def test_forward_shapes_and_determinism():
    m = C.random_model(1)
    x = RNG.integers(-128, 128, (1, 40, 50))
    C.calibrate_requant(m, [x])
    st = {}
    y = C.cnn_forward(x, m, st)
    assert y.shape == (1,) and abs(int(y[0])) <= 32767
    assert np.array_equal(y, C.cnn_forward(x, m))
    assert set(st) == {"conv1", "conv2", "conv3", "fc1", "fc2"}
    assert max(st.values()) < 1 << 23


def test_weight_count_of_model():
    m = C.random_model(2)
    assert sum(L.w.size for L in m.layers) == 14496
    assert sum(L.b.size for L in m.layers) == 97


def test_accumulator_worst_case_fits_24_bits():
    """Batas analitik: 288 * 127 * 128 + bias harus < 2^23."""
    fan_in = {l["name"]: l["cin"] * l["k"] ** 2 for l in params.layer_descs()}
    for nm, fi in fan_in.items():
        a_max = 128 if nm == "conv1" else 127          # masukan conv1 bertanda; selanjutnya pasca-ReLU
        assert C.worst_case_acc(fi, a_max) + (1 << 20) < 1 << 23, nm


def test_overflow_assertion_triggers():
    m = C.random_model(3)
    m.layers[0].b[:] = (1 << 23)                        # bias di luar rentang 24 bit
    with pytest.raises(C.AccOverflow):
        C.cnn_forward(np.zeros((1, 40, 50), dtype=np.int64), m)


def test_extreme_inputs_no_overflow():
    m = C.random_model(4)
    for L in m.layers:
        L.w[:] = -128
        L.b[:] = 0
    for v in (127, -128):
        C.cnn_forward(np.full((1, 40, 50), v, dtype=np.int64), m)


def test_parametric_channels():
    m = C.random_model(5, c1=8, c2=16, c3=16)
    y = C.cnn_forward(RNG.integers(-128, 128, (1, 40, 50)), m)
    assert y.shape == (1,)


def test_m0_shift_decomposition():
    for v in (0.001, 0.0173, 0.5, 3.2):
        m0, s = C.m0_shift_from_scale(v)
        assert (1 << 14) <= m0 < (1 << 15) and abs(m0 / (1 << s) - v) / v < 1e-3
