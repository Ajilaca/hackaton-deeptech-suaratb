import math
from golden import params
from golden.analysis import budget


def test_counts_match_spec():
    assert params.n_weights() == 14496
    assert params.n_biases() == 97
    assert params.n_macs() == 3_698_448


def test_frame_math():
    assert (params.CLIP_SAMPLES_PADDED - params.FRAME_LEN) // params.HOP + 1 == params.N_FRAMES
    assert params.PRE_ROLL + params.POST_ONSET == params.N_FRAMES
    assert params.PREEMPH_Q15 == round(0.97 * 32768)
    assert params.FFT_GUARD == params.FFT_W - 16
    assert len(params.FFT_STAGE_SHIFT) == 9


def test_clock_plan():
    # clk 98,304 MHz: BCLK = clk/96 = 1,024 MHz = 64 * 16 kHz (I2S 2 x 32 bit)
    assert 98_304_000 / 96 == 64 * params.FS


def test_log_range_covers_mel_bound():
    assert params.LOG_IN_BITS >= params.MEL_ACC_BITS - params.MEL_SHIFT


def test_layer_chain():
    L = params.layer_descs()
    assert [l["name"] for l in L] == ["conv1", "conv2", "conv3", "fc1", "fc2"]
    assert (L[1]["h"], L[1]["w"]) == (20, 25) and (L[2]["h"], L[2]["w"]) == (10, 12)
    assert L[3]["gap_hw"] == 5 * 6 and L[3]["cin"] == params.C3


def test_derived_channels_parametric():
    L = params.layer_descs(8, 16, 16)
    assert params.n_weights(c1=8, c2=16, c3=16) == sum(l["cin"] * l["cout"] * l["k"] ** 2 for l in L)


def test_sv_package():
    t = params.sv_package()
    assert "localparam int FFT_W = 24;" in t and t.strip().endswith("endpackage")


def test_budget_cycles_match_analytic():
    assert budget.cycles_cnn(16)["total"] == 231_168
    assert budget.cycles_cnn(8)["total"] == 462_320
    c32 = budget.cycles_cnn(32)       # conv1 hanya 16 kanal: separuh lajur PE menganggur
    assert c32["conv1"] == 18_000 and c32["conv2"] == 72_000 and c32["conv3"] == 34_560
    assert c32["total"] == 124_608


def test_budget_memory_in_expected_band():
    r = budget.report()
    assert 250_000 < r["mem_bits"] < 330_000
    assert r["devices"]["XC7A35T"]["mem_ok"]
    assert not r["devices"]["EP4CE6"]["mem_ok"]          # BASE tak muat di EP4CE6 (memori)
    assert r["devices_note"].startswith("dari ingatan")


def test_budget_scales_with_params():
    small = budget.report(P=8, c=(8, 16, 16))
    big = budget.report()
    assert small["mem_bits"] < big["mem_bits"] and small["macs"] < big["macs"]
    assert small["weights"] == 3800


def test_mult_tiles():
    assert budget.n_tiles(24, 25) == 1 and budget.n_tiles(26, 25) == 2
    assert budget.mult_cost(24, 16, True, True, 25, 18) == 1
    assert budget.mult_cost(24, 24, True, True, 25, 18) == 2
    assert budget.mult_cost(8, 8, True, True, 25, 18) == 1
