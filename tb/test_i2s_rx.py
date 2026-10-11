"""cocotb: i2s_rx dengan model mikrofon I2S (slave) di slot kiri. Dijalankan HANYA di Fase T."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import cocotb
from cocotb.triggers import FallingEdge, ClockCycles, RisingEdge

from tb.common import vec
from tb.common.cocotb_utils import start_clock, reset, StreamMonitor


async def i2s_mic(dut, words):
    """Model INMP441: setelah WS berubah, MSB keluar pada tepi turun BCLK berikutnya; slot kanan = 0."""
    prev_ws = None
    bits = []
    idx = 0
    dut.sd.value = 0
    while True:
        await FallingEdge(dut.bclk)
        ws = int(dut.ws.value)
        if prev_ws is None or ws != prev_ws:           # tepi turun pertama = awal slot kiri bingkai pertama
            if ws == 0 and idx < len(words):          # awal slot kiri
                w = words[idx] & 0xFFFFFF
                bits = [(w >> (23 - i)) & 1 for i in range(24)]
                idx += 1
            else:
                bits = []
        prev_ws = ws
        dut.sd.value = bits.pop(0) if bits else 0


@cocotb.test()
async def test_samples(dut):
    await start_clock(dut)
    dut.en.value = 0
    dut.sd.value = 0
    await reset(dut)
    for case in vec.cases("i2s_rx"):
        words = vec.load("i2s_rx", case, "in_words", 24, signed=False)
        exp = vec.load("i2s_rx", case, "out", 16)
        await reset(dut)
        mon = StreamMonitor(dut, "sample_valid", "sample")
        task = cocotb.start_soon(i2s_mic(dut, words))
        dut.en.value = 1
        # 64 BCLK/bingkai x 96 clk = 6144 clk per sampel; +3 bingkai margin
        await ClockCycles(dut.clk, 6144 * (len(words) + 3))
        task.cancel()
        mon.stop()
        dut.en.value = 0
        got = mon.column(0)
        vec.assert_equal(f"{case}.sample", got[: len(exp)], exp)
        assert len(got) >= len(exp), f"{case}: hanya {len(got)} sampel"


@cocotb.test()
async def test_rates(dut):
    """BCLK = clk/96 dan WS = BCLK/64 (periode 6144 clk)."""
    await start_clock(dut)
    dut.en.value = 1
    dut.sd.value = 0
    await reset(dut)
    await RisingEdge(dut.bclk)
    t0 = cocotb.utils.get_sim_time("ns")
    await RisingEdge(dut.bclk)
    t1 = cocotb.utils.get_sim_time("ns")
    assert abs((t1 - t0) - 96 * 10.0) < 1.0, f"periode BCLK {t1 - t0} ns"


def test_runner():
    from tb.common.runner import run
    run("i2s_rx", "test_i2s_rx")


if __name__ == "__main__":
    test_runner()
