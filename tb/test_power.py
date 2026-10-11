"""cocotb: power vs golden (257 bin). Dijalankan HANYA di Fase T."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import cocotb
from cocotb.triggers import RisingEdge, ClockCycles

from golden import params
from tb.common import vec
from tb.common.cocotb_utils import start_clock, reset, StreamMonitor

W = params.FFT_W


@cocotb.test()
async def test_vectors(dut):
    await start_clock(dut)
    dut.in_valid.value = 0
    dut.in_last.value = 0
    await reset(dut)
    for case in vec.cases("power"):
        re = vec.load("power", case, "in_re", W)
        im = vec.load("power", case, "in_im", W)
        exp = vec.load("power", case, "out", 2 * W, signed=False)
        assert len(re) == len(im) == len(exp) == params.N_BINS
        mon = StreamMonitor(dut, "out_valid", "out_p", "out_idx", "out_last", signed=(False, False, False))
        for k in range(params.N_BINS):
            dut.in_re.value = re[k] & ((1 << W) - 1)
            dut.in_im.value = im[k] & ((1 << W) - 1)
            dut.in_idx.value = k
            dut.in_last.value = int(k == params.N_BINS - 1)
            dut.in_valid.value = 1
            await RisingEdge(dut.clk)
        dut.in_valid.value = 0
        dut.in_last.value = 0
        await ClockCycles(dut.clk, 6)
        mon.stop()
        vec.assert_equal(f"{case}.p", mon.column(0), exp)
        vec.assert_equal(f"{case}.idx", mon.column(1), list(range(params.N_BINS)))
        assert mon.column(2)[-1] == 1 and sum(mon.column(2)) == 1, f"{case}: out_last salah"


def test_runner():
    from tb.common.runner import run
    run("power", "test_power")


if __name__ == "__main__":
    test_runner()
