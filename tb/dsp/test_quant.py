"""cocotb: quant vs golden (L Q6.8 -> int8). Dijalankan HANYA di Fase T."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import cocotb
from cocotb.triggers import ClockCycles

from golden import params
from tb.common import vec
from tb.common.cocotb_utils import start_clock, reset, StreamMonitor
from tb.common.drive import send

IN_W = 14


@cocotb.test()
async def test_vectors(dut):
    await start_clock(dut)
    dut.in_valid.value = 0
    dut.in_last.value = 0
    dut.in_idx.value = 0
    dut.in_l.value = 0
    await reset(dut)
    for case in vec.cases("quant"):
        L = vec.load("quant", case, "in", IN_W, signed=False)
        exp = vec.load("quant", case, "out", 8)
        mon = StreamMonitor(dut, "out_valid", "out_q", "out_idx", "out_last", signed=(True, False, False))
        await send(dut, L, "in_l", IN_W, last="in_last", idx="in_idx")
        await ClockCycles(dut.clk, 4)
        mon.stop()
        vec.assert_equal(f"{case}.q", mon.column(0), exp)
        assert mon.column(2)[-1] == 1 and sum(mon.column(2)) == 1, f"{case}: out_last salah"


@cocotb.test()
async def test_sweep(dut):
    """Sapuan seluruh rentang L (0..16383) terhadap quant_feat golden: tepi saturasi dan pembulatan."""
    from golden.blocks.quant import quant_feat
    import numpy as np
    await start_clock(dut)
    dut.in_valid.value = 0
    await reset(dut)
    L = np.arange(0, 1 << IN_W, 7)
    exp = quant_feat(L).tolist()
    mon = StreamMonitor(dut, "out_valid", "out_q", signed=(True,))
    await send(dut, L.tolist(), "in_l", IN_W)
    await ClockCycles(dut.clk, 4)
    mon.stop()
    vec.assert_equal("sweep.q", mon.column(0), exp)


def test_runner():
    from tb.common.runner import run
    run("quant", "test_quant")


if __name__ == "__main__":
    test_runner()
