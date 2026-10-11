"""cocotb: requant (P lane berurutan) vs golden requant."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import cocotb
import numpy as np
from cocotb.triggers import RisingEdge, ClockCycles

from golden import params
from golden.blocks.cnn import requant as g_requant
from tb.common import vec
from tb.common.cocotb_utils import start_clock, reset, StreamMonitor
from tb.common.drive import wait_for

P = params.P
ACC = params.ACC_BITS


@cocotb.test()
async def test_random(dut):
    await start_clock(dut)
    dut.start.value = 0
    await reset(dut)
    rng = np.random.default_rng(11)
    for it in range(40):
        final = it % 4 == 3
        lanes = int(rng.integers(1, P + 1))
        m0 = int(rng.integers(1, 1 << 15))
        shift = int(rng.integers(8, 28))
        scale = 1 << int(rng.integers(8, ACC - 1))
        acc = rng.integers(-scale, scale, size=P)
        acc[0] = (1 << (ACC - 1)) - 1 if it % 5 == 0 else acc[0]
        base, stride = int(rng.integers(0, 4000)), int(rng.integers(1, 500))
        exp = g_requant(acc[:lanes], m0, shift, relu=not final, out_bits=16 if final else 8).tolist()
        word = 0
        for i, a in enumerate(acc):
            word |= (int(a) & ((1 << ACC) - 1)) << (ACC * i)
        dut.res.value = word
        dut.lanes.value = lanes
        dut.base.value = base
        dut.stride.value = stride
        dut.m0.value = m0
        dut.shift.value = shift
        dut.final_l.value = int(final)
        mon = StreamMonitor(dut, "we", "wdata", "waddr", signed=(True, False))
        dut.start.value = 1
        await RisingEdge(dut.clk)
        dut.start.value = 0
        await wait_for(dut, lambda: int(dut.done.value) == 1, 200, f"it{it}: done")
        await ClockCycles(dut.clk, 2)
        mon.stop()
        vec.assert_equal(f"it{it}.wdata", mon.column(0), exp)
        vec.assert_equal(f"it{it}.waddr", mon.column(1), [base + i * stride for i in range(lanes)])


def test_runner():
    from tb.common.runner import run
    run("requant", "test_requant")


if __name__ == "__main__":
    test_runner()
