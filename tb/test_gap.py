"""cocotb: gap vs golden gap (CH x HW nilai int8 >= 0)."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import cocotb
import numpy as np
from cocotb.triggers import RisingEdge, ClockCycles

from golden.blocks.cnn import gap as g_gap
from tb.common import vec
from tb.common.cocotb_utils import start_clock, reset, StreamMonitor
from tb.common.drive import wait_for

CH, HW = 32, 30


@cocotb.test()
async def test_random(dut):
    await start_clock(dut)
    dut.start.value = 0
    dut.rd_data.value = 0
    await reset(dut)
    rng = np.random.default_rng(5)
    for it in range(3):
        x = rng.integers(0, 128, size=(CH, HW)) if it < 2 else np.full((CH, HW), 127)
        flat = x.ravel().tolist()
        exp = g_gap(x.reshape(CH, 5, 6)).tolist()
        mon = StreamMonitor(dut, "we", "wdata", "waddr", signed=(True, False))
        dut.start.value = 1
        await RisingEdge(dut.clk)
        dut.start.value = 0

        async def feed():
            # rd_data valid 1 siklus setelah rd_en
            while True:
                await RisingEdge(dut.clk)
                if int(dut.rd_en.value):
                    a = int(dut.rd_addr.value)
                    await ClockCycles(dut.clk, 0)
                    dut.rd_data.value = flat[a] & 0xFF
        t = cocotb.start_soon(feed())
        await wait_for(dut, lambda: int(dut.done.value) == 1, 5000, f"it{it}: done")
        await ClockCycles(dut.clk, 2)
        t.cancel(); mon.stop()
        vec.assert_equal(f"it{it}.wdata", mon.column(0), exp)
        vec.assert_equal(f"it{it}.waddr", mon.column(1), list(range(CH)))


def test_runner():
    from tb.common.runner import run
    run("gap", "test_gap")


if __name__ == "__main__":
    test_runner()
