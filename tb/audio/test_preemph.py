"""cocotb: preemph vs golden (jalur lebar 16+FRAC dan jalur int16). Dijalankan HANYA di Fase T."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import cocotb
from cocotb.triggers import ClockCycles

from golden import params
from tb.common import vec
from tb.common.cocotb_utils import start_clock, reset, StreamDriver, StreamMonitor

WIDE = 16 + params.PREEMPH_FRAC


@cocotb.test()
async def test_vectors(dut):
    await start_clock(dut)
    dut.in_valid.value = 0
    dut.in_sample.value = 0
    await reset(dut)
    for case in vec.cases("preemph"):
        x = vec.load("preemph", case, "in", 16)
        exp_w = vec.load("preemph", case, "out", WIDE)
        exp_s = vec.load("preemph", case, "out_s16", 16)
        await reset(dut)                         # x[-1] = 0 untuk tiap kasus
        mon = StreamMonitor(dut, "out_valid", "out_wide", "out_s16")
        drv = StreamDriver(dut, "in_valid", "in_sample", gap=1)
        await drv.send(x, mask_bits=16)
        await ClockCycles(dut.clk, 6)
        mon.stop()
        vec.assert_equal(f"{case}.out_wide", mon.column(0), exp_w)
        vec.assert_equal(f"{case}.out_s16", mon.column(1), exp_s)


@cocotb.test()
async def test_saturation(dut):
    """Loncatan penuh-skala: -32768 lalu +32767 harus jenuh persis seperti golden."""
    from golden.blocks.preemph import preemph
    import numpy as np
    await start_clock(dut)
    dut.in_valid.value = 0
    await reset(dut)
    x = [-32768, 32767, -32768, 32767, 0, 0, 32767, 32767]
    mon = StreamMonitor(dut, "out_valid", "out_wide", "out_s16")
    drv = StreamDriver(dut, "in_valid", "in_sample", gap=1)
    await drv.send(x, mask_bits=16)
    await ClockCycles(dut.clk, 6)
    mon.stop()
    vec.assert_equal("sat.wide", mon.column(0), preemph(np.array(x), frac=params.PREEMPH_FRAC).tolist())
    vec.assert_equal("sat.s16", mon.column(1), preemph(np.array(x)).tolist())


def test_runner():
    from tb.common.runner import run
    run("preemph", "test_preemph")


if __name__ == "__main__":
    test_runner()
