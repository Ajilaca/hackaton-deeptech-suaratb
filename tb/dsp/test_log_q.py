"""cocotb: log_q vs golden (40 energi -> Q6.8). Dijalankan HANYA di Fase T."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import cocotb
from cocotb.triggers import ClockCycles

from golden import params
from tb.common import vec
from tb.common.cocotb_utils import start_clock, reset, StreamMonitor
from tb.common.drive import send

OUT_W = 14


@cocotb.test()
async def test_vectors(dut):
    await start_clock(dut)
    dut.in_valid.value = 0
    dut.in_last.value = 0
    dut.in_idx.value = 0
    dut.in_e.value = 0
    await reset(dut)
    for case in vec.cases("log_q"):
        e = vec.load("log_q", case, "in", params.LOG_IN_BITS, signed=False)
        exp = vec.load("log_q", case, "out", OUT_W, signed=False)
        mon = StreamMonitor(dut, "out_valid", "out_l", "out_idx", "out_last", signed=(False, False, False))
        await send(dut, e, "in_e", params.LOG_IN_BITS, last="in_last", idx="in_idx")
        await ClockCycles(dut.clk, 8)
        mon.stop()
        vec.assert_equal(f"{case}.l", mon.column(0), exp)
        vec.assert_equal(f"{case}.idx", mon.column(1), list(range(len(exp))))
        assert mon.column(2)[-1] == 1 and sum(mon.column(2)) == 1, f"{case}: out_last salah"


@cocotb.test()
async def test_edges(dut):
    """Nilai tepi: 0, 1, 2^k, 2^k - 1 untuk semua k, dan 2^58 - 1."""
    from golden.blocks.log_q import log2_q
    import numpy as np
    await start_clock(dut)
    dut.in_valid.value = 0
    await reset(dut)
    vals = [0, 1, 2, 3, 255, 256, 257, (1 << 58) - 1]
    for k in range(1, 58):
        vals += [1 << k, (1 << k) - 1, (1 << k) + 1]
    vals = [v for v in vals if v < (1 << 58)]
    exp = log2_q(np.array(vals, dtype=np.int64)).tolist()
    mon = StreamMonitor(dut, "out_valid", "out_l", signed=(False,))
    await send(dut, vals, "in_e", params.LOG_IN_BITS)
    await ClockCycles(dut.clk, 8)
    mon.stop()
    vec.assert_equal("edges.l", mon.column(0), exp)


def test_runner():
    from tb.common.runner import run, mem_param
    run("log_q", "test_log_q", parameters={"INIT_FILE": mem_param("log_lut.mem")})


if __name__ == "__main__":
    test_runner()
