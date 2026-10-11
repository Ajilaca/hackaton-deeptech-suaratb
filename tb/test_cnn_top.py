"""cocotb: cnn_top vs golden cnn_forward (model int8 acak + kalibrasi requant sintetis, tb/vectors/cnn)."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import cocotb
from cocotb.triggers import RisingEdge, ClockCycles

from tb.common import vec
from tb.common.cocotb_utils import start_clock, reset, sval
from tb.common.drive import wait_for

CASES = None   # semua kasus; set lewat env CNN_CASES="noise3,cough" untuk mempercepat


@cocotb.test()
async def test_logits(dut):
    import os
    await start_clock(dut)
    dut.ld_valid.value = 0
    dut.start.value = 0
    dut.ld_addr.value = 0
    dut.ld_q.value = 0
    await reset(dut)
    cases = os.environ.get("CNN_CASES", "").split(",") if os.environ.get("CNN_CASES") else vec.cases("cnn")
    for case in cases:
        x = vec.load("cnn", case, "in", 8)
        exp = vec.load("cnn", case, "out_logit", 16)
        assert len(x) == 2000 and len(exp) == 1
        for a, v in enumerate(x):
            dut.ld_addr.value = a
            dut.ld_q.value = v & 0xFF
            dut.ld_valid.value = 1
            await RisingEdge(dut.clk)
        dut.ld_valid.value = 0
        dut.start.value = 1
        await RisingEdge(dut.clk)
        dut.start.value = 0
        await wait_for(dut, lambda: int(dut.done.value) == 1, 900000, f"{case}: done")
        await ClockCycles(dut.clk, 2)
        got = sval(dut.logit)
        assert got == exp[0], f"{case}: logit {got} != golden {exp[0]}"
        dut._log.info("%s: logit=%d OK", case, got)


def test_runner():
    from tb.common.runner import run, mem_param
    run("cnn_top", "test_cnn_top", parameters={"INIT_W": mem_param("../../tb/vectors/cnn/cnn_wrom.mem")})


if __name__ == "__main__":
    test_runner()
