"""cocotb: feat_buf (cincin fitur + pembacaan segmen urutan CNN). HANYA di Fase T."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import cocotb
import numpy as np
from cocotb.triggers import RisingEdge, ClockCycles

from golden import params
from golden.blocks import frontend as F
from tb.common.cocotb_utils import start_clock, reset, StreamMonitor
from tb.common.drive import wait_for

NF, NM = params.N_FRAMES, params.N_MEL


async def write_frames(dut, arr):
    for row in arr:
        for m, v in enumerate(row):
            dut.in_q.value = int(v) & 0xFF
            dut.in_idx.value = m
            dut.in_last.value = int(m == NM - 1)
            dut.in_valid.value = 1
            await RisingEdge(dut.clk)
        dut.in_valid.value = 0
        dut.in_last.value = 0
        await ClockCycles(dut.clk, 2)


async def read_segment(dut, start):
    mon = StreamMonitor(dut, "out_valid", "out_q", "out_addr", "out_last", signed=(True, False, False))
    dut.seg_start.value = start
    dut.seg_go.value = 1
    await RisingEdge(dut.clk)
    dut.seg_go.value = 0
    await wait_for(dut, lambda: len(mon.items) >= NF * NM, NF * NM + 50, "segmen")
    await ClockCycles(dut.clk, 3)
    mon.stop()
    return mon


@cocotb.test()
async def test_segments(dut):
    await start_clock(dut)
    dut.in_valid.value = 0
    dut.in_last.value = 0
    dut.seg_go.value = 0
    dut.seg_start.value = 0
    await reset(dut)
    rng = np.random.default_rng(7)
    arr = rng.integers(-128, 128, size=(100, NM)).astype(np.int8)       # 100 frame: cincin 64 membungkus
    await write_frames(dut, arr[:60])
    for start in (0, 7, 10):
        mon = await read_segment(dut, start)
        exp = F.to_cnn_input(F.segment(arr[:60], start)).ravel().tolist()
        assert mon.column(0) == exp, f"start={start}: nilai tidak sama"
        assert mon.column(1) == list(range(NF * NM)), f"start={start}: out_addr salah"
        assert mon.column(2)[-1] == 1 and sum(mon.column(2)) == 1, f"start={start}: out_last salah"
    await write_frames(dut, arr[60:])                                   # frame_count = 100
    mon = await read_segment(dut, 50)                                   # frame 50..99, melewati batas cincin
    exp = F.to_cnn_input(F.segment(arr, 50)).ravel().tolist()
    assert mon.column(0) == exp, "start=50 (cincin membungkus): nilai tidak sama"
    assert int(dut.frame_count.value) == 100


def test_runner():
    from tb.common.runner import run
    run("feat_buf", "test_feat_buf")


if __name__ == "__main__":
    test_runner()
