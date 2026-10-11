"""cocotb: frame_buf vs golden (frame 400 sampel tiap hop 160 dan energi hop). Dijalankan HANYA di Fase T."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import cocotb
from cocotb.triggers import ClockCycles

from golden import params
from tb.common import vec
from tb.common.cocotb_utils import start_clock, reset, StreamDriver, StreamMonitor

W = 16 + params.PREEMPH_FRAC
GAP = params.FRAME_LEN + 8          # jarak antar sampel >= FRAME_LEN + 4 (kontrak frame_buf)


@cocotb.test()
async def test_frames_and_energy(dut):
    await start_clock(dut)
    dut.in_valid.value = 0
    dut.in_wide.value = 0
    dut.in_s16.value = 0
    for case in ("cough", "noise3"):
        wide = vec.load("frame_buf", case, "in_wide", W)
        s16 = vec.load("frame_buf", case, "in_s16", 16)
        exp_fr = vec.load("frame_buf", case, "out_frames", W)
        exp_e = vec.load("frame_buf", case, "out_e", 48, signed=False)
        await reset(dut)
        fmon = StreamMonitor(dut, "frame_valid", "frame_data", "frame_first", "frame_last", "frame_idx",
                             signed=(True, False, False, False))
        emon = StreamMonitor(dut, "e_valid", "e_out", signed=(False,))
        for a, b in zip(wide, s16):
            dut.in_wide.value = a & ((1 << W) - 1)
            dut.in_s16.value = b & 0xFFFF
            dut.in_valid.value = 1
            await ClockCycles(dut.clk, 1)
            dut.in_valid.value = 0
            await ClockCycles(dut.clk, GAP)
        await ClockCycles(dut.clk, 8)
        fmon.stop(); emon.stop()
        vec.assert_equal(f"{case}.frames", fmon.column(0), exp_fr)
        vec.assert_equal(f"{case}.energy", emon.column(0), exp_e)
        firsts = [i for i, r in enumerate(fmon.items) if r[1]]
        lasts = [i for i, r in enumerate(fmon.items) if r[2]]
        assert firsts == list(range(0, len(exp_fr), params.FRAME_LEN)), "frame_first salah"
        assert lasts == [i + params.FRAME_LEN - 1 for i in firsts], "frame_last salah"
        idx = sorted({r[3] for r in fmon.items})
        assert idx == list(range(len(exp_fr) // params.FRAME_LEN)), f"frame_idx {idx}"


def test_runner():
    from tb.common.runner import run
    run("frame_buf", "test_frame_buf")


if __name__ == "__main__":
    test_runner()
