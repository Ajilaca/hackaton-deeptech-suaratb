"""cocotb: cough_det vs golden pada vektor sintetis dan deret energi acak berisi letupan. HANYA di Fase T."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import cocotb
import numpy as np
from cocotb.triggers import ClockCycles, RisingEdge

from golden import params
from golden.blocks.cough_det import cough_det
from tb.common import vec
from tb.common.cocotb_utils import start_clock, reset

GAP = 9


async def run_seq(dut, E):
    """Mengirim deret E; mengembalikan (trig_onsets, starts, ends, seg_done_frames)."""
    await reset(dut)
    trig, starts, ends, done = [], [], [], []
    frame = 0

    async def watch():
        while True:
            await RisingEdge(dut.clk)
            if dut.trig.value.is_resolvable and int(dut.trig.value):
                trig.append(int(dut.onset_frame.value)); starts.append(int(dut.start_frame.value))
                ends.append(int(dut.end_frame.value))
            if dut.seg_done.value.is_resolvable and int(dut.seg_done.value):
                done.append(cur[0])

    cur = [0]
    task = cocotb.start_soon(watch())
    for f, e in enumerate(E):
        cur[0] = f
        dut.e.value = int(e)
        dut.e_valid.value = 1
        await ClockCycles(dut.clk, 1)
        dut.e_valid.value = 0
        await ClockCycles(dut.clk, GAP)
    await ClockCycles(dut.clk, 4)
    task.cancel()
    return trig, starts, ends, done


def expect(E):
    ev = cough_det(E)
    return [e.onset for e in ev], [e.start for e in ev], [e.end for e in ev]


@cocotb.test()
async def test_synthetic_vectors(dut):
    await start_clock(dut)
    dut.e_valid.value = 0
    dut.e.value = 0
    for case in vec.cases("cough_det"):
        E = vec.load("cough_det", case, "in_E", 48, signed=False)
        on, st, en = expect(E)
        trig, starts, ends, _ = await run_seq(dut, E)
        vec.assert_equal(f"{case}.onsets", trig, on)
        vec.assert_equal(f"{case}.starts", starts, st)
        vec.assert_equal(f"{case}.ends", ends, en)


@cocotb.test()
async def test_random_bursts(dut):
    """Deret acak berisi letupan: onset, start, end, dan seg_done identik dengan golden."""
    await start_clock(dut)
    dut.e_valid.value = 0
    rng = np.random.default_rng(5)
    for seed in range(3):
        rng = np.random.default_rng(seed)
        n = 600
        E = rng.integers(2000, 6000, n).astype(np.int64)
        for _ in range(8):
            p = int(rng.integers(10, n - 60))
            E[p: p + int(rng.integers(1, 6))] *= int(rng.integers(20, 400))
        E[int(rng.integers(0, 3))] = 10 ** 6       # tepi awal aliran
        on, st, en = expect(E)
        trig, starts, ends, done = await run_seq(dut, E)
        vec.assert_equal(f"seed{seed}.onsets", trig, on)
        vec.assert_equal(f"seed{seed}.starts", starts, st)
        vec.assert_equal(f"seed{seed}.ends", ends, en)
        vec.assert_equal(f"seed{seed}.seg_done", done, en)


def test_runner():
    from tb.common.runner import run
    run("cough_det", "test_cough_det")


if __name__ == "__main__":
    test_runner()
