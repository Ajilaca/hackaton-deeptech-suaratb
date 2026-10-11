"""cocotb: dsp_chain (window -> fft -> power -> mel -> log -> quant) vs fitur golden per frame. HANYA di Fase T."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import cocotb
import numpy as np
from cocotb.triggers import ClockCycles

from golden import params
from golden.blocks import frontend as F
from golden.blocks.preemph import preemph
from tb.common import vec
from tb.common.cocotb_utils import start_clock, reset, StreamMonitor
from tb.common.drive import send, wait_for

PF = params.PREEMPH_FRAC
IN_W = 16 + PF
FRAMES = (0, 1, 20, 49)               # frame representatif (awal, tengah, akhir klip)


@cocotb.test()
async def test_frames(dut):
    await start_clock(dut)
    dut.in_valid.value = 0
    dut.in_first.value = 0
    dut.in_data.value = 0
    await reset(dut)
    for case in ("cough", "noise3"):
        x = np.array(vec.load("frontend", case, "in", 16), dtype=np.int64)
        feat = np.array(vec.load("frontend", case, "out_feat", 8)).reshape(params.N_FRAMES, params.N_MEL)
        fr_all = F.frame_signal(preemph(x, frac=PF))
        for f in FRAMES:
            mon = StreamMonitor(dut, "out_valid", "out_q", "out_idx", "out_last", signed=(True, False, False))
            await send(dut, fr_all[f].tolist(), "in_data", IN_W, first="in_first")
            await wait_for(dut, lambda: len(mon.items) >= params.N_MEL, 30000, f"{case}.f{f}: 40 keluaran")
            await ClockCycles(dut.clk, 4)
            mon.stop()
            vec.assert_equal(f"{case}.f{f}.q", mon.column(0), feat[f].tolist())
            vec.assert_equal(f"{case}.f{f}.idx", mon.column(1), list(range(params.N_MEL)))
            assert sum(mon.column(2)) == 1 and mon.column(2)[-1] == 1, f"{case}.f{f}: out_last salah"
            await ClockCycles(dut.clk, 32)     # fft kembali ke LOAD


def test_runner():
    from tb.common.runner import run, mem_param
    run("dsp_chain", "test_dsp_chain", parameters={
        "INIT_WIN": mem_param("window_hamming.mem"), "INIT_COS": mem_param("twiddle_cos.mem"),
        "INIT_SIN": mem_param("twiddle_sin.mem"), "INIT_MEL": mem_param("mel_rom.mem"),
        "INIT_LOG": mem_param("log_lut.mem")})


if __name__ == "__main__":
    test_runner()
