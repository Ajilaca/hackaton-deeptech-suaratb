"""cocotb: window vs golden (400 sampel -> 512 sampel W bit). Dijalankan HANYA di Fase T."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import cocotb
from cocotb.triggers import ClockCycles

from golden import params
from tb.common import vec
from tb.common.cocotb_utils import start_clock, reset, StreamMonitor
from tb.common.drive import send

W = params.FFT_W
IN_W = 16 + params.PREEMPH_FRAC


@cocotb.test()
async def test_vectors(dut):
    await start_clock(dut)
    dut.in_valid.value = 0
    dut.in_data.value = 0
    dut.in_first.value = 0
    await reset(dut)
    for case in vec.cases("window"):
        fr = vec.load("window", case, "in", IN_W)
        exp = vec.load("window", case, "out", W)
        assert len(fr) == params.FRAME_LEN and len(exp) == params.NFFT
        mon = StreamMonitor(dut, "out_valid", "out_data", "out_first", "out_last", signed=(True, False, False))
        await send(dut, fr, "in_data", IN_W, first="in_first")
        await ClockCycles(dut.clk, params.NFFT + 16)
        mon.stop()
        vec.assert_equal(f"{case}.out", mon.column(0), exp)
        assert [i for i, r in enumerate(mon.items) if r[1]] == [0], f"{case}: out_first salah"
        assert [i for i, r in enumerate(mon.items) if r[2]] == [params.NFFT - 1], f"{case}: out_last salah"


def test_runner():
    from tb.common.runner import run, mem_param
    run("window", "test_window", parameters={"INIT_FILE": mem_param("window_hamming.mem")})


if __name__ == "__main__":
    test_runner()
