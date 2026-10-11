"""cocotb: fft512 vs golden (urutan alami bin 0..256, jumlah saturasi). Dijalankan HANYA di Fase T."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import cocotb
import numpy as np
from cocotb.triggers import ClockCycles

from golden import params
from golden.blocks.fft512 import fft512 as golden_fft
from tb.common import vec
from tb.common.cocotb_utils import start_clock, reset, StreamMonitor, uval
from tb.common.drive import send, wait_for

W = params.FFT_W


@cocotb.test()
async def test_vectors(dut):
    """Semua kasus berurutan TANPA reset ulang: sekaligus menguji bahwa modul siap kembali setelah done."""
    await start_clock(dut)
    dut.in_valid.value = 0
    dut.in_data.value = 0
    dut.in_first.value = 0
    await reset(dut)
    for case in vec.cases("fft512"):
        xin = vec.load("fft512", case, "in", W)
        exp_re = vec.load("fft512", case, "out_re", W)[: params.N_BINS]
        exp_im = vec.load("fft512", case, "out_im", W)[: params.N_BINS]
        _, _, n_sat = golden_fft(np.array(xin, dtype=np.int64), W=W)
        mon = StreamMonitor(dut, "out_valid", "out_re", "out_im", "out_idx", "out_last",
                            signed=(True, True, False, False))
        await send(dut, xin, "in_data", W, first="in_first")
        await wait_for(dut, lambda: int(dut.done.value) == 1, 20000, f"{case}: done")
        got_sat = uval(dut.n_sat)
        await ClockCycles(dut.clk, 4)
        mon.stop()
        vec.assert_equal(f"{case}.re", mon.column(0), exp_re)
        vec.assert_equal(f"{case}.im", mon.column(1), exp_im)
        vec.assert_equal(f"{case}.idx", mon.column(2), list(range(params.N_BINS)))
        assert mon.column(3).count(1) == 1 and mon.column(3)[-1] == 1, f"{case}: out_last salah"
        assert got_sat == n_sat, f"{case}: n_sat {got_sat} != golden {n_sat}"


def test_runner():
    from tb.common.runner import run, mem_param
    run("fft512", "test_fft512", parameters={"INIT_COS": mem_param("twiddle_cos.mem"),
                                              "INIT_SIN": mem_param("twiddle_sin.mem")})


if __name__ == "__main__":
    test_runner()
