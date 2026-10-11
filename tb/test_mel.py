"""cocotb: mel vs golden (257 bin -> 40 energi band). Dijalankan HANYA di Fase T."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import cocotb
from cocotb.triggers import ClockCycles

from golden import params
from tb.common import vec
from tb.common.cocotb_utils import start_clock, reset, StreamMonitor
from tb.common.drive import send

PW = 2 * params.FFT_W


@cocotb.test()
async def test_vectors(dut):
    """Semua kasus berurutan tanpa reset: akumulator harus kosong lagi setelah tiap frame."""
    await start_clock(dut)
    dut.in_valid.value = 0
    dut.in_first.value = 0
    dut.in_last.value = 0
    dut.in_p.value = 0
    await reset(dut)
    for case in vec.cases("mel"):
        p = vec.load("mel", case, "in", PW, signed=False)
        exp = vec.load("mel", case, "out", params.LOG_IN_BITS, signed=False)
        assert len(p) == params.N_BINS and len(exp) == params.N_MEL
        mon = StreamMonitor(dut, "out_valid", "out_e", "out_idx", "out_last", signed=(False, False, False))
        await send(dut, p, "in_p", PW, first="in_first", last="in_last", gap=3)   # jarak bin 4 siklus
        await ClockCycles(dut.clk, 3 * params.N_MEL + 16)
        mon.stop()
        vec.assert_equal(f"{case}.e", mon.column(0), exp)
        vec.assert_equal(f"{case}.idx", mon.column(1), list(range(params.N_MEL)))
        assert mon.column(2)[-1] == 1 and sum(mon.column(2)) == 1, f"{case}: out_last salah"


def test_runner():
    from tb.common.runner import run, mem_param
    run("mel", "test_mel", parameters={"INIT_FILE": mem_param("mel_rom.mem")})


if __name__ == "__main__":
    test_runner()
