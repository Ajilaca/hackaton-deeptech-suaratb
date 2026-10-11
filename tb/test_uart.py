"""cocotb: uart_rx dan uart_tx (loopback lewat model host) + bridge CSR 'W'/'R'."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import cocotb
from cocotb.triggers import ClockCycles

from tb.common.cocotb_utils import start_clock, reset
from tb.common.uart_model import uart_send, UartMonitor

CPB = 16


@cocotb.test()
async def test_csr_rw(dut):
    await start_clock(dut)
    dut.rx.value = 1
    dut.csr_rdata.value = 0
    dut.res_valid.value = 0
    await reset(dut)
    mon = UartMonitor(dut, CPB, pin="tx")
    # tulis CSR[0x12] = 0xAB
    ev = []

    async def watch():
        while True:
            await ClockCycles(dut.clk, 1)
            if dut.csr_we.value.is_resolvable and int(dut.csr_we.value):
                ev.append((int(dut.csr_waddr.value), int(dut.csr_wdata.value)))
    w = cocotb.start_soon(watch())
    await uart_send(dut, b"W\x12\xab", CPB, pin="rx")
    await ClockCycles(dut.clk, 20)
    assert ev == [(0x12, 0xAB)], f"tulis salah: {ev}"
    # baca CSR[0x34]: rdata dipasang kombinasional = 0x5A bila raddr == 0x34
    async def rom():
        while True:
            await ClockCycles(dut.clk, 1)
            ra = dut.csr_raddr.value
            dut.csr_rdata.value = 0x5A if (ra.is_resolvable and int(ra) == 0x34) else 0
    r = cocotb.start_soon(rom())
    await uart_send(dut, b"R\x34", CPB, pin="rx")
    await ClockCycles(dut.clk, CPB * 14)
    assert mon.bytes == [0x5A], f"balasan R salah: {mon.bytes}"
    # hasil inferensi: 'L' lo hi seq
    dut.res_logit.value = (-3) & 0xFFFF
    dut.res_seq.value = 7
    dut.res_valid.value = 1
    await ClockCycles(dut.clk, 1)
    dut.res_valid.value = 0
    await ClockCycles(dut.clk, CPB * 10 * 4 + 40)
    assert mon.bytes[1:] == [0x4C, 0xFD, 0xFF, 7], f"bingkai L salah: {mon.bytes}"
    w.cancel(); r.cancel(); mon.stop()


@cocotb.test()
async def test_dataset_stream(dut):
    """Perintah 'D': 8240 sampel little-endian keluar sebagai ds_valid/ds_sample, ds_start sekali."""
    await start_clock(dut)
    dut.rx.value = 1
    dut.res_valid.value = 0
    await reset(dut)
    got, starts = [], []

    async def watch():
        while True:
            await ClockCycles(dut.clk, 1)
            if dut.ds_start.value.is_resolvable and int(dut.ds_start.value):
                starts.append(1)
            if dut.ds_valid.value.is_resolvable and int(dut.ds_valid.value):
                got.append(dut.ds_sample.value.to_signed())
    w = cocotb.start_soon(watch())
    import numpy as np
    from golden import params
    n = params.CLIP_SAMPLES_PADDED
    x = (np.arange(n) * 37 - 20000).astype(np.int16)
    await uart_send(dut, b"D" + x.astype("<i2").tobytes(), CPB, pin="rx")
    await ClockCycles(dut.clk, 60)
    w.cancel()
    assert len(starts) == 1
    assert got == x.tolist(), f"{len(got)} sampel; selisih pertama di {next((i for i,(a,b) in enumerate(zip(got,x.tolist())) if a!=b), None)}"


def test_runner():
    from tb.common.runner import run
    run("uart_bridge", "test_uart", parameters={"CLK_PER_BIT": CPB})


if __name__ == "__main__":
    test_runner()
