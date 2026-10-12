"""cocotb: suaratb_top ujung ke ujung: UART dataset -> skor; mikrofon I2S -> detektor -> skor. Bandingkan dengan golden."""
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import cocotb
import numpy as np
from cocotb.triggers import ClockCycles, FallingEdge

from golden import params
from golden.blocks import cnn as C
from golden.blocks import frontend as F
from golden.blocks.cough_det import hop_energy, cough_det
from golden.blocks.preemph import preemph
from tb.common import vec
from tb.common.cocotb_utils import start_clock, reset
from tb.common.golden_model import random_calibrated_model
from golden import paths
from tb.common.uart_model import uart_send, UartMonitor

MODEL_NAME = os.environ.get("TOP_MODEL", "")     # "" = model acak terkalibrasi; selain itu runs/<nama>/qmodel.npz
CPB = 24            # clk per bit UART (2 byte/sampel = 480 siklus >= 404 yang dibutuhkan frame_buf)
CPBCLK = 8          # clk per BCLK I2S (512 siklus/sampel >= 404)


def get_model():
    if MODEL_NAME:
        return C.load_qmodel(paths.runs_dir() / MODEL_NAME / "qmodel.npz")
    return random_calibrated_model()


async def i2s_mic(dut, words):
    """Model INMP441: MSB keluar pada tepi turun BCLK berikutnya setelah WS berubah; slot kanan = 0."""
    prev_ws = None
    bits, idx = [], 0
    dut.sd.value = 0
    while True:
        await FallingEdge(dut.bclk)
        ws = int(dut.ws.value)
        if prev_ws is None or ws != prev_ws:
            if ws == 0 and idx < len(words):
                w = words[idx] & 0xFFFFFF
                bits = [(w >> (23 - i)) & 1 for i in range(24)]
                idx += 1
            else:
                bits = []
        prev_ws = ws
        dut.sd.value = bits.pop(0) if bits else 0


async def wait_bytes(dut, mon, n, max_cycles, what):
    for _ in range(max_cycles // 1000):
        await ClockCycles(dut.clk, 1000)
        if len(mon.bytes) >= n:
            return
    raise AssertionError(f"timeout menunggu {what}: {len(mon.bytes)} byte")


@cocotb.test()
async def test_dataset_and_mic(dut):
    await start_clock(dut)
    dut.uart_rx.value = 1
    dut.sd.value = 0
    await reset(dut)
    mon = UartMonitor(dut, CPB)
    n_res = 0
    # ---- mode dataset: dua kasus ----------------------------------------------------------------------------
    await uart_send(dut, b"W\x00\x00", CPB)
    for case in ("cough", "noise3"):
        x = np.array(vec.load("frontend", case, "in", 16), dtype=np.int16)
        exp = vec.load("cnn", case, "out_logit", 16)[0]
        if MODEL_NAME:
            exp = int(C.cnn_forward(F.to_cnn_input(F.features_q(x.astype(np.int64))), get_model())[0])
        await uart_send(dut, b"D" + x.astype("<i2").tobytes(), CPB)
        await wait_bytes(dut, mon, 4 * (n_res + 1), 3_000_000, f"{case}: hasil")
        fr = mon.bytes[4 * n_res:4 * n_res + 4]
        got = int.from_bytes(bytes(fr[1:3]), "little", signed=True)
        assert fr[0] == 0x4C and got == exp, f"{case}: bingkai {fr} logit {got} != golden {exp}"
        assert fr[3] == n_res, f"{case}: seq {fr[3]} != {n_res}"
        n_res += 1
        dut._log.info("dataset %s: logit=%d OK", case, got)
    # ---- baca CSR ---------------------------------------------------------------------------------------
    await uart_send(dut, b"R\x07", CPB)
    await wait_bytes(dut, mon, 4 * n_res + 1, 200_000, "balasan VERSION")
    assert mon.bytes[-1] == 0x01, f"VERSION {mon.bytes[-1]}"
    # ---- mode mikrofon: klip batuk + diam --------------------------------------------------------------------
    x = np.array(vec.load("frontend", "cough", "in", 16), dtype=np.int64)
    x_ext = np.concatenate([x, np.zeros(4000, dtype=np.int64)])
    ev = cough_det(hop_energy(preemph(x_ext)))
    assert ev, "golden tidak mendeteksi batuk"
    seg = F.segment(F.features_q(x_ext), ev[0].start)
    model = get_model()
    exp = int(C.cnn_forward(F.to_cnn_input(seg), model)[0])
    words = [(int(s) & 0xFFFF) << 8 for s in x_ext]
    await uart_send(dut, b"W\x00\x01", CPB)                 # mode mikrofon (i2s_rx aktif)
    mic = cocotb.start_soon(i2s_mic(dut, words))
    before = len(mon.bytes)
    await wait_bytes(dut, mon, before + 4, 12_000_000, "hasil mikrofon")
    mic.cancel()
    fr = mon.bytes[before:before + 4]
    got = int.from_bytes(bytes(fr[1:3]), "little", signed=True)
    assert fr[0] == 0x4C and got == exp, f"mikrofon: bingkai {fr} logit {got} != golden {exp} (onset golden {ev[0].onset})"
    dut._log.info("mikrofon: onset=%d logit=%d OK", ev[0].onset, got)
    mon.stop()


@cocotb.test()
async def test_mic(dut):
    """Mode mikrofon saja (cepat): klip batuk 8240 sampel tanpa ekor; catat onset/start detektor RTL."""
    await start_clock(dut)
    dut.uart_rx.value = 1
    dut.sd.value = 0
    await reset(dut)
    mon = UartMonitor(dut, CPB)
    x = np.array(vec.load("frontend", "cough", "in", 16), dtype=np.int64)
    ev = cough_det(hop_energy(preemph(x)))
    assert ev, "golden tidak mendeteksi batuk"
    exp = int(C.cnn_forward(F.to_cnn_input(F.segment(F.features_q(x), ev[0].start)), get_model())[0])
    words = [(int(v) & 0xFFFF) << 8 for v in x]
    await uart_send(dut, b"W\x00\x01", CPB)
    mic = cocotb.start_soon(i2s_mic(dut, words))
    await wait_bytes(dut, mon, 4, 12_000_000, "hasil mikrofon")
    mic.cancel()
    fr = mon.bytes[0:4]
    got = int.from_bytes(bytes(fr[1:3]), "little", signed=True)
    dut._log.info("RTL onset=%s start=%s end=%s; golden onset=%d start=%d; logit RTL=%d golden=%d",
                  dut.det_onset.value, dut.det_start.value, dut.det_end.value, ev[0].onset, ev[0].start, got, exp)
    assert got == exp, f"mikrofon: logit {got} != golden {exp}"
    mon.stop()


def test_runner():
    from tb.common.runner import run
    if MODEL_NAME:
        from golden.gen_cnn_rom import write_roms
        tmp = pathlib.Path(__file__).resolve().parents[1] / "target" / "build" / f"top_{MODEL_NAME}"
        write_roms(get_model(), tmp / "tb" / "vectors" / "cnn")
        (tmp / "rtl").mkdir(parents=True, exist_ok=True)
        link = tmp / "rtl" / "mem"
        if not link.exists():
            link.symlink_to(pathlib.Path(__file__).resolve().parents[1] / "rtl" / "mem")    # ROM DSP (jalur relatif terhadap cwd)
        run("suaratb_top", "test_top", parameters={"CLK_PER_BIT": CPB, "CLK_PER_BCLK": CPBCLK}, cwd=str(tmp), tag=f"_{MODEL_NAME}")
        return
    run("suaratb_top", "test_top", parameters={"CLK_PER_BIT": CPB, "CLK_PER_BCLK": CPBCLK})


if __name__ == "__main__":
    test_runner()
