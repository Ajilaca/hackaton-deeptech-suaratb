"""Utilitas cocotb 2.x bersama: clock/reset, driver dan monitor stream (valid saja atau valid/ready)."""
from __future__ import annotations
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, ClockCycles


async def start_clock(dut, period_ns: float = 10.0, name: str = "clk"):
    cocotb.start_soon(Clock(getattr(dut, name), period_ns, unit="ns").start())


async def reset(dut, cycles: int = 4, rst: str = "rst_n", active_low: bool = True):
    r = getattr(dut, rst)
    r.value = 0 if active_low else 1
    await ClockCycles(dut.clk, cycles)
    r.value = 1 if active_low else 0
    await RisingEdge(dut.clk)


def sval(sig, bits: int | None = None) -> int:
    """Nilai bertanda dari sinyal (X/Z -> error yang jelas)."""
    v = sig.value
    if not v.is_resolvable:
        raise AssertionError(f"{sig._name} bernilai tak terdefinisi: {v}")
    return v.to_signed() if hasattr(v, "to_signed") else int(v)


def uval(sig) -> int:
    v = sig.value
    if not v.is_resolvable:
        raise AssertionError(f"{sig._name} bernilai tak terdefinisi: {v}")
    return v.to_unsigned() if hasattr(v, "to_unsigned") else int(v)


class StreamDriver:
    """Mengirim kata ke antarmuka valid(/ready). gap = jumlah siklus idle antar kata."""

    def __init__(self, dut, valid: str, data: str, ready: str | None = None, gap: int = 0):
        self.dut, self.valid, self.data, self.ready, self.gap = dut, getattr(dut, valid), getattr(dut, data), \
            (getattr(dut, ready) if ready else None), gap
        self.valid.value = 0

    async def send(self, words, mask_bits: int | None = None):
        for w in words:
            self.data.value = w & ((1 << mask_bits) - 1) if mask_bits else w
            self.valid.value = 1
            await RisingEdge(self.dut.clk)
            if self.ready is not None:
                while not int(self.ready.value):
                    await RisingEdge(self.dut.clk)
            self.valid.value = 0
            if self.gap:
                await ClockCycles(self.dut.clk, self.gap)


class StreamMonitor:
    """Mengumpulkan nilai data saat valid (dan ready bila ada) aktif pada tepi naik clk."""

    def __init__(self, dut, valid: str, *datas: str, signed: tuple[bool, ...] | None = None, ready: str | None = None):
        self.dut = dut
        self.valid = getattr(dut, valid)
        self.datas = [getattr(dut, d) for d in datas]
        self.signed = signed or tuple(True for _ in datas)
        self.ready = getattr(dut, ready) if ready else None
        self.items: list[tuple[int, ...]] = []
        self._task = cocotb.start_soon(self._run())

    async def _run(self):
        while True:
            await RisingEdge(self.dut.clk)
            if self.valid.value.is_resolvable and int(self.valid.value) and \
                    (self.ready is None or int(self.ready.value)):
                row = tuple(sval(d) if s else uval(d) for d, s in zip(self.datas, self.signed))
                self.items.append(row)

    def column(self, i: int = 0) -> list[int]:
        return [r[i] for r in self.items]

    def stop(self):
        self._task.cancel()
