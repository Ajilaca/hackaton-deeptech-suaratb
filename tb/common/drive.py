"""Pengemudi stream bertingkat untuk modul DSP Fase 3 (data + bendera sidebands per siklus)."""
from __future__ import annotations
from cocotb.triggers import RisingEdge, ClockCycles


async def send(dut, words, data: str, bits: int, first: str | None = None, last: str | None = None,
               idx: str | None = None, valid: str = "in_valid", gap: int = 0):
    """Kirim `words` satu per siklus (atau tiap gap+1 siklus). first: bendera pada kata ke-0; last: pada kata terakhir;
    idx: nomor urut kata. Seluruh sinyal valid diturunkan setelah kata terakhir."""
    n = len(words)
    for i, w in enumerate(words):
        getattr(dut, data).value = w & ((1 << bits) - 1)
        if first:
            getattr(dut, first).value = int(i == 0)
        if last:
            getattr(dut, last).value = int(i == n - 1)
        if idx:
            getattr(dut, idx).value = i
        getattr(dut, valid).value = 1
        await RisingEdge(dut.clk)
        if gap:
            getattr(dut, valid).value = 0
            await ClockCycles(dut.clk, gap)
    getattr(dut, valid).value = 0
    if first:
        getattr(dut, first).value = 0
    if last:
        getattr(dut, last).value = 0


async def wait_for(dut, cond, max_cycles: int, what: str = "kondisi"):
    """Tunggu hingga cond() benar (diperiksa tiap siklus)."""
    for _ in range(max_cycles):
        await RisingEdge(dut.clk)
        if cond():
            return
    raise AssertionError(f"timeout menunggu {what} ({max_cycles} siklus)")
