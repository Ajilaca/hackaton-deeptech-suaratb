"""Model UART host (8N1) untuk cocotb: kirim byte ke pin rx perangkat dan terima dari pin tx."""
from __future__ import annotations
import cocotb
from cocotb.triggers import ClockCycles, FallingEdge


async def uart_send(dut, data, cpb: int, pin: str = "uart_rx"):
    line = getattr(dut, pin)
    for b in bytes(data):
        line.value = 0                                   # start
        await ClockCycles(dut.clk, cpb)
        for i in range(8):
            line.value = (b >> i) & 1
            await ClockCycles(dut.clk, cpb)
        line.value = 1                                   # stop
        await ClockCycles(dut.clk, cpb)


class UartMonitor:
    """Menerima byte dari pin tx perangkat; hasil di self.bytes."""

    def __init__(self, dut, cpb: int, pin: str = "uart_tx"):
        self.dut, self.cpb, self.line = dut, cpb, getattr(dut, pin)
        self.bytes: list[int] = []
        self._task = cocotb.start_soon(self._run())

    async def _run(self):
        while True:
            await FallingEdge(self.line)
            await ClockCycles(self.dut.clk, self.cpb // 2)
            if int(self.line.value) != 0:
                continue
            v = 0
            for i in range(8):
                await ClockCycles(self.dut.clk, self.cpb)
                v |= int(self.line.value) << i
            await ClockCycles(self.dut.clk, self.cpb)    # stop
            self.bytes.append(v)

    def stop(self):
        self._task.cancel()
