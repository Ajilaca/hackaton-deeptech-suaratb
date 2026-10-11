"""cnn_top dengan bobot TERLATIH (model_v2, di luar repo) vs logit int8 golden. Dilewati bila model tidak ada."""
import os
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cocotb
import pytest
from cocotb.triggers import RisingEdge, ClockCycles

from golden import paths
from tb.common import vec
from tb.common.cocotb_utils import start_clock, reset, sval
from tb.common.drive import wait_for

N_CLIPS = int(os.environ.get("CNN_TRAINED_CLIPS", "8"))


@cocotb.test()
async def test_trained(dut):
    vdir = pathlib.Path(os.environ["CNN_VEC_DIR"])
    exp = vec.read_mem(vdir / "logits.mem", 16)
    await start_clock(dut)
    dut.ld_valid.value = 0
    dut.start.value = 0
    await reset(dut)
    for k in range(N_CLIPS):
        x = vec.read_mem(vdir / f"clip{k:02d}_in.mem", 8)
        assert len(x) == 2000
        for a, v in enumerate(x):
            dut.ld_addr.value = a
            dut.ld_q.value = v & 0xFF
            dut.ld_valid.value = 1
            await RisingEdge(dut.clk)
        dut.ld_valid.value = 0
        dut.start.value = 1
        await RisingEdge(dut.clk)
        dut.start.value = 0
        await wait_for(dut, lambda: int(dut.done.value) == 1, 900000, f"clip{k}: done")
        await ClockCycles(dut.clk, 2)
        got = sval(dut.logit)
        assert got == exp[k], f"clip{k}: logit {got} != golden {exp[k]}"
        dut._log.info("clip%02d: logit=%d OK", k, got)


def test_runner():
    model_dir = paths.runs_dir() / "model_v2"
    if not (model_dir / "qmodel.npz").exists():
        pytest.skip("model_v2 tidak ada")
    from golden.blocks import cnn as C
    from golden.gen_cnn_rom import write_roms
    from tb.common.runner import run
    tmp = ROOT / "target" / "build" / "trained"
    write_roms(C.load_qmodel(model_dir / "qmodel.npz"), tmp / "tb" / "vectors" / "cnn")
    os.environ["CNN_VEC_DIR"] = str(model_dir / "vectors")
    run("cnn_top", "test_cnn_trained", cwd=str(tmp), tag="_trained")
