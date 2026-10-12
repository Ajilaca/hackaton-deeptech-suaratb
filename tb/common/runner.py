"""Pembangun + penjalan cocotb (Verilator atau Icarus) memakai cocotb_tools.runner.

Dipakai oleh tb/test_<modul>.py. Direktori build ada di target/build/<modul> (symlink ke $SUARATB_DATA/build).
HANYA dipanggil di Fase T (make test); Fase 1-5 tidak menjalankannya.
"""
from __future__ import annotations
import os
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]


def mem_param(name: str) -> str:
    """Nilai parameter string untuk berkas .mem di rtl/mem (path absolut, diberi tanda kutip untuk -G Verilator)."""
    return '"' + str(ROOT / "rtl" / "mem" / name) + '"'


# Berkas yang tidak didukung Icarus 12 (properti SVA bersamaan) dan tidak dipakai modul mana pun saat simulasi.
ICARUS_SKIP = ("sva_stream.sv",)


def rtl_files(extra: list[str] | None = None, sim: str | None = None) -> list[str]:
    sim = sim or os.environ.get("SIM", "verilator")
    out = []
    for ln in (ROOT / "rtl" / "files.f").read_text().splitlines():
        ln = ln.strip()
        if ln and not ln.startswith("//"):
            if sim == "icarus" and ln.endswith(ICARUS_SKIP):
                continue
            out.append(str(ROOT / ln))
    return out + (extra or [])


def run(toplevel: str, test_module: str, parameters: dict | None = None, simulator: str | None = None,
        waves: bool = False, sources: list[str] | None = None, build_args: list | None = None,
        cwd: str | None = None, tag: str = ""):
    from cocotb_tools.runner import get_runner
    sim = simulator or os.environ.get("SIM", "verilator")
    runner = get_runner(sim)
    build_dir = ROOT / "target" / "build" / f"{toplevel}{tag}_{sim}"
    args = list(build_args or [])
    parameters = dict(parameters or {})
    if sim == "icarus":
        # Icarus 12 gagal mengikat parameter string lewat -P; jalur .mem relatif terhadap cwd = ROOT sudah benar (default).
        parameters = {k: v for k, v in parameters.items() if not k.startswith("INIT_")}
    if sim == "verilator":
        args += ["-Wno-fatal", "--assert"]
    runner.build(sources=sources or rtl_files(sim=sim), hdl_toplevel=toplevel, parameters=parameters,
                 build_dir=str(build_dir), waves=waves, always=True, build_args=args, timescale=("1ns", "1ps"),
                 includes=[str(ROOT / "rtl" / "common")])
    if "." in test_module:
        mod = test_module
    else:
        hit = sorted((ROOT / "tb").glob(f"*/{test_module}.py"))
        mod = f"tb.{hit[0].parent.name}.{test_module}"
    runner.test(hdl_toplevel=toplevel, test_module=mod, test_dir=cwd or str(ROOT), waves=waves,
                build_dir=str(build_dir), results_xml=str(build_dir / "results.xml"))
