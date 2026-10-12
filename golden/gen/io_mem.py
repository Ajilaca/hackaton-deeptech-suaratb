"""Penulisan berkas .mem (format $readmemh: satu kata hex per baris, komplemen-dua)."""
from __future__ import annotations
import pathlib
import numpy as np


def write_mem(path, values, bits: int, header: str = "") -> None:
    v = np.asarray(values, dtype=np.int64).ravel()
    lim_lo, lim_hi = -(1 << (bits - 1)), (1 << bits) - 1
    assert v.min() >= lim_lo and v.max() <= lim_hi, f"nilai di luar {bits} bit untuk {path}"
    digits = (bits + 3) // 4
    mask = (1 << bits) - 1
    lines = [f"// {header}"] if header else []
    lines += [format(int(x) & mask, f"0{digits}x") for x in v]
    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(lines) + "\n")


def read_mem(path, bits: int, signed: bool = True) -> np.ndarray:
    out = []
    for ln in pathlib.Path(path).read_text().splitlines():
        ln = ln.strip()
        if not ln or ln.startswith("//"):
            continue
        x = int(ln, 16)
        if signed and x >= 1 << (bits - 1):
            x -= 1 << bits
        out.append(x)
    return np.array(out, dtype=np.int64)
