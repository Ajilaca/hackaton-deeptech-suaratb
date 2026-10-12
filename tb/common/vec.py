"""Pemuat vektor uji (.mem hex komplemen-dua) dan pembanding bit-exact.

Format: satu kata hex per baris; baris yang diawali '//' adalah komentar; lebar kata di meta.json tiap blok.
Letak : tb/vectors/<blok>/<kasus>_<sufiks>.mem  (dihasilkan golden/gen/gen_vectors.py; sintetis).
"""
from __future__ import annotations
import pathlib
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
VEC = ROOT / "tb" / "vectors"


def vector_path(block: str, case: str, suffix: str) -> pathlib.Path:
    return VEC / block / f"{case}_{suffix}.mem"


def read_mem(path, bits: int, signed: bool = True) -> list[int]:
    out = []
    for ln in pathlib.Path(path).read_text().splitlines():
        ln = ln.strip()
        if not ln or ln.startswith("//"):
            continue
        v = int(ln, 16)
        if signed and v >= 1 << (bits - 1):
            v -= 1 << bits
        out.append(v)
    return out


def load(block: str, case: str, suffix: str, bits: int, signed: bool = True) -> list[int]:
    return read_mem(vector_path(block, case, suffix), bits, signed)


def to_signed(v: int, bits: int) -> int:
    v &= (1 << bits) - 1
    return v - (1 << bits) if v >> (bits - 1) else v


def compare(name: str, got, exp, max_report: int = 8) -> list[str]:
    """Daftar pesan selisih (kosong = identik bit-exact)."""
    got, exp = list(got), list(exp)
    msgs = []
    if len(got) != len(exp):
        msgs.append(f"{name}: panjang {len(got)} != {len(exp)}")
    n = min(len(got), len(exp))
    bad = [i for i in range(n) if got[i] != exp[i]]
    for i in bad[:max_report]:
        msgs.append(f"{name}[{i}]: got {got[i]} exp {exp[i]}")
    if len(bad) > max_report:
        msgs.append(f"{name}: {len(bad)} selisih total")
    return msgs


def assert_equal(name: str, got, exp) -> None:
    msgs = compare(name, got, exp)
    assert not msgs, "\n".join(msgs)


def cases(block: str) -> list[str]:
    """Nama kasus yang tersedia untuk sebuah blok (dari berkas *_in*.mem)."""
    names = {p.name.split("_")[0] for p in (VEC / block).glob("*_in*.mem")}
    return sorted(names)


def np_array(x) -> np.ndarray:
    return np.asarray(x, dtype=np.int64)
