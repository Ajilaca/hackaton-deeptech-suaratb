"""Sapuan P (8 dan 32): cnn_top harus tetap bit-exact terhadap golden. Membuat paket konfigurasi dan ROM per-P di folder
sementara dan menjalankan test_cnn_top di sana (cwd sementara agar jalur .mem relatif menunjuk ROM P tersebut)."""
import os
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _run_for_p(p: int):
    from golden import params
    from golden.blocks import cnn as C
    from golden.gen_cnn_rom import cfg_package, write_roms
    from golden.blocks import frontend as F
    from golden.gen_vectors import test_signals
    from tb.common.runner import run, rtl_files
    tmp = ROOT / "target" / "build" / f"sweep_P{p}"
    (tmp / "tb" / "vectors" / "cnn").mkdir(parents=True, exist_ok=True)
    (tmp / "rtl").mkdir(parents=True, exist_ok=True)
    (tmp / "rtl" / "cnn_cfg_pkg.sv").write_text(cfg_package(p))
    model = C.random_model(seed=123)
    feats = [F.to_cnn_input(F.features_q(x)) for x in test_signals().values()]
    C.calibrate_requant(model, feats)
    write_roms(model, tmp / "tb" / "vectors" / "cnn", p)
    src = [str(tmp / "rtl" / "cnn_cfg_pkg.sv") if f.endswith("cnn_cfg_pkg.sv") else f for f in rtl_files()]
    os.environ["CNN_CASES"] = "cough,noise100"
    run("cnn_top", "test_cnn_top", sources=src, cwd=str(tmp), tag=f"_P{p}")


def test_p8():
    _run_for_p(8)


def test_p32():
    _run_for_p(32)
