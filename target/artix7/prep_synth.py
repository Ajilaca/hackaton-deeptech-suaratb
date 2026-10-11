"""Siapkan berkas sintesis untuk satu nilai P di <out>: paket konfigurasi CNN + ROM bobot (model terlatih bila ada).

    python target/artix7/prep_synth.py --p 16 --out /data/suaratb/build/synth/P16 [--model <qmodel.npz>]
Keluaran: cnn_cfg_pkg.sv, cnn_wrom.mem, cnn_bias.mem, cnn_rq.mem. Tidak menyentuh rtl/ maupun tb/.
"""
import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from golden import gen_cnn_rom  # noqa: E402
from golden.blocks import cnn as C  # noqa: E402
from tb.common.golden_model import random_calibrated_model  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=None)
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "cnn_cfg_pkg.sv").write_text(gen_cnn_rom.cfg_package(a.p))
    m = C.load_qmodel(a.model) if a.model else random_calibrated_model()
    print(gen_cnn_rom.write_roms(m, out, a.p))


if __name__ == "__main__":
    main()
