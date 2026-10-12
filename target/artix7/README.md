# target/artix7

| Berkas | Peran |
|---|---|
| `prep_synth.py` | Menyiapkan berkas sintesis untuk satu P: paket konfigurasi CNN dan ROM bobot (model terlatih bila ada) |
| `synth.tcl` | Sintesis + implementasi Vivado XC7A35T-1 mode OOC; menulis laporan utilisasi, timing, daya |
| `run_sweep.sh` | Sapuan beberapa nilai P (default 16, 8, 32) |

Keluaran besar ditulis ke `$SUARATB_DATA/build`, bukan ke repo.
