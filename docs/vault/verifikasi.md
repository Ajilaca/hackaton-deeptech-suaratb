# Infrastruktur verifikasi (Fase 1)

Aturan: Fase 1-5 hanya MENULIS kode. Simulasi dijalankan di Fase T (`make test FASE_T=1`). `make lint`
(verilator --lint-only) hanya bila pengguna meminta. Alat terpasang: Verilator 5.020, Icarus 12.0, Yosys 0.33, cocotb 2.1.

## Alur
```
golden/ (numpy, bit-exact) --gen_vectors--> tb/vectors/<blok>/<kasus>_<sufiks>.mem --> cocotb tb/test_<modul>.py --> DUT (rtl/)
golden/params.py --gen_tables--> golden/tables/{suaratb_pkg.sv,*.mem} --make tables--> rtl/common/suaratb_pkg.sv, rtl/mem/*.mem
```

## Format vektor
- `.mem`: satu kata hex komplemen-dua per baris, lebar = ceil(bit/4) digit; baris `// ...` komentar (`$readmemh` dan `tb/common/vec.py`).
- Nama: `<kasus>_in*.mem` masukan, `<kasus>_out*.mem` harapan golden; `meta.json` per blok (lebar FFT, daftar kasus, sumber kalibrasi).
- Kasus: `sine1k`, `noise3`, `noise100`, `impulse`, `cough` (sintetis). Vektor CNN memakai bobot ACAK (bukan terlatih).
- Bit-exact: perbandingan `==` integer (`vec.assert_equal`), tanpa toleransi.

## Berkas
| Berkas | Fungsi |
|---|---|
| `Makefile` | `golden`, `tables`, `test-golden`, `lint`, `test` (dijaga FASE_T), `synth-yosys`, `vivado`, `wave` |
| `rtl/files.f` | Urutan kompilasi RTL; tambah modul baru per fase |
| `rtl/common/rom_sync.sv`, `ram_sdp.sv`, `sync_fifo.sv` | Template ROM/RAM/FIFO inferensi (tanpa IP vendor) |
| `rtl/common/sva_stream.sv` | Pemeriksa protokol stream (valid stabil sampai ready, tanpa X) |
| `rtl/common/suaratb_pkg.sv`, `rtl/mem/*.mem` | Dihasilkan dari `golden/params.py`; JANGAN diedit |
| `tb/common/vec.py` | Muat `.mem`, perbandingan bit-exact, daftar kasus |
| `tb/common/cocotb_utils.py` | clock/reset, `StreamDriver`, `StreamMonitor` |
| `tb/common/runner.py` | Build+run cocotb (Verilator/Icarus) ke `target/build/<modul>_<sim>` |
| `tb/test_<modul>.py` | Satu berkas per modul: tes cocotb + `test_runner()` untuk pytest |

## Konvensi RTL
- `logic`, `always_ff`, reset aktif-rendah `rst_n`, antarmuka `valid` (stream tanpa backpressure; didokumentasikan di header) atau `valid/ready`.
- Satu modul per berkas, nama berkas = nama modul; header: fungsi, parameter, antarmuka, latensi, blok golden.
- Parameter default dari `suaratb_pkg` (tidak ada angka di-hardcode); saat sintesis `suaratb_pkg` diimpor, tak ada primitif vendor.
