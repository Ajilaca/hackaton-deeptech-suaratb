# golden: spesifikasi bit-exact

Model numpy yang menjadi spesifikasi hardware. Bila RTL dan golden berbeda, salah satunya diperbaiki dan dicatat di
`docs/vault/keputusan.md`. Semua fungsi murni (tanpa state global).

| Berkas | Peran |
|---|---|
| `params.py` | Sumber tunggal parameter beku (lebar FFT, skala, ambang detektor, deskriptor layer CNN); menghasilkan paket SystemVerilog |
| `gen_tables.py` | Tabel `.mem` (window, twiddle, mel, log LUT) dan `suaratb_pkg.sv` |
| `gen_cnn_rom.py` | Paket konfigurasi CNN dan ROM bobot terkemas untuk P lane |
| `gen_vectors.py` | Vektor uji sintetis per blok (masukan/keluaran) untuk `tb/vectors` |
| `budget.py` | Anggaran memori, pengali, dan siklus analitik; penilaian muat per perangkat |
| `calibrate.py`, `calibrate_real.py` | Penentuan lebar FFT, offset fitur, dan detektor pada audio sintetis dan nyata (Fase 0) |
| `synth.py` | Generator audio sintetis (sinus, derau, impuls) untuk uji dan kalibrasi |
| `io_mem.py`, `paths.py` | Penulisan `.mem` dan lokasi data (`$SUARATB_DATA`) |
| `test_*.py` | 67 tes golden (blok, CNN, jadwal FFT, pengalamatan DSP, front-end, parameter, tabel) |
| `blocks/` | Implementasi tiap blok pengolahan (lihat README di dalamnya) |
| `tables/` | Tabel dan paket hasil `gen_tables.py`/`gen_cnn_rom.py` |

Keterangan angka: nilai data-dependen (offset fitur, ambang) berasal dari audio nyata perekam (K-14) dan bersifat provisional
untuk INMP441. Angka sumber daya nyata hanya berasal dari laporan alat (`docs/vault/sintesis.md`).
