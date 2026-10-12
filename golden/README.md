# golden: spesifikasi bit-exact

Model numpy yang menjadi spesifikasi hardware. Bila RTL dan golden berbeda, salah satunya diperbaiki dan dicatat di
`docs/vault/keputusan.md`. Semua fungsi murni (tanpa state global).

| Jalur | Peran |
|---|---|
| `params.py` | Sumber tunggal parameter beku (lebar FFT, skala, ambang detektor, deskriptor layer CNN); menghasilkan paket SystemVerilog |
| `paths.py` | Lokasi data (`$SUARATB_DATA`) |
| `blocks/` | Implementasi tiap blok pengolahan (lihat README di dalamnya) |
| `gen/` | Pembangkit: tabel `.mem`, paket konfigurasi dan ROM CNN, vektor uji sintetis, audio sintetis, penulis `.mem` |
| `analysis/` | Anggaran memori/pengali/siklus dan kalibrasi lebar FFT, offset fitur, detektor (Fase 0) |
| `tests/` | 67 tes golden (blok, CNN, jadwal FFT, pengalamatan DSP, front-end, parameter, tabel) |
| `tables/` | Tabel dan paket hasil `gen/gen_tables.py` dan `gen/gen_cnn_rom.py` |

Keterangan angka: nilai data-dependen (offset fitur, ambang) berasal dari audio nyata perekam (K-14) dan bersifat provisional
untuk INMP441. Angka sumber daya nyata hanya berasal dari laporan alat (`docs/vault/sintesis.md`).
