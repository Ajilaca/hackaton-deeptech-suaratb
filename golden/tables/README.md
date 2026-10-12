# golden/tables: tabel dan paket hasil pembangkit

Berkas dihasilkan oleh `golden/gen/gen_tables.py` dan `golden/gen/gen_cnn_rom.py`; **jangan diedit manual**. Salinannya ada di
`rtl/mem` dan `rtl/common`/`rtl/cnn` (lihat `Makefile`, target `tables`).

| Berkas | Isi |
|---|---|
| `suaratb_pkg.sv` | Paket parameter SystemVerilog (lebar data, jumlah frame, mel, deskriptor) |
| `cnn_cfg_pkg.sv` | Konfigurasi layer CNN dan peta ROM untuk P terpilih |
| `window_hamming.mem` | Jendela Hamming Q1.15 |
| `twiddle_cos.mem`, `twiddle_sin.mem` | Twiddle FFT Q1.15 |
| `mel_rom.mem` | Bobot filterbank mel sparse |
| `log_lut.mem` | LUT log2 256 entri |
