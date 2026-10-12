# golden/gen

Pembangkit hasil golden.

| Berkas | Isi |
|---|---|
| `gen_tables.py` | Tabel `.mem` (window, twiddle, mel, log LUT) dan `suaratb_pkg.sv` |
| `gen_cnn_rom.py` | Paket konfigurasi CNN dan ROM bobot terkemas untuk P lane |
| `gen_vectors.py` | Vektor uji sintetis per blok untuk `tb/vectors` |
| `synth.py` | Audio sintetis (sinus, derau, impuls) |
| `io_mem.py` | Penulisan dan pembacaan `.mem` |
