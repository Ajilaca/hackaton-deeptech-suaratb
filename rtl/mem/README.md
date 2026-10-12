# rtl/mem

Tabel `.mem` yang dibaca RTL (`$readmemh`), salinan dari `golden/tables` (target `tables` di `Makefile`). Jalur dibaca relatif
terhadap direktori kerja simulator/sintesis.

| Berkas | Dipakai oleh |
|---|---|
| `window_hamming.mem` | `rtl/dsp/window.sv` |
| `twiddle_cos.mem`, `twiddle_sin.mem` | `rtl/dsp/fft512.sv` |
| `mel_rom.mem` | `rtl/dsp/mel.sv` |
| `log_lut.mem` | `rtl/dsp/log_q.sv` |

Bobot CNN tidak ada di sini: dibangkitkan per model oleh `golden/gen/gen_cnn_rom.py` ke `tb/vectors/cnn` atau direktori sintesis.
