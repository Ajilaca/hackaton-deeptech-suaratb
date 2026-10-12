# rtl/cnn

Akselerator CNN int8 output-stationary dengan P lane MAC paralel (P dari `cnn_cfg_pkg.sv`: 8/16/32). Pool dilakukan pada
akumulator sebelum requant (bit-exact dengan golden).

| Berkas | Peran |
|---|---|
| `cnn_cfg_pkg.sv` | Konfigurasi layer dan peta ROM (dihasilkan `golden/gen/gen_cnn_rom.py`; jangan diedit) |
| `cnn_top.sv` | Level atas: buffer A/B, ROM bobot/bias/requant, sekuenser, GAP, FC |
| `cnn_ctrl.sv` | Sekuenser layer: token MAC (alamat aktivasi, bobot, bias) dan urutan layer |
| `pe_array.sv` | P lane MAC 8x8 bertanda, akumulator 24 bit, maks antar-posisi pool |
| `requant.sv` | Requantisasi (M0 Q15, shift, saturasi) dan penulisan ke buffer tujuan |
| `gap.sv` | Global average pooling bilangan bulat |

Golden: `golden/blocks/cnn.py`.
