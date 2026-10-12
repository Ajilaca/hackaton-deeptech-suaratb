# rtl: SystemVerilog vendor-neutral

Tidak ada IP/primitif vendor di sini; RAM/ROM ditulis dengan template inferensi, pengali dengan `*` dan register bertahap.
Semua ukuran berasal dari `suaratb_pkg.sv` dan `cnn_cfg_pkg.sv` (dihasilkan `golden/`). Satu modul per berkas, nama berkas = nama modul.

| Subdirektori | Isi |
|---|---|
| `common/` | RAM, ROM, FIFO, pemeriksa protokol stream, paket parameter |
| `audio/` | Penerima I2S, pre-emphasis, penyangga frame |
| `detect/` | Detektor batuk |
| `dsp/` | Window, FFT 512, power, mel, log2, kuantisasi, penyangga fitur, rantai DSP |
| `cnn/` | Akselerator CNN output-stationary |
| `sys/` | UART, CSR, jembatan UART, top level |
| `mem/` | Tabel `.mem` hasil golden |

`files.f` memuat daftar berkas urut kompilasi. Aliran data: `i2s_rx` -> `preemph` -> `frame_buf` -> `cough_det`/`dsp_chain` ->
`feat_buf` -> `cnn_top` -> `uart_bridge` (dalam `suaratb_top`).
