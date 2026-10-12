# rtl/sys

| Berkas | Peran |
|---|---|
| `suaratb_top.sv` | Sistem lengkap: sumber sampel mikrofon I2S (mode 1) atau UART dataset (mode 0), DSP, CNN, keluaran skor |
| `uart_rx.sv`, `uart_tx.sv` | UART 8N1 |
| `uart_bridge.sv` | Protokol bingkai host: tulis CSR, baca CSR, kirim dataset, hasil bingkai skor |
| `csr.sv` | File register 8 bit: mode dan clear, status, logit terakhir, SEQ, saturasi FFT, versi, jumlah frame fitur |

Parameter waktu: `CLK_PER_BIT` (UART) dan `CLK_PER_BCLK` (I2S); nilai untuk clock inti 147,456 MHz dicatat di K-22.
