# rtl/audio

| Berkas | Peran | Golden |
|---|---|---|
| `i2s_rx.sv` | Penerima I2S (FPGA master) untuk INMP441, satu kanal; BCLK = clk/CLK_PER_BCLK | - |
| `preemph.sv` | Pre-emphasis y[n] = x[n] - a*x[n-1] (a = 0,97), dua keluaran | `golden/blocks/preemph.py` |
| `frame_buf.sv` | Penyangga sampel, frame 400 setiap hop 160, energi hop untuk detektor | `golden/blocks/frontend.py` |
