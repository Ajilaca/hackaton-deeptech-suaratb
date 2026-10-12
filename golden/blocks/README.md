# golden/blocks: blok pengolahan bit-exact

Satu berkas per blok; masing-masing punya pasangan modul RTL dengan nama yang sama.

| Berkas | Blok | RTL pasangan |
|---|---|---|
| `preemph.py` | Pre-emphasis 0,97 (Q1.15), dua keluaran | `rtl/audio/preemph.sv` |
| `window.py` | Jendela Hamming 400 + zero-pad 512 | `rtl/dsp/window.sv` |
| `fft512.py` | FFT 512 radix-2 DIT, skala per tahap, twiddle Q1.15 | `rtl/dsp/fft512.sv` |
| `power.py` | Spektrum daya re^2 + im^2 | `rtl/dsp/power.sv` |
| `mel.py` | Filterbank mel 40 band (sparse, Q0.12) | `rtl/dsp/mel.sv` |
| `log_q.py` | log2 titik tetap (leading-one + LUT 256) | `rtl/dsp/log_q.sv` |
| `quant.py` | Kuantisasi fitur ke int8 | `rtl/dsp/quant.sv` |
| `frontend.py` | Rangkaian front-end: sampel int16 -> fitur int8 (frame x 40) | `rtl/dsp/dsp_chain.sv` |
| `cough_det.py` | Detektor batuk berbasis energi (EMA, rasio, refractory) | `rtl/detect/cough_det.sv` |
| `cnn.py` | CNN int8 (conv, requant, pool, GAP, FC), `QModel`/`QLayer` | `rtl/cnn/*` |
| `fixed.py` | Utilitas aritmetika titik tetap (geser dibulatkan, saturasi) | - |
