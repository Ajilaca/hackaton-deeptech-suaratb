# rtl/dsp

Rantai: window -> fft512 -> power -> mel -> log_q -> quant, dikemas di `dsp_chain`; hasilnya masuk `feat_buf`.

| Berkas | Peran | Golden |
|---|---|---|
| `window.sv` | Hamming 400 titik + zero-pad 512 | `golden/blocks/window.py` |
| `fft512.sv` | FFT 512 radix-2 DIT, 1 butterfly, RAM in-place, skala per tahap | `golden/blocks/fft512.py` |
| `power.sv` | re^2 + im^2 | `golden/blocks/power.py` |
| `mel.sv` | Filterbank mel 40 band sparse | `golden/blocks/mel.py` |
| `log_q.sv` | log2 titik tetap dengan LUT 256 | `golden/blocks/log_q.py` |
| `quant.sv` | Kuantisasi fitur ke int8 | `golden/blocks/quant.py` |
| `dsp_chain.sv` | Rangkaian satu frame | `golden/blocks/frontend.py` |
| `feat_buf.sv` | Penyangga cincin fitur dan pembaca segmen 50 frame | `golden/blocks/frontend.py` |
