# tb: testbench cocotb

Satu `test_<modul>.py` per modul RTL, membandingkan keluaran RTL dengan vektor/golden secara bit-exact. Simulator: Icarus
(`SIM=icarus`) atau Verilator. Regresi penuh hanya pada Fase T (`make test FASE_T=1`).

| Berkas | Modul |
|---|---|
| `test_i2s_rx.py`, `test_preemph.py`, `test_frame_buf.py`, `test_cough_det.py` | Front-end dan detektor |
| `test_window.py`, `test_fft512.py`, `test_power.py`, `test_mel.py`, `test_log_q.py`, `test_quant.py`, `test_feat_buf.py`, `test_dsp_chain.py` | DSP |
| `test_requant.py`, `test_gap.py`, `test_cnn_top.py`, `test_cnn_sweep.py`, `test_cnn_trained.py` | CNN (bobot acak, sapuan P, bobot terlatih) |
| `test_uart.py`, `test_top.py` | UART dan sistem ujung ke ujung (dataset dan mikrofon) |

`test_cnn_trained.py` dan `test_top.py` dapat memakai model terlatih dari `$SUARATB_DATA/runs/<nama>` lewat variabel lingkungan
(`CNN_TRAINED_MODEL`, `TOP_MODEL`); tanpa itu dipakai model acak terkalibrasi atau `model_v2`.
